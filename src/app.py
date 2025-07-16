from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import uvicorn
from typing import Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.CompatibilityLLMAgent import agent  # Import the existing agent instance
from src.Agents.ContextQueryAgent import agent as context_query_agent
from src.Agents.ChatTitleAgent import ChatTitleAgent
from src.Agents.SimpleRouter import route_query
from src.rag.determine_recs import CheckCompatibility
from src.rag.query_engine import QueryEngine
from src.evaluation.feedback_system import FeedbackLogger, run_feedback_loop
from src.data_processing.analyze_compatibility import CompatibilityAnalyzer, get_db_options, get_osi_options

check_compat = CheckCompatibility()
feedback_logger = FeedbackLogger()
analyzer = CompatibilityAnalyzer()

# Instantiate the agent once
# agent = SMARTLLMAgent( # This line is removed as the agent is now imported directly
#     name="CompatibilityLLMAgent",
#     model_client=model_client,
#     system_message=system_message,
#     description="An agent for software compatibility and system integration analysis."
# )

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Analytics Helper Functions ---
def get_analytics_df():
    try:
        all_feedback = feedback_logger.get_all_feedback()
        if not all_feedback:
            return None
        df = pd.DataFrame(all_feedback)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        return df
    except Exception as e:
        print(f"Error getting analytics data: {e}")
        return None

def get_overall_stats():
    df = get_analytics_df()
    if df is None or df.empty:
        return {
            "total_queries": 0,
            "positive_rate": 0,
            "recent_queries": 0,
            "negative_feedback": 0
        }
    total_queries = len(df)
    positive_feedback = len(df[df['feedback_score'] == 1])
    negative_feedback = len(df[df['feedback_score'] == 0])
    positive_rate = (positive_feedback / total_queries * 100) if total_queries > 0 else 0
    recent_queries = len(df[df['timestamp'] >= datetime.now() - timedelta(days=7)])
    return {
        "total_queries": total_queries,
        "positive_rate": positive_rate,
        "recent_queries": recent_queries,
        "negative_feedback": negative_feedback
    }

def get_query_analysis():
    df = get_analytics_df()
    if df is None or df.empty:
        return {
            "avg_length": 0,
            "length_distribution": [0, 0, 0],
            "score_counts": {"positive": 0, "negative": 0, "unrated": 0}
        }
    df['query_length'] = df['query'].str.len()
    avg_length = df['query_length'].mean()
    short = len(df[df['query_length'] < 50])
    medium = len(df[(df['query_length'] >= 50) & (df['query_length'] <= 100)])
    long = len(df[df['query_length'] > 100])
    score_counts = {
        "positive": int(df['feedback_score'].value_counts().get(1, 0)),
        "negative": int(df['feedback_score'].value_counts().get(0, 0)),
        "unrated": int(df['feedback_score'].value_counts().get(-1, 0)),
    }
    return {
        "avg_length": avg_length,
        "length_distribution": [short, medium, long],
        "score_counts": score_counts
    }

def get_os_analysis():
    df = get_analytics_df()
    if df is None or df.empty:
        return {}
    os_counts = df['user_os'].value_counts().to_dict()
    return os_counts

def get_recent_feedback():
    df = get_analytics_df()
    if df is None or df.empty:
        return []
    recent_df = df.sort_values('timestamp', ascending=False).head(10)
    feedback_list = []
    for _, row in recent_df.iterrows():
        feedback_list.append({
            "date": row['timestamp'].strftime('%Y-%m-%d %H:%M'),
            "query": row['query'],
            "score": row['feedback_score'],
            "os": row['user_os']
        })
    return feedback_list

# Initialize database
def init_db():
    """Initialize the SQLite database for user configurations and chat history."""
    try:
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        # Create user_configs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                config_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Create user_chats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")

# Initialize database on startup
init_db()

# --- API Endpoints ---

@app.post("/api/analyze")
async def analyze(request: Request):
    data = await request.json()
    request_text = data.get("request")
    user_os = data.get("os")
    user_database = data.get("database")
    user_web_servers = data.get("webServers", [])
    user_environment = data.get("environment")
    
    if not request_text:
        raise HTTPException(status_code=400, detail="Missing 'request' field.")
    
    # Log the user configuration being used
    print(f"Analyzing with user config: OS={user_os}, DB={user_database}, WebServers={user_web_servers}")
    
    change_requests = check_compat.parse_multiple_change_requests(request_text)
    results = check_compat.analyze_multiple_compatibility(
        change_requests, 
        target_os=user_os,
        user_database=user_database,
        user_web_servers=user_web_servers
    )
    
    # Serialize results
    output = []
    for cr, result in results:
        # Add user context to the result
        user_context = {}
        if user_os:
            user_context["operating_system"] = user_os
        if user_database:
            user_context["database"] = user_database
        if user_web_servers:
            user_context["web_servers"] = user_web_servers
        if user_environment:
            user_context["environment"] = user_environment
        
        output.append({
            "request": f"{cr.action.title()} {cr.software_name} {cr.version or ''}",
            "is_compatible": result.is_compatible,
            "confidence": result.confidence,
            "affected_servers": result.affected_servers,
            "conflicts": result.conflicts,
            "warnings": result.warnings,
            "recommendations": result.recommendations,
            "alternative_versions": result.alternative_versions,
            "user_context": user_context
        })
    
    return {"results": output}

@app.post("/api/feedback")
async def feedback(request: Request):
    data = await request.json()
    score = data.get("score")
    correction = data.get("correction")
    tags = data.get("tags", [])
    notes = data.get("notes")
    query = data.get("query")
    results = data.get("results")
    user_os = data.get("os")
    user_database = data.get("database")
    user_web_servers = data.get("webServers", [])
    user_environment = data.get("environment")
    session_id = data.get("session_id")
    
    metadata = {"tags": tags}
    if correction and correction.strip():
        metadata["correction"] = correction.strip()
    
    # Add user configuration to metadata
    user_config = {}
    if user_os:
        user_config["operating_system"] = user_os
    if user_database:
        user_config["database"] = user_database
    if user_web_servers:
        user_config["web_servers"] = user_web_servers
    if user_environment:
        user_config["environment"] = user_environment
    
    if user_config:
        metadata["user_configuration"] = user_config
    
    # Require correction for negative feedback
    if score == "no" or score == 0 or score == "👎 No":
        if not correction or not correction.strip():
            return JSONResponse(status_code=400, content={"error": "Correction required for negative feedback."})
        score_val = 0
    elif score == "yes" or score == 1 or score == "👍 Yes":
        score_val = 1
    else:
        score_val = -1
    
    try:
        success = feedback_logger.log(
            query=query,
            generated_output=results,
            feedback_score=score_val,
            user_os=user_os,
            session_id=session_id,
            notes=notes,
            metadata=metadata
        )
        if success:
            try:
                run_feedback_loop()
            except Exception as e:
                print(f"Auto-retraining failed: {e}")
        return {"success": success}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/analytics/overall")
async def analytics_overall():
    try:
        return get_overall_stats()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/analytics/query")
async def analytics_query():
    try:
        return get_query_analysis()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/analytics/os")
async def analytics_os():
    try:
        return get_os_analysis()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.api_route("/api/query_intent", methods=["GET", "POST"])
async def query_intent(request: Request):
    if request.method == "GET":
        query = request.query_params.get("query")
    else:  # POST
        try:
            data = await request.json()
            query = data.get("query")
        except Exception:
            query = None
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    result = ""
    async for chunk in context_query_agent.run_stream(task=query):
        if hasattr(chunk, "content"):
            result += str(chunk.content)
        else:
            result += str(chunk)
    return {"result": result}

chat_title_agent = ChatTitleAgent()

@app.post("/api/generate_title")
async def generate_title(request: Request):
    data = await request.json()
    first_message = data.get('message')
    if not first_message:
        return JSONResponse(status_code=400, content={"error": "Message required"})
    title = await chat_title_agent.generate_title(first_message)
    return {"title": title}

@app.post("/api/rag_query")
async def rag_query(request: Request):
    """Process a user query using simple routing - casual or RAG."""
    try:
        data = await request.json()
        user_query = data.get('query')
        user_os = data.get('os')
        user_database = data.get('database')
        user_web_servers = data.get('webServers', [])
        
        if not user_query:
            return JSONResponse(status_code=400, content={"error": "Query required"})
        
        # Step 1: Use simple router to determine query type
        routing_result = await route_query(user_query)
        
        print(f"Query type: {routing_result['type']}")
        
        # Step 2: Handle based on query type
        if routing_result['type'] == 'casual':
            # Return casual response directly
            return {"result": routing_result['response']}
        
        else:  # technical query
            # Use the full RAG pipeline for technical questions
            # Step 2a: Extract intent and entities using ContextQueryAgent
            intent_result = ""
            async for chunk in context_query_agent.run_stream(task=user_query):
                if hasattr(chunk, "content"):
                    intent_result += str(chunk.content)
                else:
                    intent_result += str(chunk)
            
            # Step 2b: Use the RAG system to get relevant context
            query_engine = QueryEngine()
            rag_results = query_engine.query(user_query, top_k=5)
            rag_context = query_engine.format_results_for_llm(rag_results)
            
            # Step 2c: Use CheckCompatibility for structured analysis
            check_compat = CheckCompatibility()
            change_requests = check_compat.parse_multiple_change_requests(user_query)
            
            if change_requests:
                # Analyze compatibility with user config
                results = check_compat.analyze_multiple_compatibility(
                    change_requests,
                    target_os=user_os,
                    user_database=user_database,
                    user_web_servers=user_web_servers
                )
                compatibility_analysis = check_compat.format_multiple_results(results)
            else:
                compatibility_analysis = "No specific software change requests detected in your query."
            
            # Step 2d: Combine all information for a comprehensive response
            comprehensive_response = f"""
**Intent Analysis:**
{intent_result}

**Relevant Context from Your Infrastructure:**
{rag_context}

**Compatibility Analysis:**
{compatibility_analysis}

**Personalized Recommendations:**
Based on your system configuration (OS: {user_os or 'Not specified'}, Database: {user_database or 'Not specified'}, Web Servers: {', '.join(user_web_servers) if user_web_servers else 'Not specified'}):
- Consider the compatibility analysis above
- Review the relevant infrastructure context
- Ensure proper testing in your environment before proceeding
"""
            
            return {"result": comprehensive_response.strip()}
        
    except Exception as e:
        print(f"Error in query processing: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/options/databases")
async def get_db_options_endpoint():
    """Get available database options for dropdown from analyze_compatibility.py."""
    try:
        return {"databases": get_db_options()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/options/operating-systems")
async def get_osi_options_endpoint():
    try:
        return {"operating_systems": analyzer.get_osi_options()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/options/web-servers")
async def get_web_server_options_endpoint():
    """Get available web server options for dropdown from actual data."""
    try:
        import pandas as pd
        import os
        import json
        
        # Try to extract web server options from the data files
        web_servers = []
        
        # First, try to get from compatibility analysis JSON
        analysis_file = 'data/processed/compatibility_analysis.json'
        if os.path.exists(analysis_file):
            try:
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Extract web server models from the analysis data
                for server in analysis_data.get('servers', []):
                    if 'web_server' in server:
                        web_servers.append(server['web_server'])
            except Exception as e:
                print(f"Error reading compatibility analysis: {e}")
        
        # If no web servers found, try CSV files
        if not web_servers:
            data_files = [
                'data/processed/Webserver_OS_Mapping.csv',
                'data/raw/WebServer.csv'
            ]
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Look for web server related columns
                        web_server_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['webserver', 'web_server', 'server'])]
                        
                        for col in web_server_columns:
                            unique_values = df[col].dropna().unique()
                            web_servers.extend([str(val) for val in unique_values if val and str(val).strip()])
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        # Remove duplicates and sort
        web_servers = sorted(list(set(web_servers)))
        
        # Fallback to hardcoded options if no data found
        if not web_servers:
            web_servers = [
                'Apache HTTP Server 2.4', 'Nginx 1.20', 'Nginx 1.22', 'Microsoft IIS 10',
                'Tomcat 9', 'Tomcat 10', 'Node.js 16', 'Node.js 18', 'Node.js 20'
            ]
        
        return {"web_servers": web_servers}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/user/config")
async def save_user_config(config_data: dict):
    """Save user configuration to database."""
    try:
        user_id = config_data.get('user_id', 'default_user')
        # Validate required fields
        required_fields = ['operatingSystem', 'database', 'webServers', 'useInChat']
        for field in required_fields:
            if field not in config_data:
                return JSONResponse(
                    status_code=400, 
                    content={"error": f"Missing required field: {field}"}
                )
        # Save to database
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        # Convert config to JSON string
        config_json = json.dumps(config_data)
        # Insert or update user config
        cursor.execute('''
            INSERT OR REPLACE INTO user_configs (user_id, config_data, updated_at)
            VALUES (?, ?, ?)
        ''', (user_id, config_json, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        print(f"User configuration saved to database for user: {user_id}")
        return {"message": "Configuration saved successfully"}
    except Exception as e:
        print(f"Error saving configuration to database: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/user/config")
async def get_user_config(user_id: str = Query('default_user')):
    """Get user configuration from database."""
    try:
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        # Get user config from database
        cursor.execute('''
            SELECT config_data FROM user_configs 
            WHERE user_id = ?
        ''', (user_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            # Parse the JSON config data
            config_data = json.loads(result[0])
            return config_data
        else:
            # Return default config if no saved config found
            return {
                "operatingSystem": "",
                "database": "",
                "webServers": [],
                "useInChat": {
                    "os": True,
                    "database": True,
                    "webServers": True
                }
            }
    except Exception as e:
        print(f"Error retrieving configuration from database: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/user/chat")
async def save_user_chat(chat_data: dict):
    """Save a chat message for a user and session."""
    try:
        user_id = chat_data.get('user_id', 'default_user')
        session_id = chat_data.get('session_id')
        message_id = chat_data.get('message_id')
        message_data = chat_data.get('message_data')
        if not (user_id and session_id and message_id and message_data):
            return JSONResponse(status_code=400, content={"error": "Missing required chat fields."})
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_chats (user_id, session_id, message_id, message_data, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, session_id, message_id, json.dumps(message_data), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {"message": "Chat message saved successfully"}
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/user/chat")
async def get_user_chat(user_id: str = Query('default_user')):
    """Get all chat sessions/messages for a user."""
    try:
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, message_id, message_data, created_at FROM user_chats
            WHERE user_id = ?
            ORDER BY session_id, created_at
        ''', (user_id,))
        rows = cursor.fetchall()
        conn.close()
        # Group messages by session_id
        sessions = {}
        for row in rows:
            session_id, message_id, message_data, created_at = row
            msg = json.loads(message_data)
            msg['message_id'] = message_id
            msg['created_at'] = created_at
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(msg)
        return {"sessions": sessions}
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/analytics/recent")
async def analytics_recent():
    try:
        return {"recent_feedback": get_recent_feedback()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True) 