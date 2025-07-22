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
from pydantic import BaseModel

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.CompatibilityLLMAgent import agent  # Import the existing agent instance
from src.Agents.ContextQueryAgent import agent as context_query_agent
from src.Agents.ChatTitleAgent import ChatTitleAgent
from src.Agents.SimpleRouter import route_query
from src.rag.determine_recs import get_co_upgrades
from src.rag.query_engine import QueryEngine
from src.evaluation.feedback_system import FeedbackLogger, run_feedback_loop
from src.data_processing.analyze_compatibility import CompatibilityAnalyzer, get_db_options, get_osi_options
from src.rag.RecommendationValidationAgent import RecommendationValidationAgent  # Import the validation agent

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
        # Create chat_session_titles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_session_titles (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                title TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id)
            )
        ''')
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")

# Initialize database on startup
init_db()

# Create global instances to avoid reloading on every request
print("Initializing heavy components...")
check_compat = CheckCompatibility()
query_engine = QueryEngine()
print("Heavy components initialized successfully!")

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
    
    change_requests = await check_compat.parse_multiple_change_requests(request_text)
    print("DEBUG: change_requests passed to analyze_multiple_compatibility:", change_requests)
    results = await check_compat.analyze_multiple_compatibility(
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
    """Handle only technical/compatibility queries: analyze, recommend, label, validate."""
    try:
        data = await request.json()
        user_query = data.get('query')
        user_os = data.get('os')
        user_database = data.get('database')
        user_web_servers = data.get('webServers', [])

        if not user_query:
            return JSONResponse(status_code=400, content={"error": "Query required"})

        # Only handle technical queries for now
        change_requests = await check_compat.parse_multiple_change_requests(user_query)
        if change_requests:
            results = await check_compat.analyze_multiple_compatibility(
                change_requests,
                target_os=user_os,
                user_database=user_database,
                user_web_servers=user_web_servers
            )
            compatibility_analysis = check_compat.format_multiple_results(results)
            # Gather all recommendations from all results
            all_recommendations = []
            all_filtering_steps = []
            for cr, res in results:
                if hasattr(res, 'recommendations') and res.recommendations:
                    all_recommendations.extend(res.recommendations)
                if hasattr(res, 'filtering_steps') and res.filtering_steps:
                    all_filtering_steps.extend(res.filtering_steps)
            # Generate labels for filtering steps using ChatTitleAgent
            labeled_filtering_steps = []
            if all_filtering_steps:
                title_agent = ChatTitleAgent()
                for step in all_filtering_steps:
                    try:
                        label_prompt = f"Generate a 3-5 word label for this filtering step: {step['description']}"
                        label_result = await title_agent.generate_title(label_prompt)
                        label = None
                        if hasattr(label_result, 'messages') and label_result.messages:
                            for message in label_result.messages:
                                if hasattr(message, 'type') and message.type == 'TextMessage' and hasattr(message, 'content'):
                                    label = message.content
                                    break
                        elif hasattr(label_result, 'content'):
                            label = label_result.content
                        elif isinstance(label_result, str):
                            label = label_result
                        if label and isinstance(label, str):
                            if 'TaskResult(' in label or 'TextMessage(' in label:
                                import re
                                content_match = re.search(r"content='([^']*)'", label)
                                if content_match:
                                    label = content_match.group(1)
                                else:
                                    label = step['stage'].replace('_', ' ').title()
                            label = label.replace('source= TitleAgent', '').replace('3-5 words:', '').strip()
                        if not label or len(label) > 50:
                            label = step['stage'].replace('_', ' ').title()
                        labeled_filtering_steps.append({
                            "stage": step['stage'],
                            "count": step['count'],
                            "description": step['description'],
                            "label": label
                        })
                    except Exception as e:
                        labeled_filtering_steps.append({
                            "stage": step['stage'],
                            "count": step['count'],
                            "description": step['description'],
                            "label": step['stage'].replace('_', ' ').title()
                        })
            # Only validate if there are recommendations
            validated_recommendations = None
            if all_recommendations:
                validator = RecommendationValidationAgent()
                main_cr = change_requests[0]
                cr_text = main_cr.raw_text if hasattr(main_cr, 'raw_text') else str(main_cr)
                context_str = f"OS: {user_os}, DB: {user_database}, WebServers: {user_web_servers}"
                validated_recommendations = await validator.validate_recommendations(
                    change_request=cr_text,
                    recommendations=all_recommendations,
                    context=context_str
                )
            else:
                validated_recommendations = "No recommendations to validate."
            return {
                "result": compatibility_analysis.strip(),
                "validated_recommendations": validated_recommendations,
                "filtering_steps": labeled_filtering_steps
            }
        else:
            return {"result": "No specific software change requests detected in your query.",
                    "validated_recommendations": None,
                    "filtering_steps": []}
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
        from src.data_processing.analyze_compatibility import get_web_server_options
        return {"web_servers": get_web_server_options()}
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

@app.post("/api/user/chat_title")
async def save_chat_title(user_id: str, session_id: str, title: str):
    """Save or update a chat session title."""
    try:
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO chat_session_titles (user_id, session_id, title)
            VALUES (?, ?, ?)
        ''', (user_id, session_id, title))
        conn.commit()
        conn.close()
        return {"message": "Title saved"}
    except Exception as e:
        print(f"Error saving chat title: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/user/chat_titles")
async def get_chat_titles(user_id: str):
    """Get all chat session titles for a user."""
    try:
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, title FROM chat_session_titles WHERE user_id = ?
        ''', (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return {"titles": {row[0]: row[1] for row in rows}}
    except Exception as e:
        print(f"Error fetching chat titles: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/user/chat")
async def get_user_chat(user_id: str = Query('default_user')):
    """Get all chat sessions/messages for a user, including session titles if available."""
    try:
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, message_id, message_data, created_at FROM user_chats
            WHERE user_id = ?
            ORDER BY session_id, created_at
        ''', (user_id,))
        rows = cursor.fetchall()
        # Fetch titles
        cursor.execute('''
            SELECT session_id, title FROM chat_session_titles WHERE user_id = ?
        ''', (user_id,))
        title_rows = cursor.fetchall()
        session_titles = {row[0]: row[1] for row in title_rows}
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
        return {"sessions": sessions, "session_titles": session_titles}
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/api/user/chat_session")
async def delete_user_chat_session(user_id: str, session_id: str):
    """Delete all chat messages for a user and session."""
    try:
        conn = sqlite3.connect('user_configs.db')
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM user_chats WHERE user_id = ? AND session_id = ?
        ''', (user_id, session_id))
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        return {"message": f"Deleted {deleted} messages for session {session_id}"}
    except Exception as e:
        print(f"Error deleting chat session: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/analytics/recent")
async def analytics_recent():
    try:
        return {"recent_feedback": get_recent_feedback()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

class CoUpgradeRequest(BaseModel):
    catalogid: str

@app.post('/api/test_co_upgrade')
async def test_co_upgrade(body: CoUpgradeRequest):
    catalogid = body.catalogid
    co_upgrades = get_co_upgrades(catalogid)
    return {'co_upgrades': co_upgrades}

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True) 