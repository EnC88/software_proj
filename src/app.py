from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import uvicorn
from typing import Dict, Any
import pandas as pd
from datetime import datetime, timedelta

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.determine_recs import CheckCompatibility
from src.evaluation.feedback_system import FeedbackLogger, run_feedback_loop
from src.data_processing.analyze_compatibility import CompatibilityAnalyzer

check_compat = CheckCompatibility()
feedback_logger = FeedbackLogger()
analyzer = CompatibilityAnalyzer()

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (React build)
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates', 'frontend', 'dist'))
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

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

# --- API Endpoints ---

@app.post("/api/analyze")
async def analyze(request: Request):
    data = await request.json()
    request_text = data.get("request")
    user_os = data.get("os")
    if not request_text:
        raise HTTPException(status_code=400, detail="Missing 'request' field.")
    change_requests = check_compat.parse_multiple_change_requests(request_text)
    results = check_compat.analyze_multiple_compatibility(change_requests, target_os=user_os)
    # Serialize results
    output = []
    for cr, result in results:
        output.append({
            "request": f"{cr.action.title()} {cr.software_name} {cr.version or ''}",
            "is_compatible": result.is_compatible,
            "confidence": result.confidence,
            "affected_servers": result.affected_servers,
            "conflicts": result.conflicts,
            "warnings": result.warnings,
            "recommendations": result.recommendations,
            "alternative_versions": result.alternative_versions,
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
    session_id = data.get("session_id")
    metadata = {"tags": tags}
    if correction and correction.strip():
        metadata["correction"] = correction.strip()
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

@app.get("/api/analytics/recent")
async def analytics_recent():
    try:
        return {"recent_feedback": get_recent_feedback()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Fallback: serve index.html for any other route (for React Router)
if os.path.exists(frontend_dist):
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        index_path = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse(status_code=404, content={"error": "Not found"})

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True) 