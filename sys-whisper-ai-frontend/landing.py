import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.determine_recs import CheckCompatibility, CompatibilityResult, ChangeRequest
from src.data_processing.analyze_compatibility import CompatibilityAnalyzer
from src.evaluation.feedback_system import FeedbackLogger, run_feedback_loop
import uuid
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import json

# Instantiate the correct classes
check_compat = CheckCompatibility()
analyzer = CompatibilityAnalyzer()
feedback_logger = FeedbackLogger()

# --- Formatting Functions ---
def format_status(result: CompatibilityResult) -> str:
    status = "COMPATIBLE" if result.is_compatible else "INCOMPATIBLE"
    color = "#22c55e" if result.is_compatible else "#ef4444"
    return f"<span style='display:inline-block;padding:0.4em 1em;border-radius:1em;background:{color};color:white;font-weight:bold;font-size:1.1em;'>{status}</span> <span style='color:#64748b;font-size:1em;'>(Confidence: {result.confidence:.1%})</span>"

def format_affected_models(result: CompatibilityResult) -> str:
    if not result.affected_servers:
        return "<div class='section-empty'>No specific models identified</div>"
    model_env_map = {}
    for server in result.affected_servers:
        model = server.get('server_info', {}).get('model', 'Unknown')
        product_type = server.get('server_info', {}).get('product_type', 'Unknown')
        env = server.get('environment', 'Unknown')
        if str(model) in ['Unknown', 'Closed'] or str(product_type) in ['Unknown', 'Closed'] or str(env) in ['Unknown', 'Closed', 'nan']:
            continue
        key = f"{model} ({product_type})"
        if key not in model_env_map:
            model_env_map[key] = set()
        model_env_map[key].add(env)
    if not model_env_map:
        return "<div class='section-empty'>No specific models identified</div>"
    output = []
    for model, envs in list(model_env_map.items())[:5]:
        envs_str = ', '.join(sorted([str(e) for e in envs]))
        output.append(f"<li><b>{model}</b> <span style='color:#64748b;'>[{envs_str}]</span></li>")
    if len(model_env_map) > 5:
        output.append(f"<li>... and {len(model_env_map) - 5} more</li>")
    return f"<ul class='section-list'>{''.join(output)}</ul>"

def format_list_section(items: list, icon: str = "", highlight: bool = False) -> str:
    if not items:
        return ""
    output = []
    for item in items:
        if highlight:
            # Try to bold product names and server counts
            import re
            item = re.sub(r"([A-Z][A-Z0-9 \-]+) ([0-9]+\.[0-9.]+): ([0-9,]+) server\(s\) across ([A-Z0-9, \-]+)",
                          r"<b>\1 \2</b>: <span style='color:#0ea5e9;font-weight:bold;'>\3 servers</span> <span style='color:#64748b;'>across \4</span>",
                          item)
        output.append(f"<li>{icon} {item}</li>" if icon else f"<li>{item}</li>")
    return f"<ul class='section-list'>{''.join(output)}</ul>"

# --- Analytics Functions ---
def get_analytics_data():
    """Get analytics data from feedback database."""
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

def create_overall_stats():
    """Create overall statistics."""
    df = get_analytics_data()
    if df is None or df.empty:
        return "No feedback data available yet."
    
    total_queries = len(df)
    positive_feedback = len(df[df['feedback_score'] == 1])
    negative_feedback = len(df[df['feedback_score'] == 0])
    positive_rate = (positive_feedback / total_queries * 100) if total_queries > 0 else 0
    recent_queries = len(df[df['timestamp'] >= datetime.now() - timedelta(days=7)])
    
    stats_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
        <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #0ea5e9;">{total_queries}</div>
            <div style="color: #64748b;">Total Queries</div>
        </div>
        <div style="background: #f0fdf4; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #22c55e;">{positive_rate:.1f}%</div>
            <div style="color: #64748b;">Positive Rate</div>
        </div>
        <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #f59e0b;">{recent_queries}</div>
            <div style="color: #64748b;">Last 7 Days</div>
        </div>
        <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #ef4444;">{negative_feedback}</div>
            <div style="color: #64748b;">Needs Improvement</div>
        </div>
    </div>
    """
    return stats_html

def create_query_analysis():
    """Create query analysis charts."""
    df = get_analytics_data()
    if df is None or df.empty:
        return "No feedback data available yet."
    
    # Query length analysis
    df['query_length'] = df['query'].str.len()
    avg_length = df['query_length'].mean()
    
    # Feedback score distribution
    score_counts = df['feedback_score'].value_counts().sort_index()
    
    # Create charts
    fig1 = go.Figure(data=[
        go.Bar(x=['Short (<50)', 'Medium (50-100)', 'Long (>100)'], 
               y=[len(df[df['query_length'] < 50]), 
                  len(df[(df['query_length'] >= 50) & (df['query_length'] <= 100)]),
                  len(df[df['query_length'] > 100])],
               marker_color=['#3b82f6', '#10b981', '#f59e0b'])
    ])
    fig1.update_layout(title="Query Length Distribution", height=300)
    
    fig2 = go.Figure(data=[
        go.Pie(labels=['Positive', 'Negative', 'Unrated'], 
               values=[score_counts.get(1, 0), score_counts.get(0, 0), score_counts.get(-1, 0)],
               marker_colors=['#22c55e', '#ef4444', '#64748b'])
    ])
    fig2.update_layout(title="Feedback Score Distribution", height=300)
    
    return f"""
    <div style="margin: 1rem 0;">
        <h3>Query Analysis</h3>
        <p><strong>Average query length:</strong> {avg_length:.1f} characters</p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>
            <div>{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>
        </div>
    </div>
    """

def create_os_analysis():
    """Create OS distribution analysis."""
    df = get_analytics_data()
    if df is None or df.empty:
        return "No feedback data available yet."
    
    os_counts = df['user_os'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(x=os_counts.index, y=os_counts.values, marker_color='#8b5cf6')
    ])
    fig.update_layout(title="User OS Distribution", height=300)
    
    return f"""
    <div style="margin: 1rem 0;">
        <h3>Operating System Analysis</h3>
        <div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    """

def create_recent_feedback_table():
    """Create recent feedback table."""
    df = get_analytics_data()
    if df is None or df.empty:
        return "No feedback data available yet."
    
    # Get recent feedback (last 10)
    recent_df = df.sort_values('timestamp', ascending=False).head(10)
    
    table_html = "<table style='width: 100%; border-collapse: collapse; margin: 1rem 0;'>"
    table_html += """
    <thead>
        <tr style='background: #f8fafc;'>
            <th style='padding: 0.5rem; text-align: left; border-bottom: 1px solid #e2e8f0;'>Date</th>
            <th style='padding: 0.5rem; text-align: left; border-bottom: 1px solid #e2e8f0;'>Query</th>
            <th style='padding: 0.5rem; text-align: center; border-bottom: 1px solid #e2e8f0;'>Score</th>
            <th style='padding: 0.5rem; text-align: left; border-bottom: 1px solid #e2e8f0;'>OS</th>
        </tr>
    </thead>
    <tbody>
    """
    
    for _, row in recent_df.iterrows():
        score_icon = "✅" if row['feedback_score'] == 1 else "❌" if row['feedback_score'] == 0 else "⏳"
        query_preview = row['query'][:50] + "..." if len(row['query']) > 50 else row['query']
        date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        
        table_html += f"""
        <tr>
            <td style='padding: 0.5rem; border-bottom: 1px solid #f1f5f9;'>{date_str}</td>
            <td style='padding: 0.5rem; border-bottom: 1px solid #f1f5f9;'>{query_preview}</td>
            <td style='padding: 0.5rem; text-align: center; border-bottom: 1px solid #f1f5f9;'>{score_icon}</td>
            <td style='padding: 0.5rem; border-bottom: 1px solid #f1f5f9;'>{row['user_os']}</td>
        </tr>
        """
    
    table_html += "</tbody></table>"
    
    return f"""
    <div style="margin: 1rem 0;">
        <h3>Recent Feedback</h3>
        {table_html}
    </div>
    """

def dummy_chatbot(message, history):
    # Replace with real backend logic if available
    return history + [[message, "This is a dummy response."]]

def get_stats():
    # Replace with real analytics if available
    return 12847, 1247, "94.2%", "1.2s"

def plot_query_trends():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=["Jan", "Feb", "Mar", "Apr", "May", "Jun"], y=[1200, 1400, 1600, 1800, 2000, 2200], mode='lines+markers'))
    fig.add_trace(go.Scatter(x=["Jan", "Feb", "Mar", "Apr", "May", "Jun"], y=[1100, 1350, 1550, 1750, 1950, 2150], mode='lines+markers'))
    fig.update_layout(title="Query Volume Trends")
    return fig

def plot_os_compat():
    fig = go.Figure([go.Bar(x=["Windows", "Linux", "macOS"], y=[85, 92, 78])])
    fig.update_layout(title="OS Compatibility Rates")
    return fig

with gr.Blocks(theme=gr.themes.Soft(), css="""
.gradio-container {background: #f8fafc;}
.stat-card {background: #fff; border-radius: 10px; padding: 1.5em; text-align: center; box-shadow: 0 2px 8px #e0e7ef; font-size: 1.3em;}
.stat-label {color: #64748b; font-size: 1em;}
.stat-value {font-size: 2.2em; font-weight: bold; margin-bottom: 0.2em;}
.quick-btn {width: 100%; margin-bottom: 0.5em; font-size: 1.1em;}
""") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            <div style='font-size:1.3em;font-weight:bold;margin-bottom:0.5em;'>System Configuration</div>
            <div style='color:#64748b;margin-bottom:1em;'>Optional - helps provide targeted recommendations</div>
            """)
            # Place your OS, DB, and Web Server dropdowns here, using your real logic/state
            os_dd = gr.Dropdown(["Windows", "Linux", "macOS"], label="Operating System")
            db_dd = gr.Dropdown(["PostgreSQL", "MySQL", "MongoDB"], label="Database")
            ws_dd = gr.Dropdown(["Apache", "Nginx", "IIS"], label="Web Servers")
        with gr.Column(scale=2):
            gr.Markdown("""
            <div style='font-size:1.3em;font-weight:bold;margin-bottom:0.5em;'>System Compatibility Assistant</div>
            <div style='color:#64748b;margin-bottom:1em;'>Ask questions about OS, databases, and web servers</div>
            """)
            with gr.Row():
                gr.Button("Check OS compatibility", elem_classes="quick-btn")
                gr.Button("Database requirements", elem_classes="quick-btn")
                gr.Button("Web server setup", elem_classes="quick-btn")
            # Place your chat interface and input here, using your real chat logic
            results_md = gr.Markdown(visible=False)
            feedback_container = gr.Column(visible=False)
            feedback_thanks_md = gr.Markdown(visible=False)
            last_query = gr.State("")
            last_results = gr.State("")
            # ... (rest of your chat/feedback UI as before) ...

    gr.Markdown("---")
    gr.Markdown("""
    <div style='font-size:2em;font-weight:bold;text-align:center;margin-bottom:0.2em;'>System Compatibility Assistant</div>
    <div style='text-align:center;color:#64748b;margin-bottom:2em;'>Enterprise-grade system compatibility analysis and recommendations</div>
    """)
    with gr.Row():
        # Use your real stats/analytics functions here
        stats_md = gr.Markdown()
        active_systems = gr.Markdown("""<div class='stat-card'><div class='stat-value'>1,247</div><div class='stat-label'>Active Systems</div></div>""")
        comp_score = gr.Markdown("""<div class='stat-card'><div class='stat-value'>94.2%</div><div class='stat-label'>Compatibility Score</div></div>""")
        resp_time = gr.Markdown("""<div class='stat-card'><div class='stat-value'>1.2s</div><div class='stat-label'>Response Time</div></div>""")
    with gr.Row():
        # Use your real chart/plot functions here
        query_analysis_md = gr.Markdown()
        os_analysis_md = gr.Markdown()

    # ... rest of your event listeners and logic remain unchanged ...

demo.launch()