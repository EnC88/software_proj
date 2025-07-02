import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.vector_store import VectorStore
from src.evaluation.feedback_system import FeedbackLogger
from src.data_processing.analyze_compatibility import CompatibilityAnalyzer, get_database_options, get_osi_options
import uuid
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Simple compatibility result classes
@dataclass
class CompatibilityResult:
    """Result of compatibility analysis."""
    is_compatible: bool
    confidence: float
    affected_servers: List[Dict[str, Any]]
    conflicts: List[str]
    recommendations: List[str]
    warnings: List[str]
    alternative_versions: List[str]

@dataclass
class ChangeRequest:
    """Represents a user's software change request."""
    software_name: str
    version: Optional[str] = None
    action: str = "upgrade"
    environment: Optional[str] = None
    target_servers: Optional[List[str]] = None
    raw_text: str = ""

# For RAG/LLM compatibility logic
vector_store = VectorStore()
# For data analysis (keeping for potential future use)
analyzer = CompatibilityAnalyzer()

feedback_logger = FeedbackLogger()

# Simple query interface using VectorStore
class SimpleQueryInterface:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Simple query using VectorStore."""
        try:
            results = self.vector_store.query(query_text, top_k=top_k)
            return results
        except Exception as e:
            return {"error": str(e), "query": query_text, "results": [], "total_results": 0}
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results for display."""
        if "error" in results:
            return f"<div class='results-container'><div class='section-empty'>❌ Error: {results['error']}</div></div>"
        
        if not results.get('results'):
            return "<div class='results-container'><div class='section-empty'>No results found. Try a different query.</div></div>"
        
        output = ["<div class='results-container'>"]
        output.append(f"<h3>🔍 Search Results for: <em>'{results['query']}'</em></h3>")
        output.append(f"<p><strong>Found {results['total_results']} relevant documents:</strong></p>")
        
        for i, result in enumerate(results['results'], 1):
            content = result.get('content', 'No content')
            metadata = result.get('metadata', {})
            
            # Format the content nicely
            content_lines = content.split('\n')
            formatted_content = '<br>'.join(content_lines)
            
            # Get metadata info
            doc_type = metadata.get('type', 'unknown')
            timestamp = metadata.get('timestamp', 'Unknown')
            
            output.append(f"""
            <div style='border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin: 1rem 0; background: #f8fafc;'>
                <h4 style='margin: 0 0 0.5rem 0; color: #1e293b;'>Result {i}</h4>
                <div style='font-size: 0.9em; color: #64748b; margin-bottom: 0.5rem;'>
                    Type: {doc_type} | Timestamp: {timestamp}
                </div>
                <div style='background: white; padding: 1rem; border-radius: 4px; border-left: 4px solid #3b82f6;'>
                    {formatted_content}
                </div>
            </div>
            """)
        
        output.append("</div>")
        return ''.join(output)

# Initialize query interface
query_interface = SimpleQueryInterface(vector_store)

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
    
    df['query_length'] = df['query'].str.len()
    avg_length = df['query_length'].mean()
    score_counts = df['feedback_score'].value_counts().sort_index()
    
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
    
    recent_df = df.sort_values('timestamp', ascending=False).head(10)
    
    table_html = "<table style='width: 100%; border-collapse: collapse; margin: 1rem 0;'>"
    table_html += "<thead>...</thead><tbody>" # Simplified for brevity
    
    for _, row in recent_df.iterrows():
        score_icon = "✅" if row['feedback_score'] == 1 else "❌" if row['feedback_score'] == 0 else "⏳"
        query_preview = row['query'][:50] + "..." if len(row['query']) > 50 else row['query']
        date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        table_html += f"<tr><td>{date_str}</td><td>{query_preview}</td><td>{score_icon}</td><td>{row['user_os']}</td></tr>"
    
    table_html += "</tbody></table>"
    return f"<div style='margin: 1rem 0;'><h3>Recent Feedback</h3>{table_html}</div>"

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
    if not items: return ""
    output = []
    for item in items:
        if highlight:
            item = re.sub(r"([A-Z][A-Z0-9 \-]+) ([0-9]+\.[0-9.]+): ([0-9,]+) server\(s\) across ([A-Z0-9, \-]+)",
                          r"<b>\1 \2</b>: <span style='color:#0ea5e9;font-weight:bold;'>\3 servers</span> <span style='color:#64748b;'>across \4</span>", item)
        output.append(f"<li>{icon} {item}</li>" if icon else f"<li>{item}</li>")
    return f"<ul class='section-list'>{''.join(output)}</ul>"

def build_interface():
    with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {background: #f8fafc;}
    .main-title {font-size: 2.5rem; font-weight: bold; color: #1e293b; margin-bottom: 0.2em;}
    .subtitle {font-size: 1.2rem; color: #64748b; margin-bottom: 1.5em;}
    .results-container {background: #fff; border-radius: 12px; padding: 2em 2.5em; box-shadow: 0 2px 8px #e0e7ef; margin-top: 1.5em;}
    .section-header {font-size: 1.2em; font-weight: 600; margin-top: 1.2em; margin-bottom: 0.5em; display: flex; align-items: center; gap: 0.5em;}
    .section-list {margin: 0 0 0 1.2em; padding: 0;}
    .section-empty {color: #a3a3a3; font-style: italic; margin-left: 1.2em;}
    .os-widget {padding: 1.5em; background-color: #f1f5f9; border-radius: 12px; text-align: center;}
    .os-question {font-size: 1.1em; color: #334155; margin-bottom: 1em;}
    .gr-accordion {margin-bottom: 1em;}
    """) as demo:
        user_os = gr.State("")
        detected_os = gr.State("")
        session_id = gr.State("")
        last_query = gr.State("")
        last_results = gr.State("")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""
                <div class='main-title'>Infrastructure Search Assistant</div>
                <div class='subtitle'>Ask questions about your servers, software, and infrastructure</div>
                """)
                
                # Database dropdown
                db_options = get_database_options()
                
                with gr.Row():
                    db_dropdown = gr.Dropdown(
                        choices=db_options,
                        label="Select Database Version (Optional)",
                        value=None,
                        allow_custom_value=True,
                        info="Choose a specific database version to focus your search",
                        scale=2
                    )
                    refresh_db_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                
                # OSI dropdown
                osi_options = get_osi_options()
                
                with gr.Row(visible=True) as osi_row:
                    osi_dropdown = gr.Dropdown(
                        choices=osi_options,
                        label="Select OSI Version",
                        value=None,
                        allow_custom_value=True,
                        info="Choose your specific OSI version",
                        scale=2
                    )
                    refresh_osi_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                
                request_box = gr.Textbox(
                    label="Your Question",
                    placeholder="E.g. 'What servers are running Apache HTTPD?' or 'What OS does Apache run best on?'",
                    lines=4,
                    elem_id="request-box"
                )
                analyze_btn = gr.Button("Search Infrastructure", elem_id="analyze-btn", variant="primary")
                
                # --- Results Section ---
                results_md = gr.Markdown(visible=False)

                # --- Feedback Section ---
                with gr.Column(visible=False) as feedback_container:
                    gr.Markdown("**Rate these search results**", elem_id="feedback-header")
                    with gr.Row():
                        feedback_good_btn = gr.Button("👍 Helpful Results")
                        feedback_bad_btn = gr.Button("👎 Not Helpful")
                    feedback_thanks_md = gr.Markdown(visible=False)

                # --- Analytics Section ---
                with gr.Accordion("📊 Analytics & Insights", open=False) as analytics_container:
                    with gr.Tabs():
                        with gr.TabItem("📈 Overall Stats"):
                            stats_md = gr.Markdown()
                        with gr.TabItem("🔍 Query Analysis"):
                            query_analysis_md = gr.Markdown()
                        with gr.TabItem("💻 OS Distribution"):
                            os_analysis_md = gr.Markdown()
                        with gr.TabItem("📋 Recent Feedback"):
                            recent_feedback_md = gr.Markdown()

            with gr.Column(scale=1):
                # --- OS Detection Widget ---
                with gr.Group(elem_classes="os-widget"):
                    os_question_md = gr.Markdown(visible=False)
                    with gr.Row(visible=False) as os_confirm_buttons:
                        os_yes_btn = gr.Button("Yes", variant="primary")
                        os_no_btn = gr.Button("No")
                    os_select_dd = gr.Dropdown(
                        ["Windows", "macOS", "Linux", "Other"],
                        label="Select Your OS",
                        visible=False
                    )
                    os_confirmed_md = gr.Markdown(visible=False)

        # --- Functions ---
        def get_os(user_agent: gr.Request):
            """Detects OS from user agent and updates the UI."""
            ua = user_agent.headers.get("user-agent", "").lower()
            os_map = {"windows": "Windows", "mac": "macOS", "linux": "Linux"}
            for key, val in os_map.items():
                if key in ua:
                    return {
                        detected_os: val,
                        os_question_md: gr.update(value=f"<div class='os-question'>Are you using <b>{val}</b>?</div>", visible=True),
                        os_confirm_buttons: gr.update(visible=True)
                    }
            return {
                detected_os: "Other",
                os_question_md: gr.update(visible=False),
                os_confirm_buttons: gr.update(visible=False),
                os_select_dd: gr.update(visible=True)
            }

        def confirm_os(os_name):
            """Confirms the OS/OSI and updates the UI."""
            return {
                user_os: os_name,
                os_question_md: gr.update(visible=False),
                os_confirm_buttons: gr.update(visible=False),
                os_select_dd: gr.update(visible=False),
                os_confirmed_md: gr.update(value=f"✅ OS set to <b>{os_name}</b>", visible=True)
            }

        def confirm_osi(osi_name):
            """Confirms the OSI selection and updates the UI."""
            return {
                os_confirmed_md: gr.update(value=f"✅ OSI version set to <b>{osi_name}</b>", visible=True)
            }

        def show_os_select():
            """Shows the OS selection dropdown."""
            return {
                os_question_md: gr.update(visible=False),
                os_confirm_buttons: gr.update(visible=False),
                os_select_dd: gr.update(visible=True)
            }

        def refresh_db_dropdown():
            """Refresh the database dropdown with current data."""
            db_options = get_database_options()
            return gr.update(choices=db_options, value=None)

        def refresh_osi_dropdown():
            """Refresh the OSI dropdown with current data."""
            osi_options = get_osi_options()
            return gr.update(choices=osi_options, value=None)

        def on_analyze(request_text, current_os, selected_db, selected_osi):
            """Simple search function using VectorStore."""
            if not request_text.strip():
                return {
                    "results_md": gr.update(value="<div class='results-container section-empty'>⚠️ Please enter a question about your infrastructure.</div>", visible=True),
                    "feedback_container": gr.update(visible=False),
                    "feedback_thanks_md": gr.update(visible=False),
                    "last_query": request_text,
                    "last_results": ""
                }
            
            try:
                # Enhance query with database and OSI selection if provided
                enhanced_query = request_text
                if selected_db and selected_db != "No database versions found":
                    enhanced_query = f"{enhanced_query} [Database: {selected_db}]"
                if selected_osi and selected_osi != "No OSI versions found":
                    enhanced_query = f"{enhanced_query} [OSI: {selected_osi}]"
                
                # Use the query interface
                results = query_interface.query(enhanced_query, top_k=5)
                formatted_results = query_interface.format_results(results)
                
                return {
                    "results_md": gr.update(value=formatted_results, visible=True),
                    "feedback_container": gr.update(visible=True),
                    "feedback_thanks_md": gr.update(visible=False),
                    "last_query": enhanced_query,
                    "last_results": str(results)
                }
            except Exception as e:
                error_msg = f"<div class='results-container'><div class='section-empty'>❌ Error: {str(e)}</div></div>"
                return {
                    "results_md": gr.update(value=error_msg, visible=True),
                    "feedback_container": gr.update(visible=False),
                    "feedback_thanks_md": gr.update(visible=False),
                    "last_query": request_text,
                    "last_results": ""
                }

        def log_feedback(score, query, results, os, sid):
            """Logs feedback and shows a thank you message."""
            feedback_logger.log(query=query, generated_output=results, feedback_score=score, user_os=os, session_id=sid)
            # Hide feedback_container, show thank you message
            return gr.update(visible=False), gr.update(value="🙏 **Thank you for your feedback!**", visible=True)

        def update_analytics():
            """Update all analytics sections."""
            return {
                stats_md: create_overall_stats(),
                query_analysis_md: create_query_analysis(),
                os_analysis_md: create_os_analysis(),
                recent_feedback_md: create_recent_feedback_table()
            }

        # --- Event Listeners ---
        demo.load(
            lambda: {session_id: str(uuid.uuid4())}, 
            outputs=[session_id]
        ).then(
            get_os, 
            inputs=None, 
            outputs=[detected_os, os_question_md, os_confirm_buttons, os_select_dd]
        )
        os_yes_btn.click(confirm_os, inputs=detected_os, outputs=[user_os, os_question_md, os_confirm_buttons, os_select_dd, os_confirmed_md])
        os_no_btn.click(show_os_select, outputs=[os_question_md, os_confirm_buttons, osi_row])
        os_select_dd.change(confirm_os, inputs=os_select_dd, outputs=[user_os, os_question_md, os_confirm_buttons, os_select_dd, os_confirmed_md])
        osi_dropdown.change(confirm_osi, inputs=osi_dropdown, outputs=[os_confirmed_md])

        analyze_btn.click(
            on_analyze, 
            inputs=[request_box, user_os, db_dropdown, osi_dropdown], 
            outputs=[results_md, feedback_container, feedback_thanks_md, last_query, last_results]
        )

        # Refresh database dropdown
        refresh_db_btn.click(
            refresh_db_dropdown,
            outputs=[db_dropdown]
        )

        # Refresh OSI dropdown
        refresh_osi_btn.click(
            refresh_osi_dropdown,
            outputs=[osi_dropdown]
        )

        # Update analytics when page loads and after feedback
        demo.load(update_analytics, outputs=[stats_md, query_analysis_md, os_analysis_md, recent_feedback_md])
        
        # Update analytics after feedback is submitted
        feedback_good_btn.click(
            lambda query, results, os, sid: log_feedback(1, query, results, os, sid),
            inputs=[last_query, last_results, user_os, session_id],
            outputs=[feedback_container, feedback_thanks_md]
        ).then(
            update_analytics,
            outputs=[stats_md, query_analysis_md, os_analysis_md, recent_feedback_md]
        )
        
        feedback_bad_btn.click(
            lambda query, results, os, sid: log_feedback(0, query, results, os, sid),
            inputs=[last_query, last_results, user_os, session_id],
            outputs=[feedback_container, feedback_thanks_md]
        ).then(
            update_analytics,
            outputs=[stats_md, query_analysis_md, os_analysis_md, recent_feedback_md]
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )