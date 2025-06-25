import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.determine_recs import CompatibilityAnalyzer, CompatibilityResult, ChangeRequest
from src.evaluation.feedback_system import FeedbackLogger
import uuid

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
        # State variables
        user_os = gr.State("")
        detected_os = gr.State("")
        session_id = gr.State("")
        last_query = gr.State("")
        last_results = gr.State("")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""
                <div class='main-title'>AI Compliance Advisor</div>
                <div class='subtitle'>Analyze your software change requests for compatibility with your infrastructure</div>
                """)
                request_box = gr.Textbox(
                    label="Software Change Request",
                    placeholder="E.g. 'Upgrade Apache 2.4.50 and remove Tomcat 9.0 in production'",
                    lines=4,
                    elem_id="request-box"
                )
                analyze_btn = gr.Button("Run Analysis", elem_id="analyze-btn", variant="primary")
                
                # --- Results Section ---
                results_md = gr.Markdown(visible=False)

                # --- Feedback Section ---
                with gr.Column(visible=False) as feedback_container:
                    gr.Markdown("**Rate this analysis**", elem_id="feedback-header")
                    with gr.Row():
                        feedback_good_btn = gr.Button("👍 Looks Good")
                        feedback_bad_btn = gr.Button("👎 Needs Improvement")
                    feedback_thanks_md = gr.Markdown(visible=False)

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
            """Confirms the OS and updates the UI."""
            return {
                user_os: os_name,
                os_question_md: gr.update(visible=False),
                os_confirm_buttons: gr.update(visible=False),
                os_select_dd: gr.update(visible=False),
                os_confirmed_md: gr.update(value=f"✅ OS set to <b>{os_name}</b>", visible=True)
            }

        def show_os_select():
            """Shows the OS selection dropdown."""
            return {
                os_question_md: gr.update(visible=False),
                os_confirm_buttons: gr.update(visible=False),
                os_select_dd: gr.update(visible=True)
            }

        def on_analyze(request_text, current_os):
            """Main analysis function using multi-upgrade logic."""
            if not request_text.strip():
                return {results_md: gr.update(value="<div class='results-container section-empty'>⚠️ Please enter a software change request.</div>", visible=True)}

            # Use the new multi-upgrade parser and analyzer
            change_requests = analyzer.parse_multiple_change_requests(request_text)
            
            # Pass the confirmed OS to the analyzer
            results = analyzer.analyze_multiple_compatibility(change_requests, target_os=current_os)
            
            # Aggregate and format results
            output = [f"<div>Analysis based on OS: <b>{current_os}</b></div><br>" if current_os else ""]
            if not results:
                output.append("<div class='section-empty'>Could not parse any valid change requests.</div>")
            else:
                for cr, result in results:
                    # Header for each request
                    output.append(f"<h3 class='section-header'>Request: {cr.action.title()} {cr.software_name} {cr.version or ''}</h3>")
                    # Format each section
                    output.append(format_status(result))
                    output.append("<div class='section-header'>🗂️ Affected Models</div>")
                    output.append(format_affected_models(result))
                    if result.conflicts:
                        output.append("<div class='section-header'>⛔ Conflicts</div>")
                        output.append(format_list_section(result.conflicts, highlight=False))
                    if result.warnings:
                        output.append("<div class='section-header'>⚠️ Warnings</div>")
                        output.append(format_list_section(result.warnings, highlight=False))
                    if result.recommendations:
                        output.append("<div class='section-header'>💡 Recommendations</div>")
                        output.append(format_list_section(result.recommendations, highlight=True))
                    if result.alternative_versions:
                        output.append("<div class='section-header'>🔄 Alternative Versions</div>")
                        output.append(format_list_section(result.alternative_versions, highlight=False))
                    output.append("<hr style='margin: 2em 0; border: 1px solid #e0e7ef;'>")

            formatted_output = f"<div class='results-container'>{''.join(output)}</div>"
            
            return {
                results_md: gr.update(value=formatted_output, visible=True),
                feedback_container: gr.update(visible=True),
                feedback_thanks_md: gr.update(visible=False),
                last_query: request_text,
                last_results: formatted_output
            }

        def log_feedback(score, query, results, os, sid):
            """Logs feedback and shows a thank you message."""
            feedback_logger.log(
                query=query,
                generated_output=results,
                feedback_score=score,
                user_os=os,
                session_id=sid
            )
            return {
                feedback_container: gr.update(visible=False),
                feedback_thanks_md: gr.update(value="🙏 **Thank you for your feedback!**", visible=True)
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
        os_no_btn.click(show_os_select, outputs=[os_question_md, os_confirm_buttons, os_select_dd])
        os_select_dd.change(confirm_os, inputs=os_select_dd, outputs=[user_os, os_question_md, os_confirm_buttons, os_select_dd, os_confirmed_md])

        analyze_btn.click(
            on_analyze, 
            inputs=[request_box, user_os], 
            outputs=[results_md, feedback_container, feedback_thanks_md, last_query, last_results]
        )

        feedback_good_btn.click(
            lambda query, results, os, sid: log_feedback(1, query, results, os, sid),
            inputs=[last_query, last_results, user_os, session_id],
            outputs=[feedback_container, feedback_thanks_md]
        )
        feedback_bad_btn.click(
            lambda query, results, os, sid: log_feedback(0, query, results, os, sid),
            inputs=[last_query, last_results, user_os, session_id],
            outputs=[feedback_container, feedback_thanks_md]
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Use standard Gradio port
        share=False,            # Don't create public link
        debug=False,            # Disable debug mode for production
        show_error=True,        # Show errors in the interface
        quiet=False             # Show startup messages
    )