import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.determine_recs import CompatibilityAnalyzer, CompatibilityResult, ChangeRequest

analyzer = CompatibilityAnalyzer()

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
    .gr-accordion {margin-bottom: 1em;}
    """) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""
                <div class='main-title'>AI Compliance Advisor</div>
                <div class='subtitle'>Analyze your software change requests for compatibility with your infrastructure</div>
                """)
                request_box = gr.Textbox(
                    label="Software Change Request",
                    placeholder="E.g. 'Upgrade Apache to 2.4.50 in production'",
                    lines=4,
                    elem_id="request-box"
                )
                analyze_btn = gr.Button("Run Analysis", elem_id="analyze-btn", variant="primary")
                
                # --- Results Section ---
                with gr.Column(visible=False, elem_id="results-container") as results_container:
                    gr.Markdown("<div class='section-header'>🟢 Status</div>", elem_id="status-header")
                    status_md = gr.Markdown()
                    gr.Markdown("<div class='section-header'>🗂️ Affected Models</div>", elem_id="affected-header")
                    affected_md = gr.Markdown()
                    conflicts_md = gr.Markdown(visible=False)
                    warnings_md = gr.Markdown(visible=False)
                    recs_md = gr.Markdown(visible=False)
                    alts_md = gr.Markdown(visible=False)

            with gr.Column(scale=1):
                gr.Markdown("""
                <h4>Instructions</h4>
                <ul>
                  <li>Describe your software change (upgrade, install, remove, etc.)</li>
                  <li>Mention software name, version, and environment if possible</li>
                  <li>Click <b>Run Analysis</b> to see compatibility results</li>
                </ul>
                """)

        def on_analyze(request_text):
            if not request_text.strip():
                return {results_container: gr.update(visible=True), status_md: gr.update(value="⚠️ Please enter a software change request.")}

            change_request = analyzer.parse_change_request(request_text)
            result = analyzer.analyze_compatibility(change_request)

            # Recommendations get special formatting
            recs_html = format_list_section(result.recommendations, icon="<span style='color:#0ea5e9;'>💡</span>", highlight=True) if result.recommendations else ""
            warnings_html = format_list_section(result.warnings, icon="<span style='color:#f59e42;'>⚠️</span>") if result.warnings else ""
            conflicts_html = format_list_section(result.conflicts, icon="<span style='color:#ef4444;'>⛔</span>") if result.conflicts else ""
            alts_html = format_list_section(result.alternative_versions, icon="<span style='color:#64748b;'>🔄</span>") if result.alternative_versions else ""

            return {
                results_container: gr.update(visible=True),
                status_md: gr.update(value=format_status(result)),
                affected_md: gr.update(value=format_affected_models(result)),
                conflicts_md: gr.update(value=conflicts_html, visible=bool(result.conflicts)),
                warnings_md: gr.update(value=warnings_html, visible=bool(result.warnings)),
                recs_md: gr.update(value=recs_html, visible=bool(result.recommendations)),
                alts_md: gr.update(value=alts_html, visible=bool(result.alternative_versions)),
            }

        analyze_btn.click(
            on_analyze, 
            inputs=request_box, 
            outputs=[results_container, status_md, affected_md, conflicts_md, warnings_md, recs_md, alts_md]
        )
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()