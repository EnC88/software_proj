import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.determine_recs import CompatibilityAnalyzer

analyzer = CompatibilityAnalyzer()

def analyze_request(request_text: str) -> str:
    if not request_text.strip():
        return "⚠️ Please enter a software change request."
    change_request = analyzer.parse_change_request(request_text)
    result = analyzer.analyze_compatibility(change_request)
    formatted = analyzer.format_analysis_result(result, change_request)
    return formatted

def build_interface():
    with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background: #f8fafc;} .main-title {font-size: 2.5rem; font-weight: bold; color: #1e293b;} .subtitle {font-size: 1.2rem; color: #64748b;} .result-markdown {background: #fff; border-radius: 12px; padding: 1.5em; box-shadow: 0 2px 8px #e0e7ef;} .sidebar-logo {height: 60px; margin-bottom: 1em;}") as demo:
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
                result_md = gr.Markdown("", elem_id="result-markdown", visible=False)
            with gr.Column(scale=1):
                gr.Markdown("""
                <img src='https://placehold.co/120x60?text=LOGO' class='sidebar-logo'/>
                <h4>Instructions</h4>
                <ul>
                  <li>Describe your software change (upgrade, install, remove, etc.)</li>
                  <li>Mention software name, version, and environment if possible</li>
                  <li>Click <b>Run Analysis</b> to see compatibility results</li>
                </ul>
                """)
        def on_analyze(request_text):
            result = analyze_request(request_text)
            return gr.update(value=result, visible=True)
        analyze_btn.click(on_analyze, inputs=request_box, outputs=result_md)
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()