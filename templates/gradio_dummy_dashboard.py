import gradio as gr
import plotly.graph_objects as go

def dummy_chatbot(message, history):
    return history + [[message, "This is a dummy response."]]

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

custom_css = """
body, .gradio-container {
    background: #f6f8fa !important;
    font-family: 'Inter', 'Roboto', 'Segoe UI', 'Arial', sans-serif !important;
    color: #1e293b !important;
}
.gr-block.gr-box, .stat-card {
    background: #fff !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 12px #e0e7ef !important;
    border: none !important;
}
.stat-card {
    padding: 1.5em;
    text-align: center;
    font-size: 1.3em;
    margin: 0.5em 0.5em 0.5em 0;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.stat-label {
    color: #64748b;
    font-size: 1em;
    margin-top: 0.2em;
}
.stat-value {
    font-size: 2.2em;
    font-weight: bold;
    margin-bottom: 0.2em;
    color: #2563eb;
}
.quick-btn button, .quick-btn {
    background: #2563eb !important;
    color: #fff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    margin-bottom: 0.5em !important;
    border: none !important;
    box-shadow: 0 1px 4px #e0e7ef !important;
    transition: background 0.2s;
}
.quick-btn button:hover, .quick-btn:hover {
    background: #1e40af !important;
}
.gr-input, .gr-textbox, .gr-dropdown {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    background: #f8fafc !important;
    color: #1e293b !important;
}
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #1e293b !important;
    font-weight: 700 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            <div style='font-size:1.3em;font-weight:bold;margin-bottom:0.5em;color:#1e293b;'>System Configuration</div>
            <div style='color:#64748b;margin-bottom:1em;'>Optional - helps provide targeted recommendations</div>
            """)
            os_dd = gr.Dropdown(["Windows", "Linux", "macOS"], label="Operating System")
            db_dd = gr.Dropdown(["PostgreSQL", "MySQL", "MongoDB"], label="Database")
            ws_dd = gr.Dropdown(["Apache", "Nginx", "IIS"], label="Web Servers")
        with gr.Column(scale=2):
            gr.Markdown("""
            <div style='font-size:1.3em;font-weight:bold;margin-bottom:0.5em;color:#1e293b;'>System Compatibility Assistant</div>
            <div style='color:#64748b;margin-bottom:1em;'>Ask questions about OS, databases, and web servers</div>
            """)
            with gr.Row():
                gr.Button("Check OS compatibility", elem_classes="quick-btn")
                gr.Button("Database requirements", elem_classes="quick-btn")
                gr.Button("Web server setup", elem_classes="quick-btn")
            chatbot = gr.Chatbot(label="System Compatibility Assistant")
            user_input = gr.Textbox(label="Ask about system compatibility...", lines=1)
            user_input.submit(dummy_chatbot, [user_input, chatbot], chatbot)

    gr.Markdown("---")
    gr.Markdown("""
    <div style='font-size:2em;font-weight:bold;text-align:center;margin-bottom:0.2em;color:#1e293b;'>System Compatibility Assistant</div>
    <div style='text-align:center;color:#64748b;margin-bottom:2em;'>Enterprise-grade system compatibility analysis and recommendations</div>
    """)
    with gr.Row():
        gr.Markdown("""<div class='stat-card'><div class='stat-value'>12,847</div><div class='stat-label'>Total Queries</div></div>""")
        gr.Markdown("""<div class='stat-card'><div class='stat-value'>1,247</div><div class='stat-label'>Active Systems</div></div>""")
        gr.Markdown("""<div class='stat-card'><div class='stat-value'>94.2%</div><div class='stat-label'>Compatibility Score</div></div>""")
        gr.Markdown("""<div class='stat-card'><div class='stat-value'>1.2s</div><div class='stat-label'>Response Time</div></div>""")
    with gr.Row():
        gr.Plot(plot_query_trends, label="Query Volume Trends")
        gr.Plot(plot_os_compat, label="OS Compatibility Rates")

demo.launch() 