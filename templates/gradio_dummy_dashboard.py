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
.stat-card {
    background: #fff !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 12px #e0e7ef !important;
    border: none !important;
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
            gr.HTML("""
            <div style='background: linear-gradient(90deg, #1e293b 80%, #22304a 100%); border-radius: 20px 20px 0 0; padding: 1.5em 1.5em 1em 1.5em; color: #fff; box-shadow: 0 2px 12px #e0e7ef; display: flex; align-items: center; gap: 1em;'>
                <span style='background: #22304a; border-radius: 12px; padding: 0.6em; display: flex; align-items: center; justify-content: center;'>
                    <svg width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="8" fill="#22304a"/><path d="M14 8a6 6 0 100 12 6 6 0 000-12zm0 10.5A4.5 4.5 0 1114 9.5a4.5 4.5 0 010 9z" fill="#fff"/><path d="M14 12.25a1.25 1.25 0 100 2.5 1.25 1.25 0 000-2.5z" fill="#cbd5e1"/></svg>
                </span>
                <div>
                    <div style='font-size: 1.5em; font-weight: bold;'>System Configuration</div>
                    <div style='color: #cbd5e1; font-size: 1em; margin-top: 0.2em;'>Optional - helps provide targeted recommendations</div>
                </div>
            </div>
            <div style='background: #fff; border-radius: 0 0 20px 20px; padding: 2em 1.5em 1.5em 1.5em; box-shadow: 0 2px 12px #e0e7ef;'>
                <div style='font-weight: 600; font-size: 1.1em; margin-bottom: 0.7em; color: #2563eb; display: flex; align-items: center; gap: 0.5em;'>
                    <svg width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;"><path d="M10 2a8 8 0 100 16 8 8 0 000-16zm0 14.5A6.5 6.5 0 1110 3.5a6.5 6.5 0 010 13z" fill="#2563eb"/></svg>
                    Operating System
                </div>
                <select style='width: 100%; padding: 0.9em 1em; border-radius: 12px; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; margin-bottom: 2em;'>
                    <option>Select operating system</option>
                    <option>Windows</option>
                    <option>Linux</option>
                    <option>macOS</option>
                </select>
                <div style='font-weight: 600; font-size: 1.1em; margin-bottom: 0.7em; color: #22c55e; display: flex; align-items: center; gap: 0.5em;'>
                    <svg width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;"><ellipse cx="10" cy="10" rx="8" ry="6" fill="#22c55e"/></svg>
                    Database
                </div>
                <select style='width: 100%; padding: 0.9em 1em; border-radius: 12px; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; margin-bottom: 2em;'>
                    <option>Select database</option>
                    <option>PostgreSQL</option>
                    <option>MySQL</option>
                    <option>MongoDB</option>
                </select>
                <div style='font-weight: 600; font-size: 1.1em; margin-bottom: 0.7em; color: #f59e0b; display: flex; align-items: center; gap: 0.5em;'>
                    <svg width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;"><rect x="2" y="6" width="16" height="8" rx="2" fill="#f59e0b"/></svg>
                    Web Servers
                </div>
                <select style='width: 100%; padding: 0.9em 1em; border-radius: 12px; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; margin-bottom: 0.2em;'>
                    <option>Select web servers</option>
                    <option>Apache</option>
                    <option>Nginx</option>
                    <option>IIS</option>
                </select>
            </div>
            """)
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