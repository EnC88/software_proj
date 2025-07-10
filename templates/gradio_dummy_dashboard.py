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
    border-radius: 0 !important;
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
    border-radius: 0 !important;
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
    border-radius: 0 !important;
    border: 1px solid #cbd5e1 !important;
    background: #f8fafc !important;
    color: #1e293b !important;
}
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #1e293b !important;
    font-weight: 700 !important;
}
.icon-box {
    background: #22304a;
    border-radius: 0;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style='background: linear-gradient(90deg, #1e293b 80%, #22304a 100%); border-radius: 0; padding: 1.5em 1.5em 1em 1.5em; color: #fff; box-shadow: 0 2px 12px #e0e7ef; display: flex; align-items: center; gap: 1em;'>
                <span class='icon-box'>
                    <!-- Material Design Gear Icon (Settings) -->
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19.14,12.94a7.07,7.07,0,0,0,0-1.88l2.11-1.65a.5.5,0,0,0,.12-.66l-2-3.46a.5.5,0,0,0-.61-.22l-2.49,1a7,7,0,0,0-1.6-.93l-.38-2.65A.5.5,0,0,0,13,2h-2a.5.5,0,0,0-.5.42l-.38,2.65a7,7,0,0,0-1.6.93l-2.49-1a.5.5,0,0,0-.61.22l-2,3.46a.5.5,0,0,0,.12.66l2.11,1.65a7.07,7.07,0,0,0,0,1.88L2.27,14.59a.5.5,0,0,0-.12.66l2,3.46a.5.5,0,0,0,.61.22l2.49-1a7,7,0,0,0,1.6.93l.38,2.65A.5.5,0,0,0,11,22h2a.5.5,0,0,0,.5-.42l.38-2.65a7,7,0,0,0,1.6-.93l2.49,1a.5.5,0,0,0,.61-.22l2-3.46a.5.5,0,0,0-.12-.66ZM12,15.5A3.5,3.5,0,1,1,15.5,12,3.5,3.5,0,0,1,12,15.5Z"/></svg>
                </span>
                <div>
                    <div style='font-size: 1.5em; font-weight: bold;'>System Configuration</div>
                    <div style='color: #cbd5e1; font-size: 1em; margin-top: 0.2em;'>Optional - helps provide targeted recommendations</div>
                </div>
            </div>
            <div style='background: #fff; border-radius: 0; padding: 2em 1.5em 1.5em 1.5em; box-shadow: 0 2px 12px #e0e7ef;'>
                <div style='font-weight: 600; font-size: 1.1em; margin-bottom: 0.7em; color: #2563eb; display: flex; align-items: center; gap: 0.5em;'>
                    <span class='icon-box' style='background: #e0e7ef;'>
                        <!-- Material Design Laptop Icon (OS) -->
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M2 20h20"/></svg>
                    </span>
                    Operating System
                </div>
                <select style='width: 100%; padding: 0.9em 1em; border-radius: 0; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; margin-bottom: 2em;'>
                    <option>Select operating system</option>
                    <option>Windows</option>
                    <option>Linux</option>
                    <option>macOS</option>
                </select>
                <div style='font-weight: 600; font-size: 1.1em; margin-bottom: 0.7em; color: #22c55e; display: flex; align-items: center; gap: 0.5em;'>
                    <span class='icon-box' style='background: #e0e7ef;'>
                        <!-- Material Design Database Icon (Stack) -->
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 5v6c0 1.657-4.03 3-9 3s-9-1.343-9-3V5"/><path d="M21 11v6c0 1.657-4.03 3-9 3s-9-1.343-9-3v-6"/></svg>
                    </span>
                    Database
                </div>
                <select style='width: 100%; padding: 0.9em 1em; border-radius: 0; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; margin-bottom: 2em;'>
                    <option>Select database</option>
                    <option>PostgreSQL</option>
                    <option>MySQL</option>
                    <option>MongoDB</option>
                </select>
                <div style='font-weight: 600; font-size: 1.1em; margin-bottom: 0.7em; color: #f59e0b; display: flex; align-items: center; gap: 0.5em;'>
                    <span class='icon-box' style='background: #e0e7ef;'>
                        <!-- Material Design Server Icon (Web Server) -->
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><path d="M6 6h.01M6 18h.01"/></svg>
                    </span>
                    Web Servers
                </div>
                <select style='width: 100%; padding: 0.9em 1em; border-radius: 0; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; margin-bottom: 0.2em;'>
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