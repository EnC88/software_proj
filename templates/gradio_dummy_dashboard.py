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
.custom-dropdown {
    appearance: none;
    transition: border 0.2s, background 0.2s;
}
.custom-dropdown:hover, .custom-dropdown:focus {
    border: 1.5px solid #2563eb !important;
    background: #f1f5f9 !important;
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
                <div style='position: relative; width: 100%; margin-bottom: 2em;'>
                  <select class="custom-dropdown" style='width: 100%; padding: 0.9em 2.5em 0.9em 1em; border-radius: 0; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; appearance: none;'>
                    <option>Select operating system</option>
                    <option>Windows</option>
                    <option>Linux</option>
                    <option>macOS</option>
                  </select>
                  <span style='position: absolute; right: 1.2em; top: 50%; transform: translateY(-50%); pointer-events: none; font-size: 1.2em; color: #64748b;'>▼</span>
                </div>
                <div style='font-weight: 600; font-size: 1.1em; margin-bottom: 0.7em; color: #22c55e; display: flex; align-items: center; gap: 0.5em;'>
                    <span class='icon-box' style='background: #e0e7ef;'>
                        <!-- Material Design Database Icon (Stack) -->
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 5v6c0 1.657-4.03 3-9 3s-9-1.343-9-3V5"/><path d="M21 11v6c0 1.657-4.03 3-9 3s-9-1.343-9-3v-6"/></svg>
                    </span>
                    Database
                </div>
                <div style='position: relative; width: 100%; margin-bottom: 2em;'>
                  <select class="custom-dropdown" style='width: 100%; padding: 0.9em 2.5em 0.9em 1em; border-radius: 0; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; appearance: none;'>
                    <option>Select database</option>
                    <option>PostgreSQL</option>
                    <option>MySQL</option>
                    <option>MongoDB</option>
                  </select>
                  <span style='position: absolute; right: 1.2em; top: 50%; transform: translateY(-50%); pointer-events: none; font-size: 1.2em; color: #64748b;'>▼</span>
                </div>
                <div style='font-weight: 600; font-size: 1.1em; margin-bottom: 0.7em; color: #f59e0b; display: flex; align-items: center; gap: 0.5em;'>
                    <span class='icon-box' style='background: #e0e7ef;'>
                        <!-- Material Design Server Icon (Web Server) -->
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><path d="M6 6h.01M6 18h.01"/></svg>
                    </span>
                    Web Servers
                </div>
                <div style='position: relative; width: 100%; margin-bottom: 0.2em;'>
                  <select class="custom-dropdown" style='width: 100%; padding: 0.9em 2.5em 0.9em 1em; border-radius: 0; border: 1.5px solid #e5e7eb; font-size: 1.1em; color: #1e293b; background: #fff; font-family: Inter, Roboto, Segoe UI, Arial, sans-serif; font-weight: 500; outline: none; box-shadow: none; appearance: none;'>
                    <option>Select web servers</option>
                    <option>Apache</option>
                    <option>Nginx</option>
                    <option>IIS</option>
                  </select>
                  <span style='position: absolute; right: 1.2em; top: 50%; transform: translateY(-50%); pointer-events: none; font-size: 1.2em; color: #64748b;'>▼</span>
                </div>
            </div>
            """)
        with gr.Column(scale=2):
            gr.HTML("""
            <!-- Main Card -->
            <div style='background: #fff; border-radius: 12px; box-shadow: 0 2px 8px #1e293b10; padding: 0; margin-bottom: 1.2em; max-width: 620px;'>
                <!-- Header -->
                <div style='background: linear-gradient(90deg, #1e293b 80%, #22304a 100%); border-radius: 12px 12px 0 0; padding: 1em 1.5em 0.7em 1.5em; color: #fff; display: flex; align-items: center; gap: 0.7em;'>
                    <span style='background: #22304a; border-radius: 6px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;'>
                        <!-- Chat bubble icon -->
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    </span>
                    <div>
                        <div style='font-size: 1.08em; font-weight: 700; letter-spacing: -0.01em;'>System Compatibility Assistant</div>
                        <div style='color: #cbd5e1; font-size: 0.97em; margin-top: 0.05em;'>Ask questions about OS, databases, and web servers</div>
                    </div>
                </div>
                <!-- Quick Start -->
                <div style='padding: 0.8em 1.5em 0.5em 1.5em;'>
                    <div style='font-size: 0.93em; color: #64748b; font-weight: 700; margin-bottom: 0.7em; letter-spacing: 0.04em;'>QUICK START</div>
                    <div style='display: flex; flex-direction: column; gap: 0.5em;'>
                        <button style='background: #f4f8ff; color: #2563eb; border: 1px solid #dbeafe; border-radius: 8px; display: flex; align-items: center; gap: 0.7em; font-weight: 700; font-size: 1em; padding: 0.7em 0.7em 0.7em 1em; width: 100%; justify-content: flex-start;'>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2.5" y="4" width="19" height="12" rx="2.5"/><path d="M2 20h20"/></svg>
                            <span style='font-weight: 700; color: #2563eb;'>Check OS compatibility</span>
                        </button>
                        <button style='background: #f3fcf6; color: #22c55e; border: 1px solid #bbf7d0; border-radius: 8px; display: flex; align-items: center; gap: 0.7em; font-weight: 700; font-size: 1em; padding: 0.7em 0.7em 0.7em 1em; width: 100%; justify-content: flex-start;'>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5.5" rx="9.5" ry="3.5"/><path d="M21.5 5.5v6c0 2-4.5 3.5-9.5 3.5s-9.5-1.5-9.5-3.5v-6"/><path d="M21.5 11.5v6c0 2-4.5 3.5-9.5 3.5s-9.5-1.5-9.5-3.5v-6"/></svg>
                            <span style='font-weight: 700; color: #22c55e;'>Database requirements</span>
                        </button>
                        <button style='background: #fff7ed; color: #f59e0b; border: 1px solid #fde68a; border-radius: 8px; display: flex; align-items: center; gap: 0.7em; font-weight: 700; font-size: 1em; padding: 0.7em 0.7em 0.7em 1em; width: 100%; justify-content: flex-start;'>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2.5" y="2.5" width="19" height="8.5" rx="2.5"/><rect x="2.5" y="13" width="19" height="8.5" rx="2.5"/><path d="M6 6.5h.01M6 17.5h.01"/></svg>
                            <span style='font-weight: 700; color: #f59e0b;'>Web server setup</span>
                        </button>
                        <!-- Chat Card -->
                        <div style='background: #fff; border-radius: 8px; box-shadow: 0 1px 4px #e0e7ef; padding: 1em 1em 0.6em 1em; width: 100%; margin-top: 0.5em;'>
                            <div style='display: flex; flex-direction: column; align-items: flex-start;'>
                                <div style='font-size: 0.93em; color: #64748b; font-weight: 700; margin-bottom: 0.3em; display: flex; align-items: center;'>
                                    <span style='background: #e0e7ef; color: #475569; border-radius: 999px; padding: 0.12em 0.8em; font-size: 0.9em; font-weight: 700; margin-right: 0.5em;'>GENERAL</span>
                                    <span style='color: #1e293b; font-weight: 700;'>System Compatibility Assistant</span>
                                </div>
                                <div style='background: #f6f8fa; color: #1e293b; border-radius: 8px; padding: 0.7em 0.9em; width: 96%; box-shadow: 0 1px 4px #e0e7ef; font-size: 0.97em; margin-bottom: 1em;'>Hello! I'm your System Compatibility Assistant. I can help you with questions about operating systems, databases, and web servers. What would you like to know?</div>
                                <div style='color: #b6c2d6; font-size: 0.93em; margin-top: 0.1em;'>02:42 PM</div>
                            </div>
                        </div>
                        <!-- Input Row -->
                        <form id='chat-form' style='display: flex; align-items: center; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; box-shadow: 0 1px 4px #e0e7ef; padding: 0.7em 1em; width: 100%; margin-top: 0.5em;'>
                            <input id='chat-input' type='text' placeholder='Ask about system compatibility...' style='flex: 1; border: none; outline: none; font-size: 1em; background: transparent; color: #1e293b; padding: 0.5em 0; border-radius: 4px;'/>
                            <button type='submit' style='background: #2563eb; border: none; border-radius: 6px; padding: 0.4em 0.7em; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 1.1em; cursor: pointer; width: 38px; height: 38px;'>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                            </button>
                        </form>
                    </div>
                </div>
            </div>
            <script>
            // Dummy chat logic for demo (no bubbles, just input clear)
            const chatForm = document.getElementById('chat-form');
            const chatInput = document.getElementById('chat-input');
            chatForm.onsubmit = function(e) {
                e.preventDefault();
                chatInput.value = '';
            };
            </script>
            """)

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