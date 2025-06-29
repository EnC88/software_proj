#!/usr/bin/env python3
"""
Run the Infrastructure Search Assistant Interface
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import and run the interface
from templates.landing import build_interface

if __name__ == "__main__":
    print("🚀 Starting Infrastructure Search Assistant...")
    print("📊 Loading vector store and data...")
    
    # Build and launch the interface
    demo = build_interface()
    
    print("✅ Interface ready!")
    print("🌐 Opening in your browser...")
    print("💡 Try asking questions like:")
    print("   - 'What servers are running Apache HTTPD?'")
    print("   - 'What OS does Apache run best on?'")
    print("   - 'Which servers are in Production?'")
    print("   - 'Show me Dell servers'")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=8501,
        share=False,
        debug=True,
        show_error=True
    ) 