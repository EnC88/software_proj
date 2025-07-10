#!/usr/bin/env python3
"""
Integrated Application Runner
Builds the React frontend and starts the Flask backend
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    # Check if Node.js is installed
    try:
        subprocess.run(['node', '--version'], check=True, capture_output=True)
        print("✅ Node.js found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found. Please install Node.js from https://nodejs.org/")
        return False
    
    # Check if npm is installed
    try:
        subprocess.run(['npm', '--version'], check=True, capture_output=True)
        print("✅ npm found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ npm not found. Please install npm")
        return False
    
    return True

def build_frontend():
    """Build the React frontend."""
    print("🔨 Building React frontend...")
    
    frontend_dir = Path("templates")
    if not frontend_dir.exists():
        print("❌ Templates directory not found")
        return False
    
    try:
        # Install dependencies
        print("📦 Installing frontend dependencies...")
        subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
        
        # Build the frontend
        print("🏗️ Building frontend...")
        subprocess.run(['npm', 'run', 'build'], cwd=frontend_dir, check=True)
        
        print("✅ Frontend built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend build failed: {e}")
        return False

def start_backend():
    """Start the Flask backend."""
    print("🚀 Starting Flask backend...")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['FLASK_ENV'] = 'development'
        env['PORT'] = '5000'
        
        # Start the Flask app
        subprocess.run([
            sys.executable, 'src/api/app.py'
        ], env=env, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend start failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
        return True

def main():
    """Main function to run the integrated application."""
    print("🎯 Starting Integrated System Compatibility Assistant")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependencies not met. Exiting.")
        sys.exit(1)
    
    # Build frontend
    if not build_frontend():
        print("❌ Frontend build failed. Exiting.")
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("📱 Frontend: http://localhost:5000")
    print("🔧 API: http://localhost:5000/api")
    print("📊 Health Check: http://localhost:5000/api/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main() 