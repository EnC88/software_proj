#!/usr/bin/env python3
"""
Integration Test Script
Tests the Flask API endpoints and verifies the integration
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000/api"
TEST_QUERIES = [
    "Upgrade Apache to 2.4.50",
    "Install MySQL 8.0 on Windows Server 2019",
    "Check compatibility with Ubuntu 22.04",
    "What servers are compatible with PostgreSQL 15?"
]

def test_health_endpoint():
    """Test the health check endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_analyze_endpoint():
    """Test the analyze endpoint."""
    print("\n🔍 Testing analyze endpoint...")
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n  Test {i}: {query}")
        try:
            data = {
                "query": query,
                "sessionId": f"test_session_{i}",
                "userOS": "test"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"    ✅ Analysis successful")
                    print(f"    📊 Results: {len(result.get('results', []))} analysis items")
                else:
                    print(f"    ⚠️ Analysis returned success=false")
            else:
                print(f"    ❌ Analysis failed: {response.status_code}")
                print(f"    📝 Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Request error: {e}")

def test_analytics_endpoint():
    """Test the analytics endpoint."""
    print("\n🔍 Testing analytics endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/analytics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analytics retrieved:")
            print(f"  📊 Total queries: {data.get('total_queries', 0)}")
            print(f"  📈 Positive rate: {data.get('positive_rate', 0)}%")
            print(f"  📅 Recent queries: {data.get('recent_queries', 0)}")
        else:
            print(f"❌ Analytics failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Analytics error: {e}")

def test_suggestions_endpoint():
    """Test the suggestions endpoint."""
    print("\n🔍 Testing suggestions endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/suggestions", timeout=10)
        if response.status_code == 200:
            data = response.json()
            suggestions = data.get('quick_actions', [])
            print(f"✅ Suggestions retrieved: {len(suggestions)} quick actions")
            for suggestion in suggestions[:3]:  # Show first 3
                print(f"  💡 {suggestion.get('text', 'Unknown')}")
        else:
            print(f"❌ Suggestions failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Suggestions error: {e}")

def test_frontend_access():
    """Test if the frontend is accessible."""
    print("\n🔍 Testing frontend access...")
    try:
        response = requests.get("http://localhost:5000/", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend access failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend access error: {e}")
        return False

def check_server_status():
    """Check if the server is running."""
    print("🔍 Checking server status...")
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Run all integration tests."""
    print("🧪 System Compatibility Assistant - Integration Tests")
    print("=" * 60)
    
    # Check if server is running
    if not check_server_status():
        print("❌ Server is not running!")
        print("💡 Start the server with: python run_integrated_app.py")
        print("💡 Or: python src/api/app.py")
        sys.exit(1)
    
    print("✅ Server is running")
    
    # Run tests
    test_health_endpoint()
    test_analyze_endpoint()
    test_analytics_endpoint()
    test_suggestions_endpoint()
    test_frontend_access()
    
    print("\n🎉 Integration tests completed!")
    print("=" * 60)
    print("📱 Frontend: http://localhost:5000")
    print("🔧 API: http://localhost:5000/api")
    print("📊 Health: http://localhost:5000/api/health")

if __name__ == "__main__":
    main() 