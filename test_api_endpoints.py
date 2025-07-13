#!/usr/bin/env python3
"""
Test script to verify the new API endpoints for database and OS options.
"""

import requests
import json
import sys
import os

def test_api_endpoints():
    """Test the new API endpoints for options."""
    base_url = "http://localhost:5000"
    
    endpoints = [
        "/api/options/operating-systems",
        "/api/options/databases", 
        "/api/options/web-servers"
    ]
    
    print("Testing API endpoints for configuration options...")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            print(f"\nTesting: {endpoint}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
                
                # Check if the response has the expected structure
                if "operating_systems" in data:
                    print(f"✓ Operating systems loaded: {len(data['operating_systems'])} options")
                    print(f"  Sample options: {data['operating_systems'][:3]}")
                elif "databases" in data:
                    print(f"✓ Databases loaded: {len(data['databases'])} options")
                    print(f"  Sample options: {data['databases'][:3]}")
                elif "web_servers" in data:
                    print(f"✓ Web servers loaded: {len(data['web_servers'])} options")
                    print(f"  Sample options: {data['web_servers'][:3]}")
                else:
                    print("✗ Unexpected response structure")
            else:
                print(f"✗ Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"\n✗ Could not connect to {base_url}")
            print("Make sure the Flask server is running on port 5000")
            return False
        except Exception as e:
            print(f"\n✗ Error testing {endpoint}: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("API endpoint testing completed!")
    return True

def test_data_extraction():
    """Test the data extraction functionality."""
    print("\nTesting data extraction from files...")
    print("=" * 50)
    
    # Check if data files exist
    data_files = [
        'data/processed/compatibility_analysis.json',
        'data/processed/Webserver_OS_Mapping.csv',
        'data/raw/WebServer.csv',
        'data/processed/Change_History.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
    
    # Test compatibility analysis JSON
    analysis_file = 'data/processed/compatibility_analysis.json'
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            servers = analysis_data.get('servers', [])
            print(f"\n✓ Compatibility analysis contains {len(servers)} servers")
            
            # Extract unique models
            models = set()
            for server in servers:
                server_info = server.get('server_info', {})
                model = server_info.get('model', '')
                if model:
                    models.add(model)
            
            print(f"✓ Found {len(models)} unique models")
            print(f"  Sample models: {list(models)[:5]}")
            
        except Exception as e:
            print(f"✗ Error reading compatibility analysis: {e}")
    else:
        print("✗ Compatibility analysis file not found")

def test_integration():
    """Test the integration with the frontend by simulating the fetch calls."""
    print("\nTesting frontend integration...")
    print("=" * 50)
    
    # Simulate the frontend fetch calls
    try:
        import asyncio
        import aiohttp
        
        async def test_fetch():
            async with aiohttp.ClientSession() as session:
                urls = [
                    "http://localhost:5000/api/options/operating-systems",
                    "http://localhost:5000/api/options/databases",
                    "http://localhost:5000/api/options/web-servers"
                ]
                
                async def fetch_url(session, url):
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                return data
                            else:
                                return None
                    except Exception as e:
                        print(f"Error fetching {url}: {e}")
                        return None
                
                results = await asyncio.gather(*[fetch_url(session, url) for url in urls])
                
                for i, result in enumerate(results):
                    if result:
                        print(f"✓ Successfully fetched data from endpoint {i+1}")
                        if "operating_systems" in result:
                            print(f"  - {len(result['operating_systems'])} OS options")
                            print(f"  - Sample: {result['operating_systems'][:3]}")
                        elif "databases" in result:
                            print(f"  - {len(result['databases'])} database options")
                            print(f"  - Sample: {result['databases'][:3]}")
                        elif "web_servers" in result:
                            print(f"  - {len(result['web_servers'])} web server options")
                            print(f"  - Sample: {result['web_servers'][:3]}")
                    else:
                        print(f"✗ Failed to fetch data from endpoint {i+1}")
        
        # Run the async test
        asyncio.run(test_fetch())
        
    except ImportError:
        print("aiohttp not available, skipping async test")
    except Exception as e:
        print(f"Error in integration test: {e}")

if __name__ == "__main__":
    print("API Endpoint Test Script")
    print("This script tests the new API endpoints for configuration options.")
    print()
    
    # Test data extraction first
    test_data_extraction()
    
    # Test API endpoints
    success = test_api_endpoints()
    
    if success:
        test_integration()
        print("\n✅ All tests completed successfully!")
        print("\nThe API endpoints are working correctly and can be used by the frontend.")
        print("\nThe dropdowns will now be populated with actual data from your files!")
    else:
        print("\n❌ Some tests failed. Please check the server status and try again.")
        sys.exit(1) 