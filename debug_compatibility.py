#!/usr/bin/env python3
"""Debug script for compatibility analysis."""

import logging
import json
from pathlib import Path
from src.rag.determine_recs import CheckCompatibility, ChangeRequest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_compatibility():
    """Debug the compatibility analysis step by step."""
    
    # Initialize the analyzer
    analyzer = CheckCompatibility()
    
    # Create a test change request
    cr = ChangeRequest(
        software_name='APACHE',
        version='2.4.50',
        action='upgrade',
        raw_text='upgrade apache to 2.4.50'
    )
    
    print("=== DEBUGGING COMPATIBILITY ANALYSIS ===")
    print(f"Change request: {cr.software_name} {cr.version} ({cr.action})")
    
    # Check the analysis data
    analysis = analyzer.analysis
    servers = analysis.get('servers', [])
    print(f"\nTotal servers in analysis: {len(servers)}")
    
    # Look at some sample server data
    print("\n=== SAMPLE SERVER DATA ===")
    for i, server in enumerate(servers[:5]):
        server_info = server.get('server_info', {})
        model = server_info.get('model', 'Unknown')
        product_type = server_info.get('product_type', 'Unknown')
        env = server.get('environment', 'Unknown')
        print(f"Server {i+1}: Model='{model}', ProductType='{product_type}', Env='{env}'")
    
    # Check what software families we're looking for
    print(f"\n=== SOFTWARE FAMILIES ===")
    print(f"Known software families: {analyzer.known_software_families}")
    
    # Test the RAG query
    print(f"\n=== RAG QUERY TEST ===")
    try:
        dependency_query = f"What are the dependencies and compatible software for {cr.software_name}?"
        rag_results = analyzer.query_engine.query(dependency_query, top_k=3)
        print(f"RAG query returned {len(rag_results) if rag_results else 0} results")
        if rag_results:
            print(f"RAG similarity scores: {[r.get('similarity_score', 0.0) for r in rag_results]}")
            context = analyzer.query_engine.format_results_for_llm(rag_results)
            print(f"RAG context: {context[:500]}...")
    except Exception as e:
        print(f"RAG query error: {e}")
    
    # Test the server filtering
    print(f"\n=== SERVER FILTERING TEST ===")
    affected_servers = servers  # No environment filter for testing
    print(f"Servers to check: {len(affected_servers)}")
    
    # Test the software matching logic
    print(f"\n=== SOFTWARE MATCHING TEST ===")
    software_to_show_families = analyzer.known_software_families
    print(f"Looking for any of: {software_to_show_families}")
    
    existing_installations = []
    for i, server in enumerate(affected_servers[:10]):  # Check first 10 servers
        model_full = server.get('server_info', {}).get('model', 'Unknown')
        if model_full == 'Unknown':
            continue
        
        env = server.get('environment', 'Unknown')
        
        # Extract product family and version
        import re
        version_match = re.search(r'(\d+(?:\.\d+)*)', model_full)
        version = version_match.group(1) if version_match else "0.0"
        product_family = model_full[:version_match.start()].strip().upper() if version_match else model_full.upper()
        
        print(f"Server {i+1}: Model='{model_full}' -> Family='{product_family}', Version='{version}'")
        
        # Check if this product family matches any known family
        for family_to_show in software_to_show_families:
            if family_to_show in product_family:
                existing_installations.append((product_family, version, env))
                print(f"  ✓ MATCH: {product_family} {version} in {env}")
                break
        else:
            print(f"  ✗ No match for '{product_family}' (looking for: {software_to_show_families})")
    
    print(f"\n=== RESULTS ===")
    print(f"Found {len(existing_installations)} existing installations")
    for product, version, env in existing_installations:
        print(f"  - {product} {version} in {env}")

if __name__ == "__main__":
    debug_compatibility() 