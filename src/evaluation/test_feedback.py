#!/usr/bin/env python3
"""
Test Script for Feedback Integration
Demonstrates the feedback collection system with a simple CLI interface.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.feedback_system import FeedbackIntegration, cli_main

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_results(response: dict):
    """Print query results in a formatted way."""
    print("\n" + "="*60)
    print("QUERY RESULTS")
    print("="*60)
    print(f"Query: {response['query']}")
    print(f"Session ID: {response['session_id']}")
    print(f"Timestamp: {response['timestamp']}")
    
    if 'error' in response:
        print(f"Error: {response['error']}")
        return
    
    print(f"\nFound {len(response['results'])} results:")
    for i, result in enumerate(response['results'], 1):
        print(f"\n{i}. {result['chunk_id']} (similarity: {result['similarity_score']:.3f})")
        print(f"   Type: {result['chunk_type']}")
    
    print("\n" + "="*60)
    print("FORMATTED RESULTS FOR LLM")
    print("="*60)
    print(response['formatted_results'])
    print("="*60)

def get_user_feedback() -> Optional[int]:
    """Get feedback from user."""
    while True:
        print("\nHow would you rate these results?")
        print("1. Good/Relevant")
        print("0. Bad/Irrelevant")
        print("s. Skip (don't provide feedback)")
        
        choice = input("Enter your choice (1/0/s): ").strip().lower()
        
        if choice == '1':
            return 1
        elif choice == '0':
            return 0
        elif choice == 's':
            return None
        else:
            print("Invalid choice. Please enter 1, 0, or s.")

def get_user_notes() -> str:
    """Get optional notes from user."""
    notes = input("\nOptional notes (press Enter to skip): ").strip()
    return notes if notes else ""

def main():
    """Main test function."""
    print("🧪 Feedback Integration Test")
    print("="*60)
    
    try:
        # Initialize feedback integration
        print("Initializing feedback integration...")
        feedback_integration = FeedbackIntegration()
        print(f"Session ID: {feedback_integration.session_id}")
        
        # Get user OS for context
        user_os = input("\nEnter your OS (e.g., Windows, macOS, Linux): ").strip()
        if not user_os:
            user_os = "Unknown"
        
        while True:
            print("\n" + "="*60)
            print("FEEDBACK TEST MENU")
            print("="*60)
            print("1. Run a query")
            print("2. View feedback analytics")
            print("3. Export session feedback")
            print("4. Start new session")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                # Run a query
                query = input("\nEnter your query: ").strip()
                if not query:
                    print("Query cannot be empty.")
                    continue
                
                print(f"\nExecuting query: {query}")
                response = feedback_integration.query_with_feedback(
                    query=query,
                    top_k=5,
                    user_os=user_os
                )
                
                print_results(response)
                
                # Get feedback
                feedback_score = get_user_feedback()
                if feedback_score is not None:
                    notes = get_user_notes()
                    
                    success = feedback_integration.submit_feedback(
                        query=query,
                        generated_output=response['formatted_results'],
                        feedback_score=feedback_score,
                        user_os=user_os,
                        notes=notes
                    )
                    
                    if success:
                        print("✅ Feedback submitted successfully!")
                    else:
                        print("❌ Failed to submit feedback.")
                else:
                    print("Skipped feedback submission.")
            
            elif choice == '2':
                # View analytics
                analytics = feedback_integration.get_feedback_analytics()
                print("\n" + "="*60)
                print("FEEDBACK ANALYTICS")
                print("="*60)
                
                if analytics:
                    global_stats = analytics.get('global_stats', {})
                    session_stats = analytics.get('session_stats', {})
                    
                    print("GLOBAL STATISTICS:")
                    print(f"  Total feedback: {global_stats.get('total_feedback', 0)}")
                    print(f"  Positive feedback: {global_stats.get('positive_feedback', 0)}")
                    print(f"  Negative feedback: {global_stats.get('negative_feedback', 0)}")
                    print(f"  Positive rate: {global_stats.get('positive_rate', 0):.1f}%")
                    print(f"  Recent feedback (7 days): {global_stats.get('recent_feedback', 0)}")
                    print(f"  Unique sessions: {global_stats.get('unique_sessions', 0)}")
                    
                    print("\nCURRENT SESSION STATISTICS:")
                    print(f"  Session ID: {analytics.get('session_id', 'Unknown')}")
                    print(f"  Total queries: {session_stats.get('total_queries', 0)}")
                    print(f"  Rated queries: {session_stats.get('rated_queries', 0)}")
                    print(f"  Positive feedback: {session_stats.get('positive_feedback', 0)}")
                    print(f"  Negative feedback: {session_stats.get('negative_feedback', 0)}")
                    print(f"  Session positive rate: {session_stats.get('session_positive_rate', 0):.1f}%")
                else:
                    print("No analytics data available.")
            
            elif choice == '3':
                # Export feedback
                format_choice = input("Export format (json/csv): ").strip().lower()
                if format_choice not in ['json', 'csv']:
                    print("Invalid format. Using JSON.")
                    format_choice = 'json'
                
                output_path = f"data/processed/session_feedback_{feedback_integration.session_id[:8]}.{format_choice}"
                
                success = feedback_integration.export_session_feedback(output_path, format_choice)
                if success:
                    print(f"✅ Session feedback exported to: {output_path}")
                else:
                    print("❌ Failed to export session feedback.")
            
            elif choice == '4':
                # New session
                new_session_id = feedback_integration.new_session()
                print(f"✅ Started new session: {new_session_id}")
            
            elif choice == '5':
                # Exit
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-5.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user.")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    cli_main() 