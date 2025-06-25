#!/usr/bin/env python3
"""
Flask API endpoints for feedback system integration.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import sys
from pathlib import Path

# Define repo root for robust file access
REPO_ROOT = Path(__file__).resolve().parents[2]

# Add src to path for imports
sys.path.append(str(REPO_ROOT))

from src.evaluation.feedback_system import FeedbackIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Get secret key from environment variable
secret_key = os.getenv('FLASK_SECRET_KEY')
if not secret_key:
    logger.warning("FLASK_SECRET_KEY not set, using default (not secure for production)")
    secret_key = 'default-secret-key-change-in-production'
app.secret_key = secret_key

# Enable CORS
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Global feedback integration instance
_feedback_integration = None

def get_feedback_integration():
    """Get or create feedback integration instance."""
    global _feedback_integration
    if _feedback_integration is None:
        try:
            _feedback_integration = FeedbackIntegration()
        except Exception as e:
            logger.error(f"Failed to initialize feedback integration: {e}")
            raise
    return _feedback_integration

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

@app.route('/health', methods=['GET'])
@limiter.limit("100 per minute")
def health_check():
    """Health check endpoint."""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'feedback-api',
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/query', methods=['POST'])
@limiter.limit("30 per minute")
def query():
    """Handle query requests."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        query_text = data.get('query', '').strip()
        user_os = data.get('user_os', 'Unknown')
        
        if not query_text:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Validate query length
        if len(query_text) > 1000:
            return jsonify({'error': 'Query too long (max 1000 characters)'}), 400
        
        # Get feedback integration
        fi = get_feedback_integration()
        
        # Execute query
        response = fi.query_with_feedback(
            query=query_text,
            top_k=5,
            user_os=user_os
        )
        
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/feedback', methods=['POST'])
@limiter.limit("50 per minute")
def submit_feedback():
    """Submit feedback for a query."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        query = data.get('query', '').strip()
        generated_output = data.get('generated_output', '').strip()
        feedback_score = data.get('feedback_score')
        user_os = data.get('user_os')
        notes = data.get('notes', '')
        metadata = data.get('metadata')
        
        # Validate required fields
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        if not generated_output:
            return jsonify({'error': 'Generated output cannot be empty'}), 400
        if feedback_score not in [-1, 0, 1]:
            return jsonify({'error': 'Feedback score must be -1, 0, or 1'}), 400
        
        # Validate input lengths
        if len(query) > 1000:
            return jsonify({'error': 'Query too long (max 1000 characters)'}), 400
        if len(generated_output) > 10000:
            return jsonify({'error': 'Generated output too long (max 10000 characters)'}), 400
        if len(notes) > 2000:
            return jsonify({'error': 'Notes too long (max 2000 characters)'}), 400
        
        # Get feedback integration
        fi = get_feedback_integration()
        
        # Submit feedback
        success = fi.submit_feedback(
            query=query,
            generated_output=generated_output,
            feedback_score=feedback_score,
            user_os=user_os,
            notes=notes,
            metadata=metadata
        )
        
        if success:
            return jsonify({'status': 'success', 'message': 'Feedback submitted successfully'})
        else:
            return jsonify({'error': 'Failed to submit feedback'}), 500
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analytics')
@limiter.limit("20 per minute")
def get_analytics():
    """Get feedback analytics."""
    try:
        fi = get_feedback_integration()
        analytics = fi.get_feedback_analytics()
        return jsonify(analytics)
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/session/new', methods=['POST'])
@limiter.limit("10 per minute")
def new_session():
    """Create a new feedback session."""
    try:
        fi = get_feedback_integration()
        session_id = fi.new_session()
        return jsonify({'session_id': session_id})
        
    except Exception as e:
        logger.error(f"Session creation error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(429)
def ratelimit_handler(error):
    """Handle rate limit errors."""
    return jsonify({'error': 'Rate limit exceeded'}), 429

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 