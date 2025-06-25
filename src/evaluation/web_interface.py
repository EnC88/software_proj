#!/usr/bin/env python3
"""
Web Interface for Feedback System
Simple Flask web app for testing the RAG pipeline with feedback collection.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, session
import uuid
import json

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.feedback_system import FeedbackIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global feedback integration instance
feedback_integration = None

def get_feedback_integration():
    """Get or create feedback integration instance."""
    global feedback_integration
    if feedback_integration is None:
        feedback_integration = FeedbackIntegration()
    return feedback_integration

@app.route('/')
def index():
    """Main page with query interface."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html', session_id=session['session_id'])

@app.route('/query', methods=['POST'])
def query():
    """Handle query requests."""
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        user_os = data.get('user_os', 'Unknown')
        
        if not query_text:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Get feedback integration
        fi = get_feedback_integration()
        
        # Execute query
        response = fi.query_with_feedback(
            query=query_text,
            top_k=5,
            user_os=user_os
        )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback submission."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        generated_output = data.get('generated_output', '')
        feedback_score = data.get('feedback_score')
        user_os = data.get('user_os', 'Unknown')
        notes = data.get('notes', '')
        
        if feedback_score not in [0, 1]:
            return jsonify({'error': 'Invalid feedback score'}), 400
        
        # Get feedback integration
        fi = get_feedback_integration()
        
        # Submit feedback
        success = fi.submit_feedback(
            query=query,
            generated_output=generated_output,
            feedback_score=feedback_score,
            user_os=user_os,
            notes=notes
        )
        
        if success:
            return jsonify({'message': 'Feedback submitted successfully'})
        else:
            return jsonify({'error': 'Failed to submit feedback'}), 500
            
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
def get_analytics():
    """Get feedback analytics."""
    try:
        fi = get_feedback_integration()
        analytics = fi.get_feedback_analytics()
        return jsonify(analytics)
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export_feedback():
    """Export feedback data."""
    try:
        data = request.get_json()
        format_type = data.get('format', 'json')
        
        if format_type not in ['json', 'csv']:
            return jsonify({'error': 'Invalid format'}), 400
        
        fi = get_feedback_integration()
        output_path = f"data/processed/web_export_{fi.session_id[:8]}.{format_type}"
        
        success = fi.export_session_feedback(output_path, format_type)
        
        if success:
            return jsonify({
                'message': 'Export successful',
                'file_path': output_path
            })
        else:
            return jsonify({'error': 'Export failed'}), 500
            
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/new-session', methods=['POST'])
def new_session():
    """Start a new feedback session."""
    try:
        global feedback_integration
        feedback_integration = FeedbackIntegration()
        session['session_id'] = feedback_integration.session_id
        
        return jsonify({
            'message': 'New session started',
            'session_id': feedback_integration.session_id
        })
        
    except Exception as e:
        logger.error(f"New session error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Create basic template
    template_path = templates_dir / 'index.html'
    if not template_path.exists():
        create_basic_template(template_path)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

def create_basic_template(template_path: Path):
    """Create a basic HTML template for the web interface."""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Feedback System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .query-section {
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"], textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .results {
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            white-space: pre-wrap;
        }
        .feedback-section {
            margin-top: 20px;
            padding: 20px;
            background-color: #e9ecef;
            border-radius: 5px;
        }
        .feedback-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .btn-good {
            background-color: #28a745;
        }
        .btn-bad {
            background-color: #dc3545;
        }
        .analytics {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .loading {
            text-align: center;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 RAG Feedback System</h1>
            <p>Test the software compatibility RAG pipeline and provide feedback</p>
            <p><strong>Session ID:</strong> <span id="sessionId">{{ session_id }}</span></p>
        </div>

        <div class="query-section">
            <h2>Query Interface</h2>
            <div class="form-group">
                <label for="userOs">Your Operating System:</label>
                <input type="text" id="userOs" placeholder="e.g., Windows, macOS, Linux" value="Unknown">
            </div>
            <div class="form-group">
                <label for="queryInput">Enter your query:</label>
                <textarea id="queryInput" placeholder="e.g., What servers are compatible with Windows Server 2019?"></textarea>
            </div>
            <button onclick="executeQuery()">🔍 Execute Query</button>
        </div>

        <div id="results" style="display: none;">
            <h2>Query Results</h2>
            <div id="resultsContent" class="results"></div>
            
            <div class="feedback-section">
                <h3>Provide Feedback</h3>
                <p>How would you rate these results?</p>
                <div class="feedback-buttons">
                    <button class="btn-good" onclick="submitFeedback(1)">👍 Good/Relevant</button>
                    <button class="btn-bad" onclick="submitFeedback(0)">👎 Bad/Irrelevant</button>
                </div>
                <div class="form-group" style="margin-top: 15px;">
                    <label for="feedbackNotes">Optional Notes:</label>
                    <textarea id="feedbackNotes" placeholder="Any additional comments..."></textarea>
                </div>
            </div>
        </div>

        <div class="analytics">
            <h2>Analytics</h2>
            <button onclick="loadAnalytics()">📊 Load Analytics</button>
            <div id="analyticsContent"></div>
        </div>

        <div style="margin-top: 30px; text-align: center;">
            <button onclick="newSession()">🔄 New Session</button>
            <button onclick="exportData('json')">📤 Export JSON</button>
            <button onclick="exportData('csv')">📤 Export CSV</button>
        </div>
    </div>

    <script>
        let currentQuery = '';
        let currentOutput = '';

        async function executeQuery() {
            const query = document.getElementById('queryInput').value.trim();
            const userOs = document.getElementById('userOs').value.trim() || 'Unknown';
            
            if (!query) {
                alert('Please enter a query');
                return;
            }

            document.getElementById('results').style.display = 'none';
            document.getElementById('resultsContent').innerHTML = '<div class="loading">Executing query...</div>';

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query, user_os: userOs })
                });

                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('resultsContent').innerHTML = `Error: ${data.error}`;
                } else {
                    currentQuery = data.query;
                    currentOutput = data.formatted_results;
                    
                    let resultsHtml = `<strong>Query:</strong> ${data.query}<br>`;
                    resultsHtml += `<strong>Found ${data.results.length} results:</strong><br><br>`;
                    
                    data.results.forEach((result, index) => {
                        resultsHtml += `${index + 1}. ${result.chunk_id} (similarity: ${result.similarity_score.toFixed(3)})<br>`;
                        resultsHtml += `   Type: ${result.chunk_type}<br><br>`;
                    });
                    
                    resultsHtml += `<strong>Formatted Results:</strong><br>${data.formatted_results}`;
                    
                    document.getElementById('resultsContent').innerHTML = resultsHtml;
                    document.getElementById('results').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('resultsContent').innerHTML = `Error: ${error.message}`;
            }
        }

        async function submitFeedback(score) {
            if (!currentQuery || !currentOutput) {
                alert('No query results to provide feedback on');
                return;
            }

            const userOs = document.getElementById('userOs').value.trim() || 'Unknown';
            const notes = document.getElementById('feedbackNotes').value.trim();

            try {
                const response = await fetch('/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: currentQuery,
                        generated_output: currentOutput,
                        feedback_score: score,
                        user_os: userOs,
                        notes: notes
                    })
                });

                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                } else {
                    alert('Feedback submitted successfully!');
                    document.getElementById('feedbackNotes').value = '';
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        async function loadAnalytics() {
            try {
                const response = await fetch('/analytics');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('analyticsContent').innerHTML = `Error: ${data.error}`;
                } else {
                    const globalStats = data.global_stats || {};
                    const sessionStats = data.session_stats || {};
                    
                    let html = '<h3>Global Statistics:</h3>';
                    html += `<p>Total feedback: ${globalStats.total_feedback || 0}</p>`;
                    html += `<p>Positive feedback: ${globalStats.positive_feedback || 0}</p>`;
                    html += `<p>Negative feedback: ${globalStats.negative_feedback || 0}</p>`;
                    html += `<p>Positive rate: ${(globalStats.positive_rate || 0).toFixed(1)}%</p>`;
                    
                    html += '<h3>Session Statistics:</h3>';
                    html += `<p>Session ID: ${data.session_id || 'Unknown'}</p>`;
                    html += `<p>Total queries: ${sessionStats.total_queries || 0}</p>`;
                    html += `<p>Rated queries: ${sessionStats.rated_queries || 0}</p>`;
                    html += `<p>Positive feedback: ${sessionStats.positive_feedback || 0}</p>`;
                    html += `<p>Negative feedback: ${sessionStats.negative_feedback || 0}</p>`;
                    html += `<p>Session positive rate: ${(sessionStats.session_positive_rate || 0).toFixed(1)}%</p>`;
                    
                    document.getElementById('analyticsContent').innerHTML = html;
                }
            } catch (error) {
                document.getElementById('analyticsContent').innerHTML = `Error: ${error.message}`;
            }
        }

        async function newSession() {
            try {
                const response = await fetch('/new-session', { method: 'POST' });
                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                } else {
                    document.getElementById('sessionId').textContent = data.session_id;
                    document.getElementById('results').style.display = 'none';
                    currentQuery = '';
                    currentOutput = '';
                    alert('New session started!');
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        async function exportData(format) {
            try {
                const response = await fetch('/export', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ format })
                });

                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                } else {
                    alert(`Export successful! File saved to: ${data.file_path}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
    </script>
</body>
</html>'''
    
    with open(template_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Created basic template at: {template_path}") 