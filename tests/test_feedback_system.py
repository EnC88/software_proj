#!/usr/bin/env python3
"""
Tests for the feedback system.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.feedback_system import FeedbackLogger, FeedbackIntegration, FeedbackLoop

class TestFeedbackLogger:
    """Test cases for FeedbackLogger."""
    
    def test_feedback_logger_initialization(self):
        """Test FeedbackLogger initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            logger = FeedbackLogger(db_path=db_path, db_type='sqlite')
            assert logger is not None
            assert logger.db_manager.db_type == 'sqlite'
        finally:
            os.unlink(db_path)
    
    def test_feedback_logger_validation(self):
        """Test input validation in feedback logging."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            logger = FeedbackLogger(db_path=db_path, db_type='sqlite')
            
            # Test empty query
            with pytest.raises(ValueError, match="Query cannot be empty"):
                logger.log("", "test output", 1)
            
            # Test empty output
            with pytest.raises(ValueError, match="Generated output cannot be empty"):
                logger.log("test query", "", 1)
            
            # Test invalid score
            with pytest.raises(ValueError, match="Feedback score must be -1, 0, or 1"):
                logger.log("test query", "test output", 2)
        
        finally:
            os.unlink(db_path)
    
    def test_feedback_logging(self):
        """Test successful feedback logging."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            logger = FeedbackLogger(db_path=db_path, db_type='sqlite')
            
            # Test successful logging
            result = logger.log(
                query="test query",
                generated_output="test output",
                feedback_score=1,
                user_os="Linux",
                session_id="test-session",
                notes="test notes"
            )
            
            assert result is True
            
            # Test retrieving feedback
            all_feedback = logger.get_all_feedback()
            assert len(all_feedback) == 1
            assert all_feedback[0]['query'] == "test query"
            assert all_feedback[0]['feedback_score'] == 1
        
        finally:
            os.unlink(db_path)

class TestFeedbackIntegration:
    """Test cases for FeedbackIntegration."""
    
    @patch('src.evaluation.feedback_system.VectorStore')
    def test_feedback_integration_initialization(self, mock_vector_store):
        """Test FeedbackIntegration initialization."""
        integration = FeedbackIntegration()
        assert integration is not None
        assert integration.session_id is not None
    
    @patch('src.evaluation.feedback_system.VectorStore')
    def test_query_with_feedback(self, mock_vector_store):
        """Test query with feedback functionality."""
        # Mock the vector store
        mock_store = Mock()
        mock_store.query.return_value = {'results': [{'chunk_id': 'test', 'similarity_score': 0.8}]}
        mock_vector_store.return_value = mock_store
        
        integration = FeedbackIntegration()
        response = integration.query_with_feedback("test query", top_k=5)
        
        assert response['query'] == "test query"
        assert response['session_id'] == integration.session_id
        assert 'timestamp' in response

class TestFeedbackLoop:
    """Test cases for FeedbackLoop."""
    
    def test_feedback_loop_initialization(self):
        """Test FeedbackLoop initialization."""
        loop = FeedbackLoop()
        assert loop is not None
        assert loop.improvement_threshold == 0.7
    
    def test_analyze_feedback_empty(self):
        """Test feedback analysis with empty data."""
        loop = FeedbackLoop()
        analysis = loop.analyze_feedback(days_back=7)
        
        # Should return empty analysis when no feedback
        assert analysis is not None
        assert 'total_feedback' in analysis
        assert analysis['total_feedback'] == 0

def test_database_manager():
    """Test DatabaseManager functionality."""
    from src.evaluation.feedback_system import DatabaseManager
    
    # Test SQLite initialization
    db_manager = DatabaseManager(db_type='sqlite')
    assert db_manager.db_type == 'sqlite'
    
    # Test PostgreSQL fallback when not available
    with patch('src.evaluation.feedback_system.POSTGRES_AVAILABLE', False):
        db_manager = DatabaseManager(db_type='postgresql')
        assert db_manager.db_type == 'sqlite'  # Should fallback to SQLite

if __name__ == "__main__":
    pytest.main([__file__]) 