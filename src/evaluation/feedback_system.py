#!/usr/bin/env python3
"""
Consolidated Feedback System for RAG Pipeline Evaluation
SQLite-only version: All feedback logging, integration, automation, and analytics use SQLite exclusively.
"""

# --- Imports ---
import logging
import json
import numpy as np
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
from contextlib import contextmanager
import schedule
import threading
import time
import os
from src.data_processing.hybrid_embedder import CompatibilityEmbedder
from src.data_processing.build_faiss_index import build_and_save_faiss_index

# Define repo root for robust file access
REPO_ROOT = Path(__file__).resolve().parents[2]

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for SQLite."""
    def __init__(self):
        self.db_path = REPO_ROOT / 'data' / 'processed' / 'feedback_log.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite database."""
        pass

    @contextmanager
    def get_connection(self):
        """Get SQLite database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as close_error:
                    logger.error(f"Connection close failed: {close_error}")

    def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                if query.strip().upper().startswith('SELECT'):
                    return [dict(row) for row in cursor.fetchall()]
                else:
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

    def create_tables(self):
        """Create necessary tables in SQLite."""
        self._create_sqlite_tables()

    def _create_sqlite_tables(self):
        """Create SQLite tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    generated_output TEXT NOT NULL,
                    feedback_score INTEGER NOT NULL CHECK (feedback_score IN (-1, 0, 1)),
                    user_os TEXT,
                    notes TEXT,
                    metadata TEXT,
                    tags TEXT,  -- Added for storing feedback tags as JSON
                    created_at TEXT DEFAULT (datetime('now'))
                )
            ''')
            # Improvement logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS improvement_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    recommendation_title TEXT,
                    recommendation_action TEXT,
                    status TEXT,
                    metadata TEXT
                )
            ''')
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON feedback(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_score ON feedback(feedback_score)')
            conn.commit()

# --- Feedback Logger ---
class FeedbackLogger:
    """Handles logging of feedback for analysis results using SQLite."""
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.db_manager.create_tables()
        logger.info(f"Feedback database initialized with SQLite")

    @contextmanager
    def _get_connection(self):
        with self.db_manager.get_connection() as conn:
            yield conn

    def log(self, query: str, generated_output: str, feedback_score, user_os: Optional[str] = None, session_id: Optional[str] = None, notes: Optional[str] = "", metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> bool:
        """
        Log user feedback in a robust, user-friendly way.
        - Accepts feedback_score as int (1 or 0), with -1 for missing/invalid.
        - Handles missing/optional fields gracefully.
        - Never raises on user error; logs and stores what it can.
        """
        # Normalize feedback_score (only 1, 0, or -1)
        try:
            score = int(feedback_score)
            if score not in [0, 1]:
                score = -1
        except Exception:
            score = -1

        # Sanitize and default fields
        query = str(query).strip() if query else ""
        generated_output = str(generated_output).strip() if generated_output else ""
        user_os = str(user_os).strip() if user_os else None
        session_id = str(session_id).strip() if session_id else None
        notes = str(notes).strip() if notes else ""
        metadata_json = json.dumps(metadata) if metadata else None
        tags_json = json.dumps(tags) if tags else None

        # If required fields are missing, log and skip
        if not query:
            logger.warning("Feedback log: Query is empty, skipping log entry.")
            return False
        if not generated_output:
            logger.warning("Feedback log: Generated output is empty, skipping log entry.")
            return False

        try:
            timestamp = datetime.now().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO feedback (timestamp, session_id, query, generated_output, 
                                        feedback_score, user_os, notes, metadata, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, session_id, query, generated_output, 
                     score, user_os, notes, metadata_json, tags_json))
                conn.commit()
            logger.info(f"Logged feedback with score: {score} for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Database error in feedback logging: {str(e)}")
            return False

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM feedback ORDER BY timestamp DESC')
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Database error retrieving feedback: {str(e)}")
            return []

    def get_feedback_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            raise ValueError("Session ID cannot be empty")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM feedback WHERE session_id = ? ORDER BY timestamp DESC', (session_id,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Database error retrieving feedback for session {session_id}: {str(e)}")
            return []

    def get_feedback_stats(self) -> Dict[str, Any]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM feedback')
                total_count = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback_score = 1')
                positive_count = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback_score = 0')
                negative_count = cursor.fetchone()[0]
                cursor.execute('''
                    SELECT COUNT(*) FROM feedback 
                    WHERE timestamp >= datetime('now', '-7 days')
                ''')
                recent_count = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(DISTINCT session_id) FROM feedback WHERE session_id IS NOT NULL')
                unique_sessions = cursor.fetchone()[0]
                return {
                    'total_feedback': total_count,
                    'positive_feedback': positive_count,
                    'negative_feedback': negative_count,
                    'recent_feedback': recent_count,
                    'unique_sessions': unique_sessions,
                    'positive_rate': (positive_count / total_count * 100) if total_count > 0 else 0
                }
        except Exception as e:
            logger.error(f"Database error retrieving feedback stats: {str(e)}")
            return {}

# --- Feedback Loop ---
def run_feedback_loop():
    logger.info("Starting feedback loop...")
    # 1. Load feedback
    feedback_logger = FeedbackLogger()
    all_feedback = feedback_logger.get_all_feedback()
    logger.info(f"Loaded {len(all_feedback)} feedback entries.")

    # 2. Separate positive and negative feedback
    positive_feedback = [f for f in all_feedback if f['feedback_score'] == 1]
    negative_feedback = [f for f in all_feedback if f['feedback_score'] == 0]
    
    logger.info(f"Found {len(positive_feedback)} positive and {len(negative_feedback)} negative feedback entries.")

    # 3. Create training data from corrections
    training_data = []
    
    # Add positive examples (query + what the system generated correctly)
    for f in positive_feedback:
        training_data.append({
            'query': f['query'],
            'target': f['generated_output'],  # System got it right
            'type': 'positive'
        })
    
    # Add negative examples with corrections (query + what user says should have been correct)
    for f in negative_feedback:
        metadata = json.loads(f.get('metadata', '{}')) if f.get('metadata') else {}
        correction = metadata.get('correction', '')
        
        if correction:
            training_data.append({
                'query': f['query'],
                'target': correction,  # User's correction
                'type': 'correction'
            })
        else:
            # No correction provided, use as negative example
            training_data.append({
                'query': f['query'],
                'target': f['generated_output'],  # Original wrong answer
                'type': 'negative'
            })
    
    logger.info(f"Created {len(training_data)} training examples: {len([t for t in training_data if t['type'] == 'positive'])} positive, {len([t for t in training_data if t['type'] == 'correction'])} corrections, {len([t for t in training_data if t['type'] == 'negative'])} negative")

    # 4. Retrain embedder on query-target pairs
    if training_data:
        logger.info("Retraining hybrid embedder on feedback data...")
        try:
            embedder = CompatibilityEmbedder()
            
            # Prepare training data for embedder
            queries = [t['query'] for t in training_data]
            targets = [t['target'] for t in training_data]
            
            # Combine queries and targets for training
            combined_texts = queries + targets
            
            # Retrain the embedder on this data
            embedder.embedder.encode(combined_texts)  # Fit on combined data
            embedder.embedder.save(REPO_ROOT / 'data' / 'embeddings' / 'hybrid_model')
            logger.info("Hybrid embedder retrained and saved with feedback data.")
            
        except Exception as e:
            logger.error(f"Error retraining embedder: {e}")
            return {
                'total_feedback': len(all_feedback),
                'negative_feedback': len(negative_feedback),
                'training_examples': len(training_data),
                'status': 'error',
                'error': str(e)
            }
    else:
        logger.warning("No training data available for retraining.")

    # 5. Rebuild the FAISS index
    logger.info("Rebuilding FAISS index...")
    try:
        build_and_save_faiss_index()
        logger.info("FAISS index rebuilt.")
    except Exception as e:
        logger.error(f"Error rebuilding FAISS index: {e}")
        return {
            'total_feedback': len(all_feedback),
            'negative_feedback': len(negative_feedback),
            'training_examples': len(training_data),
            'status': 'partial_error',
            'error': f"Embedder trained but index rebuild failed: {str(e)}"
        }

    # 6. Log/report improvements
    logger.info(f"Feedback loop complete. Model retrained with {len(training_data)} examples from {len(all_feedback)} feedback entries.")
    return {
        'total_feedback': len(all_feedback),
        'positive_feedback': len(positive_feedback),
        'negative_feedback': len(negative_feedback),
        'training_examples': len(training_data),
        'corrections_used': len([t for t in training_data if t['type'] == 'correction']),
        'status': 'success'
    }