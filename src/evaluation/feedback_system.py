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

    def log(self, query: str, generated_output: str, feedback_score: int, user_os: Optional[str] = None, session_id: Optional[str] = None, notes: Optional[str] = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not generated_output or not generated_output.strip():
            raise ValueError("Generated output cannot be empty")
        if feedback_score not in [-1, 0, 1]:
            raise ValueError("Feedback score must be -1, 0, or 1")
        try:
            timestamp = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO feedback (timestamp, session_id, query, generated_output, 
                                        feedback_score, user_os, notes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, session_id, query.strip(), generated_output.strip(), 
                     feedback_score, user_os, notes, metadata_json))
                conn.commit()
            logger.info(f"Logged feedback with score: {feedback_score} for session: {session_id}")
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"Validation error in feedback logging: {str(e)}")
            raise
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

# --- Feedback Integration ---
class FeedbackIntegration:
    """Integrates feedback collection with the RAG query engine."""
    def __init__(self, feedback_logger: Optional[FeedbackLogger] = None, query_engine=None):
        self.feedback_logger = feedback_logger or FeedbackLogger()
        self.query_engine = query_engine
        self.session_id = str(uuid.uuid4())
    def _get_query_engine(self):
        if self.query_engine is None:
            try:
                from ..rag.query_engine import QueryEngine
                self.query_engine = QueryEngine()
            except ImportError as e:
                logger.error(f"Could not import QueryEngine: {e}")
                raise
        return self.query_engine
    def query_with_feedback(self, query: str, top_k: int = 5, user_os: Optional[str] = None, auto_log: bool = True) -> Dict[str, Any]:
        try:
            engine = self._get_query_engine()
            results = engine.query(query, top_k)
            formatted_results = engine.format_results_for_llm(results)
            response = {
                'query': query,
                'results': results,
                'formatted_results': formatted_results,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'user_os': user_os,
                'feedback_id': None
            }
            if auto_log:
                self.log_query(query, formatted_results, user_os)
            return response
        except Exception as e:
            logger.error(f"Error in query_with_feedback: {str(e)}")
            return {
                'query': query,
                'error': str(e),
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat()
            }
    def log_query(self, query: str, formatted_results: str, user_os: Optional[str] = None) -> bool:
        try:
            metadata = {
                'query_type': 'unrated',
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat()
            }
            return self.feedback_logger.log(
                query=query,
                generated_output=formatted_results,
                feedback_score=-1,
                user_os=user_os,
                session_id=self.session_id,
                notes="Query logged for tracking",
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")
            return False
    def submit_feedback(self, query: str, generated_output: str, feedback_score: int, user_os: Optional[str] = None, notes: Optional[str] = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            if metadata is None:
                metadata = {}
            metadata.update({
                'session_id': self.session_id,
                'feedback_timestamp': datetime.now().isoformat()
            })
            return self.feedback_logger.log(
                query=query,
                generated_output=generated_output,
                feedback_score=feedback_score,
                user_os=user_os,
                session_id=self.session_id,
                notes=notes,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            return False
    def get_feedback_analytics(self) -> Dict[str, Any]:
        try:
            stats = self.feedback_logger.get_feedback_stats()
            session_feedback = self.feedback_logger.get_feedback_by_session(self.session_id)
            session_stats = {
                'total_queries': len([f for f in session_feedback if f['feedback_score'] == -1]),
                'rated_queries': len([f for f in session_feedback if f['feedback_score'] in [0, 1]]),
                'positive_feedback': len([f for f in session_feedback if f['feedback_score'] == 1]),
                'negative_feedback': len([f for f in session_feedback if f['feedback_score'] == 0])
            }
            if session_stats['rated_queries'] > 0:
                session_stats['session_positive_rate'] = (
                    session_stats['positive_feedback'] / session_stats['rated_queries'] * 100
                )
            else:
                session_stats['session_positive_rate'] = 0
            return {
                'global_stats': stats,
                'session_stats': session_stats,
                'session_id': self.session_id
            }
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {str(e)}")
            return {}
    def new_session(self) -> str:
        self.session_id = str(uuid.uuid4())
        logger.info(f"Started new feedback session: {self.session_id}")
        return self.session_id

# --- Feedback Loop ---
class FeedbackLoop:
    """Automated feedback loop that improves the RAG pipeline based on collected feedback."""
    def __init__(self, feedback_logger: Optional[FeedbackLogger] = None, improvement_threshold: float = 0.7):
        self.feedback_logger = feedback_logger or FeedbackLogger()
        self.improvement_threshold = improvement_threshold
        self.analysis_results = {}
    def analyze_feedback(self, days_back: int = 7) -> Dict[str, Any]:
        try:
            recent_feedback = self._get_recent_feedback(days_back)
            if not recent_feedback:
                logger.info("No recent feedback to analyze")
                return {}
            analysis = {
                'total_feedback': len(recent_feedback),
                'positive_feedback': len([f for f in recent_feedback if f['feedback_score'] == 1]),
                'negative_feedback': len([f for f in recent_feedback if f['feedback_score'] == 0]),
                'positive_rate': 0,
                'query_patterns': self._analyze_query_patterns(recent_feedback),
                'os_patterns': self._analyze_os_patterns(recent_feedback),
                'common_issues': self._identify_common_issues(recent_feedback),
                'performance_metrics': self._analyze_performance(recent_feedback),
                'recommendations': []
            }
            rated_feedback = [f for f in recent_feedback if f['feedback_score'] in [0, 1]]
            if rated_feedback:
                analysis['positive_rate'] = analysis['positive_feedback'] / len(rated_feedback)
            analysis['recommendations'] = self._generate_recommendations(analysis)
            self.analysis_results = analysis
            logger.info(f"Feedback analysis completed: {analysis['positive_rate']:.1%} positive rate")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing feedback: {str(e)}")
            return {}
    def _get_recent_feedback(self, days_back: int) -> List[Dict[str, Any]]:
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            with sqlite3.connect(self.feedback_logger.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM feedback 
                    WHERE timestamp >= ? AND feedback_score IN (0, 1)
                    ORDER BY timestamp DESC
                ''', (cutoff_date.isoformat(),))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting recent feedback: {str(e)}")
            return []
    def _analyze_query_patterns(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        patterns = {
            'query_lengths': [],
            'common_keywords': defaultdict(int),
            'query_types': defaultdict(int),
            'negative_query_keywords': defaultdict(int)
        }
        for entry in feedback:
            query = entry.get('query', '').lower()
            score = entry.get('feedback_score', 0)
            patterns['query_lengths'].append(len(query.split()))
            words = query.split()
            for word in words:
                if len(word) > 3:
                    patterns['common_keywords'][word] += 1
                    if score == 0:
                        patterns['negative_query_keywords'][word] += 1
            query_type = self._classify_query_type(query)
            patterns['query_types'][query_type] += 1
        patterns['top_keywords'] = dict(sorted(
            patterns['common_keywords'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        patterns['problematic_keywords'] = dict(sorted(
            patterns['negative_query_keywords'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5])
        return patterns
    def _classify_query_type(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ['compatible', 'compatibility', 'support']):
            return 'compatibility'
        elif any(word in query_lower for word in ['server', 'hardware']):
            return 'hardware'
        elif any(word in query_lower for word in ['environment', 'production', 'development']):
            return 'environment'
        elif any(word in query_lower for word in ['windows', 'linux', 'os']):
            return 'operating_system'
        elif any(word in query_lower for word in ['how', 'what', 'why']):
            return 'information'
        else:
            return 'general'
    def _analyze_os_patterns(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        os_patterns = defaultdict(lambda: {'total': 0, 'positive': 0, 'negative': 0})
        for entry in feedback:
            user_os = entry.get('user_os', 'Unknown')
            score = entry.get('feedback_score', 0)
            os_patterns[user_os]['total'] += 1
            if score == 1:
                os_patterns[user_os]['positive'] += 1
            elif score == 0:
                os_patterns[user_os]['negative'] += 1
        for os_name, data in os_patterns.items():
            if data['total'] > 0:
                data['success_rate'] = data['positive'] / data['total']
            else:
                data['success_rate'] = 0
        return dict(os_patterns)
    def _identify_common_issues(self, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        issues = []
        negative_feedback = [f for f in feedback if f['feedback_score'] == 0]
        note_patterns = defaultdict(int)
        for entry in negative_feedback:
            notes = entry.get('notes', '').lower()
            if notes:
                words = notes.split()
                for word in words:
                    if len(word) > 4:
                        note_patterns[word] += 1
        top_issues = sorted(note_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        for issue, count in top_issues:
            issues.append({
                'issue': issue,
                'frequency': count,
                'percentage': count / len(negative_feedback) * 100 if negative_feedback else 0
            })
        return issues
    def _analyze_performance(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        performance = {
            'avg_query_length': 0,
            'response_quality_trend': [],
            'session_analysis': defaultdict(lambda: {'queries': 0, 'positive': 0, 'negative': 0})
        }
        if not feedback:
            return performance
        query_lengths = [len(f.get('query', '').split()) for f in feedback]
        performance['avg_query_length'] = np.mean(query_lengths)
        for entry in feedback:
            session_id = entry.get('session_id', 'unknown')
            score = entry.get('feedback_score', 0)
            performance['session_analysis'][session_id]['queries'] += 1
            if score == 1:
                performance['session_analysis'][session_id]['positive'] += 1
            elif score == 0:
                performance['session_analysis'][session_id]['negative'] += 1
        return performance
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recommendations = []
        positive_rate = analysis.get('positive_rate', 0)
        if positive_rate < self.improvement_threshold:
            recommendations.append({
                'type': 'critical',
                'title': 'Low Positive Feedback Rate',
                'description': f'Only {positive_rate:.1%} of feedback is positive. Immediate attention required.',
                'action': 'Review recent negative feedback and adjust query processing.',
                'priority': 'high'
            })
        patterns = analysis.get('query_patterns', {})
        problematic_keywords = patterns.get('problematic_keywords', {})
        if problematic_keywords:
            recommendations.append({
                'type': 'improvement',
                'title': 'Problematic Query Keywords',
                'description': f'Keywords with high negative feedback: {list(problematic_keywords.keys())}',
                'action': 'Improve handling of queries containing these keywords.',
                'priority': 'medium'
            })
        os_patterns = analysis.get('os_patterns', {})
        low_performing_os = [
            os_name for os_name, data in os_patterns.items()
            if data.get('success_rate', 0) < 0.5 and data.get('total', 0) >= 3
        ]
        if low_performing_os:
            recommendations.append({
                'type': 'improvement',
                'title': 'OS-Specific Performance Issues',
                'description': f'Low performance for: {low_performing_os}',
                'action': 'Review and improve OS-specific query handling.',
                'priority': 'medium'
            })
        query_types = patterns.get('query_types', {})
        if query_types:
            most_common_type = max(query_types.items(), key=lambda x: x[1])
            recommendations.append({
                'type': 'optimization',
                'title': 'Query Type Optimization',
                'description': f'Most common query type: {most_common_type[0]} ({most_common_type[1]} queries)',
                'action': f'Optimize handling of {most_common_type[0]} queries.',
                'priority': 'low'
            })
        return recommendations
    def apply_improvements(self) -> Dict[str, Any]:
        if not self.analysis_results:
            logger.warning("No analysis results available. Run analyze_feedback() first.")
            return {}
        improvements = {
            'applied': [],
            'pending': [],
            'failed': []
        }
        recommendations = self.analysis_results.get('recommendations', [])
        for rec in recommendations:
            try:
                if rec['type'] == 'critical':
                    success = self._apply_critical_improvement(rec)
                    if success:
                        improvements['applied'].append(rec)
                    else:
                        improvements['failed'].append(rec)
                else:
                    improvements['pending'].append(rec)
            except Exception as e:
                logger.error(f"Error applying improvement: {str(e)}")
                improvements['failed'].append(rec)
        logger.info(f"Applied {len(improvements['applied'])} improvements")
        return improvements
    def _apply_critical_improvement(self, recommendation: Dict[str, Any]) -> bool:
        try:
            logger.info(f"Applying critical improvement: {recommendation['title']}")
            improvement_log = {
                'timestamp': datetime.now().isoformat(),
                'recommendation': recommendation,
                'status': 'applied'
            }
            self._save_improvement_log(improvement_log)
            return True
        except Exception as e:
            logger.error(f"Failed to apply critical improvement: {str(e)}")
            return False
    def _save_improvement_log(self, improvement_log: Dict[str, Any]):
        try:
            with sqlite3.connect(self.feedback_logger.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO improvement_logs (timestamp, recommendation_title, recommendation_action, status, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    improvement_log['timestamp'],
                    improvement_log['recommendation']['title'],
                    improvement_log['recommendation']['action'],
                    improvement_log['status'],
                    json.dumps(improvement_log)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving improvement log: {str(e)}")
    def run_automated_loop(self, interval_hours: int = 24) -> Dict[str, Any]:
        logger.info("🔄 Starting automated feedback loop")
        try:
            analysis = self.analyze_feedback(days_back=interval_hours // 24)
            if not analysis:
                logger.info("No feedback to analyze in this cycle")
                return {'status': 'no_feedback', 'analysis': {}}
            positive_rate = analysis.get('positive_rate', 0)
            if positive_rate >= self.improvement_threshold:
                logger.info(f"Performance is good ({positive_rate:.1%} positive rate). No improvements needed.")
                return {
                    'status': 'good_performance',
                    'analysis': analysis,
                    'improvements': {'applied': [], 'pending': [], 'failed': []}
                }
            improvements = self.apply_improvements()
            report = {
                'timestamp': datetime.now().isoformat(),
                'status': 'improvements_applied',
                'analysis': analysis,
                'improvements': improvements,
                'next_run': (datetime.now() + timedelta(hours=interval_hours)).isoformat()
            }
            self._save_loop_report(report)
            logger.info(f"✅ Feedback loop completed. Applied {len(improvements['applied'])} improvements.")
            return report
        except Exception as e:
            logger.error(f"❌ Feedback loop failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    def _save_loop_report(self, report: Dict[str, Any]):
        try:
            report_path = REPO_ROOT / 'data' / 'processed' / 'feedback_loop_reports'
            report_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = report_path / f'loop_report_{timestamp}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Loop report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Error saving loop report: {str(e)}")
    def get_improvement_history(self) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.feedback_logger.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM improvement_logs 
                    ORDER BY timestamp DESC 
                    LIMIT 50
                ''')
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting improvement history: {str(e)}")
            return []

# --- Section 4: Automation and Testing ---

class AutomatedScheduler:
    """Automated scheduler that runs the feedback loop at regular intervals."""
    
    def __init__(self, feedback_loop: Optional[FeedbackLoop] = None, interval_hours: int = 24, run_on_startup: bool = True):
        self.feedback_loop = feedback_loop or FeedbackLoop()
        self.interval_hours = interval_hours
        self.run_on_startup = run_on_startup
        self.is_running = False
        self.scheduler_thread = None
        self.callback = None
        
    def start(self, callback: Optional[Callable] = None):
        """Start the automated scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
            
        self.callback = callback
        self.is_running = True
        schedule.every(self.interval_hours).hours.do(self._run_feedback_loop)
        
        if self.run_on_startup:
            logger.info("Running feedback loop on startup...")
            self._run_feedback_loop()
            
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info(f"✅ Automated scheduler started. Running every {self.interval_hours} hours.")
        logger.info(f"Next run scheduled for: {datetime.now() + timedelta(hours=self.interval_hours)}")
        
    def stop(self):
        """Stop the automated scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
            
        self.is_running = False
        schedule.clear()
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        logger.info("🛑 Automated scheduler stopped")
        
    def _run_scheduler(self):
        """Internal scheduler loop."""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(300)
                
    def _run_feedback_loop(self):
        """Run the feedback loop and handle callbacks."""
        try:
            logger.info("🔄 Running scheduled feedback loop...")
            result = self.feedback_loop.run_automated_loop(self.interval_hours)
            status = result.get('status', 'unknown')
            logger.info(f"Feedback loop completed with status: {status}")
            
            if self.callback:
                try:
                    self.callback(result)
                except Exception as e:
                    logger.error(f"Error in callback: {str(e)}")
                    
            next_run = datetime.now() + timedelta(hours=self.interval_hours)
            logger.info(f"Next feedback loop scheduled for: {next_run}")
        except Exception as e:
            logger.error(f"❌ Error running feedback loop: {str(e)}")
            
    def get_status(self) -> dict:
        """Get current scheduler status."""
        return {
            'is_running': self.is_running,
            'interval_hours': self.interval_hours,
            'next_run': schedule.next_run() if schedule.jobs else None,
            'last_run': getattr(self, '_last_run', None)
        }
        
    def run_once(self) -> dict:
        """Run the feedback loop once manually."""
        logger.info("🔄 Running feedback loop manually...")
        result = self.feedback_loop.run_automated_loop(self.interval_hours)
        self._last_run = datetime.now()
        return result

def run_scheduler_daemon(interval_hours: int = 24, log_file: Optional[str] = None, pid_file: Optional[str] = None):
    """Run the scheduler as a daemon process."""
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
    if pid_file:
        with open(pid_file, 'w') as f:
            f.write(str(Path(pid_file).parent))
            
    try:
        scheduler = AutomatedScheduler(interval_hours=interval_hours)
        
        def log_results(result):
            status = result.get('status', 'unknown')
            if status == 'improvements_applied':
                improvements = result.get('improvements', {})
                applied = len(improvements.get('applied', []))
                logger.info(f"Applied {applied} improvements in this cycle")
                
        scheduler.start(callback=log_results)
        
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            scheduler.stop()
    except Exception as e:
        logger.error(f"Daemon error: {str(e)}")
        raise
    finally:
        if pid_file and Path(pid_file).exists():
            Path(pid_file).unlink()

def simulate_user_feedback(feedback_integration: FeedbackIntegration, num_queries: int = 5):
    """Simulate user queries and feedback for testing."""
    logger.info(f"🧪 Simulating {num_queries} user queries and feedback...")
    
    test_queries = [
        ("What servers are compatible with Windows Server 2019?", 1),
        ("How many Dell servers are in production?", 1),
        ("What is the weather like?", 0),
        ("Which servers support Linux?", 1),
        ("What's for lunch?", 0),
        ("Show me HP servers in development environment", 1),
        ("What time is it?", 0),
        ("Which servers are compatible with Ubuntu?", 1),
        ("Tell me a joke", 0),
        ("What servers support VMware?", 1),
    ]
    
    for i, (query, expected_score) in enumerate(test_queries[:num_queries]):
        logger.info(f"Query {i+1}: {query}")
        
        response = feedback_integration.query_with_feedback(query=query, top_k=3, user_os="macOS")
        
        if 'error' in response:
            logger.error(f"Query failed: {response['error']}")
            continue
            
        success = feedback_integration.submit_feedback(
            query=query,
            generated_output=response['formatted_results'],
            feedback_score=expected_score,
            user_os="macOS",
            notes=f"Simulated feedback - expected score: {expected_score}"
        )
        
        if success:
            logger.info(f"✅ Feedback submitted: score={expected_score}")
        else:
            logger.error("❌ Failed to submit feedback")
            
        time.sleep(1)

def run_demo():
    """Run a demonstration of the automated feedback loop."""
    print("🎯 Automated Feedback Loop Demo")
    print("=" * 60)
    
    try:
        logger.info("Initializing feedback integration...")
        feedback_integration = FeedbackIntegration()
        
        logger.info("Initializing feedback loop...")
        feedback_loop = FeedbackLoop()
        
        existing_feedback = feedback_integration.feedback_logger.get_feedback_stats()
        
        if existing_feedback.get('total_feedback', 0) < 5:
            logger.info("Generating test feedback data...")
            simulate_user_feedback(feedback_integration, num_queries=8)
        else:
            logger.info(f"Using existing feedback data: {existing_feedback.get('total_feedback', 0)} entries")
            
        logger.info("🔍 Analyzing feedback patterns...")
        analysis = feedback_loop.analyze_feedback(days_back=7)
        
        if analysis:
            print("\n📊 Feedback Analysis Results:")
            print("-" * 40)
            print(f"Total feedback: {analysis.get('total_feedback', 0)}")
            print(f"Positive feedback: {analysis.get('positive_feedback', 0)}")
            print(f"Negative feedback: {analysis.get('negative_feedback', 0)}")
            print(f"Positive rate: {analysis.get('positive_rate', 0):.1%}")
            
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Generated {len(recommendations)} recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec['title']} ({rec['priority']} priority)")
                    print(f"     {rec['description']}")
            else:
                print("\n✅ No recommendations needed - performance is good!")
                
        logger.info("🔄 Running automated feedback loop...")
        loop_result = feedback_loop.run_automated_loop(interval_hours=24)
        
        print(f"\n🔄 Feedback Loop Results:")
        print("-" * 40)
        print(f"Status: {loop_result.get('status', 'unknown')}")
        
        improvements = loop_result.get('improvements', {})
        if improvements:
            print(f"Applied improvements: {len(improvements.get('applied', []))}")
            print(f"Pending improvements: {len(improvements.get('pending', []))}")
            print(f"Failed improvements: {len(improvements.get('failed', []))}")
            
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")

def run_continuous_demo(duration_minutes: int = 5):
    """Run a continuous demo with real-time updates."""
    print(f"🔄 Continuous Demo (Duration: {duration_minutes} minutes)")
    print("=" * 60)
    
    try:
        scheduler = AutomatedScheduler(interval_hours=1, run_on_startup=False)
        
        def show_results(result):
            status = result.get('status', 'unknown')
            print(f"\n🔄 Loop completed: {status}")
            if status == 'improvements_applied':
                improvements = result.get('improvements', {})
                applied = len(improvements.get('applied', []))
                print(f"   Applied {applied} improvements")
                
        scheduler.start(callback=show_results)
        
        print("Demo started. Press Ctrl+C to stop...")
        time.sleep(duration_minutes * 60)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    finally:
        scheduler.stop()
        print("✅ Demo completed")

# --- CLI Test Interface Functions ---

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

def cli_main():
    """Main CLI test function."""
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