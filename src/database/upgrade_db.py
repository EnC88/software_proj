from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional
import logging
from contextlib import contextmanager
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpgradeDatabase:
    def __init__(self, db_url: str = 'sqlite:///upgrade_logs.db'):
        """Initialize database connection and create tables if they don't exist."""
        self.db_url = db_url
        self.engine = None
        self.Session = None
        self.initialize_database()

    def initialize_database(self):
        """Initialize database connection and create necessary tables."""
        try:
            self.engine = create_engine(self.db_url)
            self.Session = sessionmaker(bind=self.engine)
            
            with self.engine.connect() as conn:
                # Create main table for vectors
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS upgrades (
                        log_id TEXT PRIMARY KEY,
                        chunk_text TEXT,
                        object_name TEXT,
                        change_type TEXT,
                        version_info TEXT,
                        embedding BLOB
                    )
                """))
                
                # Create FTS table for full-text search
                conn.execute(text("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS upgrades_fts 
                    USING fts5(
                        chunk_text,
                        object_name,
                        change_type,
                        content='upgrades',
                        content_rowid='log_id'
                    )
                """))
                
                logger.info("Database initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def store_vectors(self, vectors: List[Dict[str, Any]]):
        """Store vectorized data in the database."""
        try:
            with self.get_session() as session:
                for vector in vectors:
                    # Convert numpy array to bytes for storage
                    embedding_bytes = vector['embedding'].tobytes()
                    
                    # Prepare version info
                    version_info = json.dumps(vector.get('version_info', {}))
                    
                    # Insert or update record
                    stmt = text("""
                        INSERT INTO upgrades (
                            log_id, chunk_text, object_name, change_type, 
                            version_info, embedding
                        ) VALUES (
                            :log_id, :chunk_text, :object_name, :change_type,
                            :version_info, :embedding
                        ) ON CONFLICT(log_id) DO UPDATE SET
                            chunk_text = excluded.chunk_text,
                            object_name = excluded.object_name,
                            change_type = excluded.change_type,
                            version_info = excluded.version_info,
                            embedding = excluded.embedding
                    """)
                    
                    session.execute(stmt, {
                        'log_id': vector['log_id'],
                        'chunk_text': vector['chunk_text'],
                        'object_name': vector['object_name'],
                        'change_type': vector['change_type'],
                        'version_info': version_info,
                        'embedding': embedding_bytes
                    })
                
                logger.info(f"Stored {len(vectors)} vectors successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to store vectors: {e}")
            raise

    def query_similar_vectors(self, query_embedding: np.ndarray, 
                            min_similarity: float = 0.05, 
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """Query similar vectors from the database."""
        try:
            with self.get_session() as session:
                # Get all vectors
                result = session.execute(text("""
                    SELECT log_id, chunk_text, object_name, change_type, 
                           version_info, embedding
                    FROM upgrades
                """))
                
                # Calculate similarities and sort
                similarities = []
                for row in result:
                    stored_embedding = np.frombuffer(row.embedding, dtype=np.float32)
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    
                    if similarity >= min_similarity:
                        similarities.append({
                            'log_id': row.log_id,
                            'chunk_text': row.chunk_text,
                            'object_name': row.object_name,
                            'change_type': row.change_type,
                            'version_info': json.loads(row.version_info),
                            'similarity': similarity
                        })
                
                # Sort by similarity and return top k
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                return similarities[:top_k]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to query similar vectors: {e}")
            raise

    def get_all_vectors(self) -> List[np.ndarray]:
        """Get all vectors from the database."""
        try:
            with self.get_session() as session:
                result = session.execute(text("""
                    SELECT embedding
                    FROM upgrades
                """))
                return [np.frombuffer(row.embedding, dtype=np.float32) for row in result]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get all vectors: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.get_session() as session:
                # Get total count
                count_result = session.execute(text("""
                    SELECT COUNT(*) as total FROM upgrades
                """))
                total_count = count_result.scalar()
                
                # Get counts by change type
                type_result = session.execute(text("""
                    SELECT change_type, COUNT(*) as count
                    FROM upgrades
                    GROUP BY change_type
                """))
                type_counts = {row.change_type: row.count for row in type_result}
                
                # Get counts by object name
                object_result = session.execute(text("""
                    SELECT object_name, COUNT(*) as count
                    FROM upgrades
                    GROUP BY object_name
                """))
                object_counts = {row.object_name: row.count for row in object_result}
                
                # Get cache statistics if available
                cache_stats = {}
                try:
                    cache_result = session.execute(text("""
                        SELECT COUNT(*) as total,
                               SUM(CASE WHEN last_accessed IS NOT NULL THEN 1 ELSE 0 END) as hits,
                               SUM(CASE WHEN last_accessed IS NULL THEN 1 ELSE 0 END) as misses
                        FROM query_cache
                    """))
                    row = cache_result.fetchone()
                    if row:
                        cache_stats = {
                            'hits': row.hits or 0,
                            'misses': row.misses or 0,
                            'size': row.total or 0
                        }
                except SQLAlchemyError:
                    # Cache table might not exist yet
                    cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
                
                return {
                    'total_records': total_count,
                    'change_type_distribution': type_counts,
                    'object_name_distribution': object_counts,
                    'cache': cache_stats
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get statistics: {e}")
            raise

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 