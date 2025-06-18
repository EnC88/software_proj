import sentence_transformers
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import re
import time
from functools import lru_cache
import threading
from sqlalchemy import text
import os
from pathlib import Path

from .models.query_parser import QueryParser
from .database.upgrade_db import Database
from .cache.query_cache import QueryCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vectorizer:
    def __init__(self, use_database: bool = True, use_cache: bool = True, 
                 model_name: str = 'all-MiniLM-L6-v2', model_path: str = None):
        """Initialize the vectorizer with caching and performance optimizations.
        
        Args:
            use_database: Whether to use database storage
            use_cache: Whether to use query caching
            model_name: Model name for sentence transformers
            model_path: Local path to model (if None, will use model_name and cache locally)
        """
        self.use_database = use_database
        self.use_cache = use_cache
        self.model_name = model_name
        self.model_path = model_path
        self.db = Database() if use_database else None
        self.model = None
        self.model_lock = threading.Lock()
        self.chunked_df = None
        self.query_parser = None
        self.query_cache = QueryCache() if use_cache else None
        self._initialize_model()
        
    def _get_model_path(self) -> str:
        """Get the local model path, downloading if necessary."""
        if self.model_path:
            # Use provided local path
            return self.model_path
        
        # Create local cache directory
        cache_dir = Path.home() / '.cache' / 'sentence_transformers'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model is already cached locally
        model_dir = cache_dir / self.model_name
        if model_dir.exists():
            logger.info(f"Using cached model at: {model_dir}")
            return str(model_dir)
        
        # Download and cache the model
        logger.info(f"Downloading model {self.model_name} to {model_dir}...")
        try:
            # This will download the model to the cache directory
            model = sentence_transformers.SentenceTransformer(self.model_name, cache_folder=str(cache_dir))
            logger.info(f"Model downloaded and cached successfully at: {model_dir}")
            return str(model_dir)
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise
        
    def _initialize_model(self):
        """Initialize the model with caching and thread safety."""
        if self.model is None:
            with self.model_lock:
                if self.model is None:  # Double-check pattern
                    try:
                        # Get local model path
                        local_model_path = self._get_model_path()
                        logger.info(f"Loading Sentence Transformer model from: {local_model_path}")
                        
                        # Load from local path
                        self.model = sentence_transformers.SentenceTransformer(local_model_path)
                        logger.info("Sentence Transformer model loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading Sentence Transformer model: {str(e)}")
                        raise
                    
                    # Initialize query parser
                    self.query_parser = QueryParser()
    
    @lru_cache(maxsize=50000)  # Increased for large datasets (couple million data points)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return zero vector with correct dimension
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            return {
                'type': 'sentence_transformers',
                'model_name': self.model_name,
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'multilingual': 'all-MiniLM-L6-v2' in self.model_name,
                'model_size': '~90MB'
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def benchmark_model(self, test_texts: List[str] = None) -> Dict[str, Any]:
        """Benchmark the current model for performance."""
        if test_texts is None:
            test_texts = [
                "How to upgrade Apache from version 2.4.1 to 2.4.2?",
                "What are common issues with MySQL 8.0 upgrade?",
                "How to handle WebSphere rollback procedures?",
                "Best practices for software version upgrades",
                "Troubleshooting upgrade failures in production"
            ]
        
        start_time = time.time()
        embeddings = []
        for text in test_texts:
            embedding = self.model.encode(text, convert_to_numpy=True)
            embeddings.append(embedding)
        total_time = time.time() - start_time
        
        results = {
            'time_per_text': total_time / len(test_texts),
            'total_time': total_time,
            'embedding_dimension': len(embeddings[0]),
            'model_info': self.get_model_info()
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Time per text: {results['time_per_text']:.4f}s")
        logger.info(f"  Total time: {results['total_time']:.4f}s")
        logger.info(f"  Embedding dimension: {results['embedding_dimension']}")
        
        return results
    
    def vectorize(self) -> None:
        """Vectorize the data with improved performance."""
        if self.chunked_df is None:
            raise ValueError("No data to vectorize")
            
        # Validate DataFrame structure
        required_columns = ['CHUNK_TEXT', 'OBJECTNAME', 'OLDVALUE', 'NEWVALUE']
        if not all(col in self.chunked_df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {required_columns}")
            
        # Validate data types and content
        if not all(isinstance(x, str) for x in self.chunked_df['CHUNK_TEXT']):
            raise ValueError("CHUNK_TEXT column must contain only strings")
        if not all(isinstance(x, str) for x in self.chunked_df['OBJECTNAME']):
            raise ValueError("OBJECTNAME column must contain only strings")
        if not all(isinstance(x, str) for x in self.chunked_df['OLDVALUE']):
            raise ValueError("OLDVALUE column must contain only strings")
        if not all(isinstance(x, str) for x in self.chunked_df['NEWVALUE']):
            raise ValueError("NEWVALUE column must contain only strings")
            
        logger.info(f"Generating vectors for {len(self.chunked_df)} records...")
        start_time = time.time()
        
        try:
            # Generate embeddings in batches
            batch_size = 32  # Larger batches for sentence transformers
            embeddings = []
            
            for i in range(0, len(self.chunked_df), batch_size):
                batch = self.chunked_df['CHUNK_TEXT'].iloc[i:i+batch_size].tolist()
                
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                
                # Format embeddings for database storage
                for j, embedding in enumerate(batch_embeddings):
                    idx = i + j
                    vector_data = {
                        'log_id': f"log_{idx}",
                        'chunk_text': self.chunked_df.iloc[idx]['CHUNK_TEXT'],
                        'object_name': self.chunked_df.iloc[idx]['OBJECTNAME'],
                        'change_type': 'UPGRADE' if not self.chunked_df.iloc[idx].get('IS_ROLLBACK', False) else 'ROLLBACK',
                        'version_info': {
                            'old_version': self.chunked_df.iloc[idx]['OLDVALUE'],
                            'new_version': self.chunked_df.iloc[idx]['NEWVALUE']
                        },
                        'embedding': embedding
                    }
                    embeddings.append(vector_data)
            
            # Store vectors
            if self.use_database:
                self.db.store_vectors(embeddings)
            
            logger.info(f"Generated {len(embeddings)} vectors successfully")
            logger.info(f"Vectorization completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during vectorization: {str(e)}")
            # Clear any partial vectors
            if self.use_database:
                try:
                    with self.db.engine.connect() as conn:
                        conn.execute(text("DELETE FROM upgrades"))
                        conn.execute(text("DELETE FROM upgrades_fts"))
                        conn.commit()
                except Exception as db_error:
                    logger.error(f"Error clearing database after vectorization failure: {str(db_error)}")
            raise
    
    def _calculate_similarity(self, query_embedding: np.ndarray, 
                            target_embedding: np.ndarray) -> float:
        """Calculate cosine similarity with improved accuracy."""
        try:
            # Normalize vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            target_norm = target_embedding / np.linalg.norm(target_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(query_norm, target_norm)
            
            # Apply sigmoid function to enhance differences
            return 1 / (1 + np.exp(-5 * (similarity - 0.5)))
        except Exception as e:
            logger.warning(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with context and intent."""
        try:
            # Parse query intent
            intent = self.query_parser.parse_query(query)
            
            # Add context based on intent
            if intent == 'upgrade':
                query += " software version upgrade process"
            elif intent == 'rollback':
                query += " software version rollback process"
            elif intent == 'issue':
                query += " software version issues and problems"
            
            return query
        except Exception as e:
            logger.warning(f"Error enhancing query: {str(e)}")
            return query
    
    def query_upgrades(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query similar upgrades with improved matching and caching."""
        try:
            # Validate input data
            if not query or not isinstance(query, str):
                logger.warning("Invalid query: empty or non-string")
                return []
                
            if self.chunked_df is None or len(self.chunked_df) == 0:
                logger.warning("No data available for querying")
                return []
                
            if not all(col in self.chunked_df.columns for col in ['CHUNK_TEXT', 'OBJECTNAME', 'OLDVALUE', 'NEWVALUE']):
                logger.warning("Invalid DataFrame structure")
                return []
            
            # Get query context
            context = self.query_parser.get_query_context(query)
            
            # Check cache first
            if self.use_cache:
                cached_results = self.query_cache.get(query, context)
                if cached_results is not None:
                    logger.info("Using cached results")
                    return cached_results
            
            # Enhance query
            enhanced_query = self._enhance_query(query)
            query_embedding = self._get_embedding(enhanced_query)
            
            # Get similar vectors from database
            if self.use_database:
                results = self.db.query_similar_vectors(query_embedding, min_similarity=0.1, top_k=top_k)
            else:
                # Fallback to in-memory comparison if database is not used
                vectors = self.model.encode(self.chunked_df['CHUNK_TEXT'].tolist(), convert_to_numpy=True)
                similarities = [self._calculate_similarity(query_embedding, vec) for vec in vectors]
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # Minimum similarity threshold
                        results.append({
                            'similarity': float(similarities[idx]),
                            'chunk_text': self.chunked_df.iloc[idx]['CHUNK_TEXT'],
                            'object_name': self.chunked_df.iloc[idx]['OBJECTNAME'],
                            'change_type': 'UPGRADE' if not self.chunked_df.iloc[idx].get('IS_ROLLBACK', False) else 'ROLLBACK',
                            'version_info': {
                                'old_version': self.chunked_df.iloc[idx]['OLDVALUE'],
                                'new_version': self.chunked_df.iloc[idx]['NEWVALUE']
                            }
                        })
            
            # Cache results
            if self.use_cache and results:
                self.query_cache.set(query, context, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying upgrades: {str(e)}")
            return []
    
    def generate_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a more detailed answer with context."""
        if not results:
            return "No similar upgrade experiences found."
        
        # Get query context
        context = self.query_parser.get_query_context(query)
        
        # Group results by change type
        grouped_results = {}
        for result in results:
            change_type = result['change_type']
            if change_type not in grouped_results:
                grouped_results[change_type] = []
            grouped_results[change_type].append(result)
        
        # Generate answer
        answer = f"Found {len(results)} similar upgrade experiences:\n\n"
        
        # Add context-specific information
        if context.get('is_software_specific'):
            software = context['detected_software'][0]
            if software in context.get('software_context', {}):
                sw_context = context['software_context'][software]
                answer += f"Context for {software}:\n"
                if sw_context.get('common_issues'):
                    answer += "Common issues to watch for:\n"
                    for issue in sw_context['common_issues'][:3]:
                        answer += f"- {issue}\n"
                answer += "\n"
        
        # Add suggested steps if available
        if 'context' in context and 'suggested_steps' in context['context']:
            answer += "Suggested steps:\n"
            for step in context['context']['suggested_steps']:
                answer += f"- {step}\n"
            answer += "\n"
        
        # Add detailed results
        for i, result in enumerate(results, 1):
            answer += f"{i}. Similarity: {result['similarity']:.2f}\n"
            answer += f"   Change Type: {result['change_type']}\n"
            answer += f"   Object: {result['object_name']}\n"
            
            if 'version' in result:
                answer += f"   Version: {result['version_info']['old_version']} to {result['version_info']['new_version']}\n"
            
            answer += f"   Details: {result['chunk_text']}...\n\n"
        
        # Add summary based on change types
        if len(grouped_results) > 1:
            answer += "\nSummary:\n"
            for change_type, type_results in grouped_results.items():
                answer += f"- {len(type_results)} {change_type} experiences\n"
        
        return answer
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the vectorized data."""
        if self.chunked_df is None:
            return {}
        
        try:
            stats = {
                'total_records': len(self.chunked_df),
                'change_type_distribution': self.chunked_df['CHANGE_TYPE'].value_counts().to_dict() if 'CHANGE_TYPE' in self.chunked_df.columns else {},
                'object_name_distribution': self.chunked_df['OBJECTNAME'].value_counts().to_dict()
            }
            
            # Add version statistics if available
            if 'VERSION' in self.chunked_df.columns:
                stats['version_distribution'] = self.chunked_df['VERSION'].value_counts().to_dict()
            
            # Add database statistics if using database
            if self.use_database:
                db_stats = self.db.get_statistics()
                stats.update(db_stats)
            
            # Add cache statistics if using cache
            if self.use_cache and self.query_cache:
                cache_stats = self.query_cache.get_stats()
                stats['cache'] = cache_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 