import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import re
import time
from functools import lru_cache
import threading
import os
from sklearn.preprocessing import normalize
from collections import defaultdict
import math
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

class WebServerVectorizer:
    def __init__(self):
        """Initialize the WebServerVectorizer with enhanced context patterns."""
        self.model = None
        self.data_df = None
        self.vectors = None
        self.vector_cache = {}
        self.model_lock = threading.Lock()
        
        # Required columns for data validation
        self.required_columns = [
            'ASSETNAME', 'MANUFACTURER', 'MODEL', 'ENVIRONMENT',
            'INSTALLPATH', 'INSTANCENAME', 'STATUS', 'SUBSTATUS',
            'PRODUCTCLASS', 'PRODUCTTYPE'
        ]
        
        # Enhanced context weights with more granular control
        self.context_weights = {
            'environment': 2.0,    # Environment matching is crucial
            'manufacturer': 1.8,   # Manufacturer matching is important
            'product_class': 1.5,  # Product class helps narrow down results
            'status': 1.3,        # Status provides operational context
            'version': 1.6,       # Version matching is important for compatibility
            'instance_name': 1.2   # Instance name helps identify specific servers
        }
        
        # Enhanced context patterns for better matching
        self.context_patterns = {
            'environment': {
                'PROD': r'\b(prod|production|live|production\s+environment)\b',
                'DEV': r'\b(dev|development|test|development\s+environment)\b',
                'UAT': r'\b(uat|staging|pre-prod|user\s+acceptance\s+testing)\b'
            },
            'manufacturer': {
                'APACHE': r'\b(apache|tomcat|httpd)\b',
                'IBM': r'\b(ibm|websphere)\b',
                'NGINX': r'\b(nginx|engine\s+x)\b'
            },
            'product_class': {
                'APPLICATION SERVER': r'\b(app\s+server|application\s+server|tomcat|websphere)\b',
                'HTTP SERVER': r'\b(http\s+server|web\s+server|httpd|nginx)\b'
            },
            'status': {
                'Installed': r'\b(installed|running|active|operational)\b',
                'Not Installed': r'\b(not\s+installed|inactive|stopped)\b'
            }
        }
        
        # Initialize the model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model with caching and thread safety."""
        if self.model is None:
            with self.model_lock:
                if self.model is None:  # Double-check pattern
                    logger.info("Loading Universal Sentence Encoder model...")
                    try:
                        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
                        logger.info("Model loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading model: {str(e)}")
                        raise
    
    def validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data with enhanced error handling."""
        try:
            # Check for required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Check for null values in critical columns
            critical_cols = ['ASSETNAME', 'MANUFACTURER', 'ENVIRONMENT', 'STATUS']
            null_cols = [col for col in critical_cols if df[col].isnull().any()]
            if null_cols:
                raise DataValidationError(f"Null values found in critical columns: {', '.join(null_cols)}")
            
            # Validate environment values
            valid_envs = {'PROD', 'DEV', 'UAT'}
            invalid_envs = set(df['ENVIRONMENT'].unique()) - valid_envs
            if invalid_envs:
                raise DataValidationError(f"Invalid environment values found: {', '.join(invalid_envs)}")
            
            # Validate status values
            valid_statuses = {'Installed', 'Not Installed'}
            invalid_statuses = set(df['STATUS'].unique()) - valid_statuses
            if invalid_statuses:
                raise DataValidationError(f"Invalid status values found: {', '.join(invalid_statuses)}")
            
            logger.info("Data validation successful")
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def _standardize_version(self, version: str) -> str:
        """Standardize version numbers to a consistent format."""
        if pd.isna(version):
            return ""
        try:
            # Extract numbers and dots, ignoring other characters
            version = str(version).strip()
            # Remove any non-numeric characters except dots
            version = ''.join(c for c in version if c.isdigit() or c == '.')
            # Split by dots and take first 3 components
            parts = version.split('.')
            # Pad with zeros if needed
            parts = [p.zfill(2) for p in parts[:3]]
            return '.'.join(parts)
        except Exception as e:
            logger.warning(f"Error standardizing version {version}: {str(e)}")
            return str(version)

    def _standardize_environment(self, env: str) -> str:
        """Standardize environment values."""
        if pd.isna(env):
            return ""
        try:
            env = str(env).strip().upper()
            # Remove any special characters and extra spaces
            env = ''.join(c for c in env if c.isalnum() or c.isspace())
            return env.strip()
        except Exception as e:
            logger.warning(f"Error standardizing environment {env}: {str(e)}")
            return str(env)

    def _standardize_manufacturer(self, manufacturer: str) -> str:
        """Standardize manufacturer names."""
        if pd.isna(manufacturer):
            return ""
        try:
            manufacturer = str(manufacturer).strip().upper()
            # Remove any special characters and extra spaces
            manufacturer = ''.join(c for c in manufacturer if c.isalnum() or c.isspace())
            return manufacturer.strip()
        except Exception as e:
            logger.warning(f"Error standardizing manufacturer {manufacturer}: {str(e)}")
            return str(manufacturer)

    def _standardize_product_class(self, product_class: str) -> str:
        """Standardize product class values."""
        if pd.isna(product_class):
            return ""
        try:
            product_class = str(product_class).strip().upper()
            # Remove any special characters and extra spaces
            product_class = ''.join(c for c in product_class if c.isalnum() or c.isspace())
            return product_class.strip()
        except Exception as e:
            logger.warning(f"Error standardizing product class {product_class}: {str(e)}")
            return str(product_class)

    def _standardize_status(self, status: str) -> str:
        """Standardize status values."""
        if pd.isna(status):
            return ""
        try:
            status = str(status).strip().upper()
            # Remove any special characters and extra spaces
            status = ''.join(c for c in status if c.isalnum() or c.isspace())
            return status.strip()
        except Exception as e:
            logger.warning(f"Error standardizing status {status}: {str(e)}")
            return str(status)

    def _standardize_instance_name(self, instance: str) -> str:
        """Standardize instance names."""
        if pd.isna(instance):
            return ""
        try:
            instance = str(instance).strip()
            # Remove any special characters except alphanumeric and basic punctuation
            instance = ''.join(c for c in instance if c.isalnum() or c in '._- ')
            return instance.strip()
        except Exception as e:
            logger.warning(f"Error standardizing instance name {instance}: {str(e)}")
            return str(instance)

    def _clean_text(self, text: str) -> str:
        """Clean and standardize text fields."""
        if pd.isna(text):
            return ""
        try:
            text = str(text).strip()
            # Remove any special characters except alphanumeric and basic punctuation
            text = ''.join(c for c in text if c.isalnum() or c in '._- ')
            return text.strip()
        except Exception as e:
            logger.warning(f"Error cleaning text {text}: {str(e)}")
            return str(text)

    def _extract_context(self, query: str) -> Dict[str, Optional[str]]:
        """Extract context information from query."""
        context = {
            'environment': None,
            'manufacturer': None,
            'product_class': None,
            'status': None,
            'version': None,
            'instance_name': None
        }
        
        try:
            # Environment patterns - more flexible matching
            env_patterns = [
                r'(?:in|from|at)\s+(\w+)(?:\s+(?:environment|env|server))?',
                r'(?:environment|env|server)\s+(?:is|are|in)\s+(\w+)',
                r'(?:running|deployed|installed)\s+(?:in|on)\s+(\w+)',
                r'(\w+)\s+(?:environment|env|server)'
            ]
            
            # Manufacturer patterns
            mfr_patterns = [
                r'(?:from|by|using)\s+(\w+)(?:\s+(?:server|instance))?',
                r'(\w+)(?:\s+(?:server|instance))',
                r'(?:manufacturer|vendor)\s+(?:is|are)\s+(\w+)',
                r'(\w+)\s+(?:servers|instances)'
            ]
            
            # Product class patterns
            prod_patterns = [
                r'(?:type|class)\s+(?:is|are)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:server|instance)',
                r'(?:running|using)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:servers|instances)'
            ]
            
            # Status patterns
            status_patterns = [
                r'(?:status|state)\s+(?:is|are)\s+(\w+(?:\s+\w+)*)',
                r'(?:running|active|inactive|installed|not\s+installed)',
                r'(?:find|show|list)\s+(?:all\s+)?(\w+(?:\s+\w+)*)\s+(?:servers|instances)',
                r'(\w+(?:\s+\w+)*)\s+(?:servers|instances)'
            ]
            
            # Version patterns
            version_patterns = [
                r'(?:version|v)\s+(\d+(?:\.\d+)*)',
                r'(\d+(?:\.\d+)*)',
                r'(?:using|running)\s+(\d+(?:\.\d+)*)',
                r'(\d+(?:\.\d+)*)\s+(?:version|v)'
            ]
            
            # Instance name patterns
            instance_patterns = [
                r'(?:instance|name)\s+(?:is|are)\s+(\w+(?:\s+\w+)*)',
                r'(?:called|named)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:instance|server)',
                r'(\w+(?:\s+\w+)*)\s+(?:instances|servers)'
            ]
            
            # Extract context using patterns
            for pattern in env_patterns:
                if match := re.search(pattern, query, re.IGNORECASE):
                    context['environment'] = self._standardize_environment(match.group(1))
                    break
                    
            for pattern in mfr_patterns:
                if match := re.search(pattern, query, re.IGNORECASE):
                    context['manufacturer'] = self._standardize_manufacturer(match.group(1))
                    break
                    
            for pattern in prod_patterns:
                if match := re.search(pattern, query, re.IGNORECASE):
                    context['product_class'] = self._standardize_product_class(match.group(1))
                    break
                    
            for pattern in status_patterns:
                if match := re.search(pattern, query, re.IGNORECASE):
                    if len(match.groups()) > 0:
                        context['status'] = self._standardize_status(match.group(1))
                    else:
                        # Handle patterns without capture groups
                        status = re.search(r'(running|active|inactive|installed|not\s+installed)', query, re.IGNORECASE)
                        if status:
                            context['status'] = self._standardize_status(status.group(1))
                    break
                    
            for pattern in version_patterns:
                if match := re.search(pattern, query, re.IGNORECASE):
                    context['version'] = self._standardize_version(match.group(1))
                    break
                    
            for pattern in instance_patterns:
                if match := re.search(pattern, query, re.IGNORECASE):
                    context['instance_name'] = self._standardize_instance_name(match.group(1))
                    break
                    
        except Exception as e:
            logger.error(f"Error extracting context from query: {str(e)}")
            
        return context

    def _enhance_query(self, query: str) -> str:
        """Enhance query with additional context information."""
        try:
            # Extract context
            context = self._extract_context(query)
            
            # Build enhanced query
            enhanced_parts = [query]
            
            # Add environment context
            if context['environment']:
                enhanced_parts.append(f"in {context['environment']} environment")
            
            # Add manufacturer context
            if context['manufacturer']:
                enhanced_parts.append(f"from {context['manufacturer']}")
            
            # Add status context
            if context['status']:
                enhanced_parts.append(f"with status {context['status']}")
            
            # Add version context
            if context['version']:
                enhanced_parts.append(f"version {context['version']}")
            
            # Combine parts
            enhanced_query = " ".join(enhanced_parts)
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return query
    
    def load_data(self, csv_path: str) -> None:
        """Load and prepare web server data for vectorization."""
        try:
            logger.info(f"Loading data from {csv_path}...")
            self.data_df = pd.read_csv(csv_path)
            
            # Validate input data
            self.validate_input_data(self.data_df)
            
            # Clean and standardize text fields
            for col in self.data_df.columns:
                if self.data_df[col].dtype == 'object':
                    self.data_df[col] = self.data_df[col].apply(self._clean_text)
            
            # Standardize versions in MODEL column
            self.data_df['MODEL'] = self.data_df['MODEL'].apply(self._standardize_version)
            
            # Create rich descriptive text for each server
            self.data_df['SERVER_DESCRIPTION'] = self.data_df.apply(
                lambda row: (
                    f"{row['MANUFACTURER']} {row['MODEL']} "
                    f"installed in {row['ENVIRONMENT']} environment. "
                    f"Path: {row['INSTALLPATH']}, "
                    f"Instance: {row['INSTANCENAME']}. "
                    f"Status: {row['STATUS']} - {row['SUBSTATUS']}. "
                    f"Product Class: {row['PRODUCTCLASS']}, "
                    f"Product Type: {row['PRODUCTTYPE']}"
                ),
                axis=1
            )
            
            # Create additional context columns
            self.data_df['ENVIRONMENT_CONTEXT'] = self.data_df['ENVIRONMENT'].map({
                'PROD': 'Production environment for live services',
                'UAT': 'User Acceptance Testing environment',
                'DEV': 'Development environment for testing'
            })
            
            self.data_df['MANUFACTURER_CONTEXT'] = self.data_df['MANUFACTURER'].map({
                'APACHE': 'Apache Software Foundation',
                'IBM': 'IBM Corporation',
                'NGINX': 'NGINX Inc'
            })
            
            logger.info(f"Loaded {len(self.data_df)} web server records")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def vectorize(self) -> Dict[str, np.ndarray]:
        """Vectorize the web server data with context-aware embeddings."""
        if self.data_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info(f"Generating vectors for {len(self.data_df)} web servers...")
        start_time = time.time()
        
        try:
            # Generate embeddings in batches
            batch_size = 32
            self.vectors = {}
            
            for i in range(0, len(self.data_df), batch_size):
                batch = self.data_df['SERVER_DESCRIPTION'].iloc[i:i+batch_size].tolist()
                batch_embeddings = self.model(batch).numpy()
                
                # Store vectors with server identifiers
                for j, embedding in enumerate(batch_embeddings):
                    idx = i + j
                    server_id = self.data_df.iloc[idx]['ASSETNAME']
                    self.vectors[server_id] = embedding
            
            # Normalize all vectors
            vector_matrix = np.array(list(self.vectors.values()))
            normalized_vectors = normalize(vector_matrix)
            for server_id, norm_vector in zip(self.vectors.keys(), normalized_vectors):
                self.vectors[server_id] = norm_vector
            
            logger.info(f"Generated {len(self.vectors)} vectors successfully")
            logger.info(f"Vectorization completed in {time.time() - start_time:.2f} seconds")
            
            return self.vectors
            
        except Exception as e:
            logger.error(f"Error during vectorization: {str(e)}")
            raise
    
    def _calculate_similarity(self, query_embedding: np.ndarray, server_embedding: np.ndarray, 
                            context: Dict[str, Optional[str]], server_row: pd.Series) -> float:
        """Calculate similarity between query and server with context awareness."""
        try:
            def safe_get_value(key: str) -> str:
                """Safely get value from Series or direct value."""
                if isinstance(server_row, pd.Series):
                    return str(server_row.get(key, ""))
                return str(server_row.get(key, ""))
            
            # Calculate base similarity using cosine similarity
            base_similarity = float(np.dot(query_embedding, server_embedding) / 
                                  (np.linalg.norm(query_embedding) * np.linalg.norm(server_embedding)))
            
            # Initialize context similarity components
            context_similarity = 0.0
            total_weight = 0.0
            
            # Environment matching
            if context.get('environment'):
                env_match = self._standardize_environment(safe_get_value('ENVIRONMENT')) == context['environment']
                context_similarity += self.context_weights['environment'] * float(env_match)
                total_weight += self.context_weights['environment']
            
            # Manufacturer matching
            if context.get('manufacturer'):
                mfr_match = self._standardize_manufacturer(safe_get_value('MANUFACTURER')) == context['manufacturer']
                context_similarity += self.context_weights['manufacturer'] * float(mfr_match)
                total_weight += self.context_weights['manufacturer']
            
            # Product class matching
            if context.get('product_class'):
                prod_match = self._standardize_product_class(safe_get_value('PRODUCTCLASS')) == context['product_class']
                context_similarity += self.context_weights['product_class'] * float(prod_match)
                total_weight += self.context_weights['product_class']
            
            # Status matching
            if context.get('status'):
                status_match = self._standardize_status(safe_get_value('STATUS')) == context['status']
                context_similarity += self.context_weights['status'] * float(status_match)
                total_weight += self.context_weights['status']
            
            # Version matching
            if context.get('version'):
                version_match = self._standardize_version(safe_get_value('MODEL')) == context['version']
                context_similarity += self.context_weights['version'] * float(version_match)
                total_weight += self.context_weights['version']
            
            # Instance name matching
            if context.get('instance_name'):
                instance_match = self._standardize_instance_name(safe_get_value('INSTANCENAME')) == context['instance_name']
                context_similarity += self.context_weights['instance_name'] * float(instance_match)
                total_weight += self.context_weights['instance_name']
            
            # Normalize context similarity
            if total_weight > 0:
                context_similarity /= total_weight
            
            # Combine base and context similarity (70% base, 30% context)
            final_similarity = 0.7 * base_similarity + 0.3 * context_similarity
            
            # Apply sigmoid function to normalize between 0 and 1
            final_similarity = 1 / (1 + np.exp(-5 * (final_similarity - 0.5)))
            
            return float(final_similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """Format a single result for display."""
        try:
            # Extract values safely
            similarity = result.get('similarity', 0.0)
            asset_name = result.get('ASSETNAME', '')
            manufacturer = result.get('MANUFACTURER', '')
            model = result.get('MODEL', '')
            environment = result.get('ENVIRONMENT', '')
            status = result.get('STATUS', '')
            product_class = result.get('PRODUCTCLASS', '')
            install_path = result.get('INSTALLPATH', '')
            instance_name = result.get('INSTANCENAME', '')
            
            # Format the result
            formatted = [
                f"Similarity: {similarity:.2f}",
                f"Asset: {asset_name}",
                f"Server: {manufacturer} {model}",
                f"Environment: {environment}",
                f"Status: {status}",
                f"Type: {product_class}",
                f"Path: {install_path}",
                f"Instance: {instance_name}"
            ]
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting result: {str(e)}")
            return "Error formatting result"
    
    def find_similar_servers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar servers based on query with enhanced context awareness."""
        try:
            # Enhance query and extract context
            enhanced_query = self._enhance_query(query)
            context = self._extract_context(enhanced_query)
            
            # Get query embedding
            query_embedding = self.model([enhanced_query])[0].numpy()
            
            # Process each server
            results = []
            for idx, row in self.data_df.iterrows():
                try:
                    # Get server embedding
                    server_text = f"{row['ASSETNAME']} {row['MANUFACTURER']} {row['MODEL']} {row['ENVIRONMENT']}"
                    server_embedding = self.model([server_text])[0].numpy()
                    
                    # Calculate similarity
                    similarity = self._calculate_similarity(
                        query_embedding=query_embedding,
                        server_embedding=server_embedding,
                        context=context,
                        server_row=row
                    )
                    
                    # Add result if similarity is above threshold
                    if similarity > 0.1:  # Minimum similarity threshold
                        result = {
                            'similarity': similarity,
                            'ASSETNAME': row['ASSETNAME'],
                            'MANUFACTURER': row['MANUFACTURER'],
                            'MODEL': row['MODEL'],
                            'ENVIRONMENT': row['ENVIRONMENT'],
                            'STATUS': row['STATUS'],
                            'PRODUCTCLASS': row['PRODUCTCLASS'],
                            'INSTALLPATH': row['INSTALLPATH'],
                            'INSTANCENAME': row['INSTANCENAME'],
                            'row_data': row.to_dict()
                        }
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error processing server {idx}: {str(e)}")
                    continue
            
            # Sort results by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar servers: {str(e)}")
            return []
    
    def _get_environment_description(self, env: str) -> str:
        """Get a human-readable description of the environment."""
        descriptions = {
            'PROD': 'Production environment for live services',
            'DEV': 'Development environment for testing',
            'UAT': 'User Acceptance Testing environment'
        }
        return descriptions.get(env, 'Unknown environment')
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the web server data."""
        if self.data_df is None:
            return {}
            
        try:
            stats = {
                'total_servers': len(self.data_df),
                'environment_distribution': self.data_df['ENVIRONMENT'].value_counts().to_dict(),
                'manufacturer_distribution': self.data_df['MANUFACTURER'].value_counts().to_dict(),
                'product_class_distribution': self.data_df['PRODUCTCLASS'].value_counts().to_dict(),
                'status_distribution': self.data_df['STATUS'].value_counts().to_dict(),
                'product_type_distribution': self.data_df['PRODUCTTYPE'].value_counts().to_dict(),
                'model_distribution': self.data_df['MODEL'].value_counts().to_dict()
            }
            
            # Calculate percentages
            for key in stats:
                if key != 'total_servers':
                    total = sum(stats[key].values())
                    stats[key] = {k: (v/total)*100 for k, v in stats[key].items()}
            
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def save_vectors(self, output_path: str) -> None:
        """Save vector embeddings to a file."""
        try:
            if self.vectors is None:
                raise ValueError("No vectors available. Call vectorize() first.")
            
            # Convert vectors to a format that can be saved
            vector_data = {
                server_id: vector.tolist() for server_id, vector in self.vectors.items()
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(vector_data, f)
            
            logger.info(f"Saved vectors to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving vectors: {str(e)}")
            raise
    
    def load_vectors(self, input_path: str) -> None:
        """Load vector embeddings from a file."""
        try:
            with open(input_path, 'r') as f:
                vector_data = json.load(f)
            
            # Convert back to numpy arrays
            self.vectors = {
                server_id: np.array(vector) for server_id, vector in vector_data.items()
            }
            
            logger.info(f"Loaded vectors from {input_path}")
            
        except Exception as e:
            logger.error(f"Error loading vectors: {str(e)}")
            raise

if __name__ == '__main__':
    # Initialize vectorizer
    vectorizer = WebServerVectorizer()
    
    # Load and process data
    vectorizer.load_data('data/processed/WebServer_Merged.csv')
    vectorizer.vectorize()
    
    # Example queries
    example_queries = [
        "Find Apache Tomcat servers in production environment",
        "Show me all IBM WebSphere instances",
        "List all development servers",
        "Find active HTTP servers",
        "Find Tomcat 9.0 instances",
        "Show me all running servers in UAT",
        "List all Apache HTTPD servers version 2.4",
        "Find inactive servers in development",
        "Show me all Nginx instances in production",
        "List all application servers with version 8.5"
    ]
    
    # Process each query
    for query in example_queries:
        print("\n" + "="*80)
        print(f"Query: {query}")
        print("="*80)
        
        results = vectorizer.find_similar_servers(query)
        
        print("\nQuery Results:")
        for result in results:
            print(vectorizer._format_result(result))
    
    # Show statistics
    print("\n" + "="*80)
    print("Server Statistics")
    print("="*80)
    stats = vectorizer.get_server_statistics()
    print(f"\nTotal Servers: {stats['total_servers']}\n")
    
    print("Environment Distribution:")
    for env, count in stats['environment_distribution'].items():
        print(f"{env}: {count:.1f}%")
    
    print("\nManufacturer Distribution:")
    for mfr, count in stats['manufacturer_distribution'].items():
        print(f"{mfr}: {count:.1f}%")
    
    print("\nProduct Class Distribution:")
    for pc, count in stats['product_class_distribution'].items():
        print(f"{pc}: {count:.1f}%")
    
    # Save vectors for future use
    vectorizer.save_vectors('data/processed/webserver_vectors.json') 