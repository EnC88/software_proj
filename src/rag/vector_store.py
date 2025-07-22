#!/usr/bin/env python3
"""
Vector Store for RAG Pipeline
Industry-standard vector store implementation using local components
Works completely offline with document search only
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
import re
from collections import defaultdict, Counter

# Add parent directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import faiss 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define repo root for robust file access
# Go up more levels to reach the root where tlcaas directory is located
REPO_ROOT = Path(__file__).resolve().parents[4]  # Go up 4 levels instead of 2

# Import CheckCompatibility, ChangeRequest, and CompatibilityResult from the correct module
from src.rag.determine_recs import CheckCompatibility, ChangeRequest, CompatibilityResult

class SimpleVectorStore:
    """Simple vector store using TF-IDF for offline operation."""
    
    def __init__(self, documents: List[Document] = None):
        self.documents = documents or []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.faiss_index = None
        self._build_index()
    

    def _build_index(self):
        """Build TF-IDF index from documents."""
        try:
            if not self.documents:
                logger.warning("No documents to build index from")
                return

            
            # Extract text content
            texts = [doc.page_content for doc in self.documents]
            logger.info(f"Building TF-IDF index with {len(texts)} documents")
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Build TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            logger.info(f"Built TF-IDF index with {len(self.documents)} documents, matrix shape: {self.tfidf_matrix.shape}")
            
            # Create FAISS index for similarity search
            try:
                logger.info("Converting sparse matrix to dense for FAISS...")
                dense_matrix = self.tfidf_matrix.toarray().astype('float32')
                logger.info(f"Dense matrix shape: {dense_matrix.shape}")
                
                dimension = dense_matrix.shape[1]
                logger.info(f"Creating FAISS index with dimension: {dimension}")
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                logger.info("Adding vectors to FAISS index...")
                self.faiss_index.add(dense_matrix)
                logger.info(f"Built FAISS index with dimension: {dimension}")
                
            except Exception as faiss_error:
                logger.error(f"Error creating FAISS index: {faiss_error}")
                self.faiss_index = None
            
        except Exception as e:
            logger.error(f"Error building TF-IDF index: {e}")
            self.vectorizer = None
            self.tfidf_matrix = None
            self.faiss_index = None
    
    def similarity_search(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents using FAISS with TF-IDF vectors.
        
        Returns:
            List of tuples: (document, score)
        """
        try:
            if self.vectorizer is None or self.tfidf_matrix is None:
                logger.warning("Vectorizer or TF-IDF matrix not available")
                return []
            
            if self.faiss_index is None:
                logger.warning("FAISS index not available - similarity search cannot be performed")
                return []
            
            # Check if we have any documents
            if not self.documents:
                logger.warning("No documents available for search")
                return []
            
            # Vectorize query using TF-IDF
            query_vector = self.vectorizer.transform([query])
            
            # Use FAISS for similarity search
            query_dense = query_vector.toarray().astype('float32')
            scores, indices = self.faiss_index.search(query_dense, k)
            
            logger.info(f"FAISS search returned scores shape: {scores.shape}, indices shape: {indices.shape}")
            
            # Convert to list of (document, score) tuples
            results = []
            
            # Process each result
            for i in range(len(indices[0])):
                try:
                    # Get index and score as Python scalars
                    doc_idx = int(indices[0][i].item())
                    doc_score = float(scores[0][i].item())
                    
                    # Check if valid
                    if doc_idx < len(self.documents) and doc_score > 0.0:
                        results.append((self.documents[doc_idx], doc_score))
                        
                except Exception as item_error:
                    logger.warning(f"Error processing result {i}: {item_error}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity_search: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the store."""
        self.documents.extend(documents)
        self._build_index()  # Rebuild index
    
    def save_local(self, path: str):
        """Save the vector store to disk."""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            docs_data = []
            for doc in self.documents:
                docs_data.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            with open(save_path / 'documents.json', 'w') as f:
                json.dump(docs_data, f, indent=2)
            
            # Save vectorizer
            import pickle
            with open(save_path / 'vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save TF-IDF matrix
            import scipy.sparse
            scipy.sparse.save_npz(save_path / 'tfidf_matrix.npz', self.tfidf_matrix)
            
            logger.info(f"Vector store saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    @classmethod
    def load_local(cls, path: str):
        """Load the vector store from disk."""
        try:
            load_path = Path(path)
            
            # Load documents
            with open(load_path / 'documents.json', 'r') as f:
                docs_data = json.load(f)
            
            documents = []
            for doc_data in docs_data:
                documents.append(Document(
                    page_content=doc_data['content'],
                    metadata=doc_data['metadata']
                ))
            
            # Create instance
            instance = cls(documents)
            
            # Load vectorizer and matrix
            import pickle
            import scipy.sparse
            
            with open(load_path / 'vectorizer.pkl', 'rb') as f:
                instance.vectorizer = pickle.load(f)
            
            instance.tfidf_matrix = scipy.sparse.load_npz(load_path / 'tfidf_matrix.npz')
            
            logger.info(f"Vector store loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return cls()

    @classmethod
    def from_compatibility_json(cls, analysis_path=None):
        """Build a SimpleVectorStore from compatibility_analysis.json."""
        import json
        from langchain.schema import Document
        from datetime import datetime
        import logging
        logger = logging.getLogger(__name__)
        
        # Determine default path if not provided
        if analysis_path is None:
            # Use the same logic as VectorStore for default data dir
            try:
                REPO_ROOT = Path(__file__).resolve().parents[4]
            except Exception:
                REPO_ROOT = Path('.')
            data_dir = REPO_ROOT / 'tlcaas' / 'tlcaas_root' / 'tlcaas-compliance-advisor' / 'data'
            analysis_path = data_dir / 'processed' / 'compatibility_analysis.json'
        else:
            analysis_path = Path(analysis_path)
        
        documents = []
        if not analysis_path.exists():
            logger.error(f"compatibility_analysis.json not found at {analysis_path}")
            return cls([])
        try:
            with open(analysis_path, 'r') as f:
                analysis_data = json.load(f)
            # Process servers
            for server in analysis_data.get('servers', []):
                server_info = server.get('server_info', {})
                model = server_info.get('model', 'Unknown')
                product_type = server_info.get('product_type', 'Unknown')
                environment = server.get('environment', 'Unknown')
                content = (
                    f"Server Model: {model}\n"
                    f"Product Type: {product_type}\n"
                    f"Environment: {environment}\n"
                    f"Server ID: {server.get('id', 'Unknown')}\n\n"
                    f"Software Information:\n"
                    f"{json.dumps(server.get('software_info', {}), indent=2)}\n\n"
                    f"Compatibility Status: {server.get('compatibility_status', 'Unknown')}"
                )
                metadata = {
                    'type': 'server_info',
                    'model': model,
                    'product_type': product_type,
                    'environment': environment,
                    'server_id': server.get('id', 'Unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                documents.append(Document(page_content=content, metadata=metadata))
            # Process compatibility rules
            for rule in analysis_data.get('compatibility_rules', []):
                content = (
                    f"Compatibility Rule:\n"
                    f"Software: {rule.get('software', 'Unknown')}\n"
                    f"Version: {rule.get('version', 'Unknown')}\n"
                    f"OS Compatibility: {rule.get('os_compatibility', 'Unknown')}\n"
                    f"Dependencies: {rule.get('dependencies', [])}\n"
                    f"Conflicts: {rule.get('conflicts', [])}"
                )
                metadata = {
                    'type': 'compatibility_rule',
                    'software': rule.get('software', 'Unknown'),
                    'version': rule.get('version', 'Unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                documents.append(Document(page_content=content, metadata=metadata))
            # Process SOR history
            for sor_record in analysis_data.get('sor_history', []):
                env = sor_record.get("ENVIRONMENT", "Unknown")
                attr = sor_record.get("ATTRIBUTENAME", "Unknown")
                old_val = sor_record.get("OLDVALUE", "Unknown")
                new_val = sor_record.get("NEWVALUE", "Unknown")
                old_mapped = sor_record.get("OLD_MAPPED", "Unknown")
                new_mapped = sor_record.get("NEW_MAPPED", "Unknown")
                osi_user = sor_record.get("VERUMCREATEDBY", "Unknown")
                date = sor_record.get("VERUMCREATEDDATE", "Unknown")
                pattern_str = (
                    f"Environment: {env}. "
                    f"User: {osi_user}. "
                    f"Date: {date}. "
                    f"Change: {attr} from {old_val} ({old_mapped}) to {new_val} ({new_mapped})."
                )
                metadata = {k: sor_record.get(k, None) for k in sor_record.keys()}
                documents.append(Document(page_content=pattern_str, metadata=metadata))
        except Exception as e:
            logger.error(f"Error loading or processing compatibility_analysis.json: {e}")
            return cls([])
        return cls(documents)

class VectorStore:
    """Industry-standard vector store implementation using local components.
    
    Works completely offline with document search only.
    No API keys or paid services required.
    """
    
    def __init__(self, 
                 data_dir: str = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize the vector store.
        
        Args:
            data_dir: Directory containing source data
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between text chunks
        """
        self.data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'tlcaas' / 'tlcaas_root' / 'tlcaas-compliance-advisor' / 'data'
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Debug logging
        logger.info(f"REPO_ROOT: {REPO_ROOT}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Data directory exists: {self.data_dir.exists()}")
        
        # Initialize components
        self.vectorstore = None
        self.text_splitter = None
        self.documents = []
        self.is_initialized = False
        
        # Initialize the system
        self._initialize_components()
        self._load_and_process_data()
        self.check_compat = CheckCompatibility()
    
    def _initialize_components(self):
        """Initialize components with improved error handling."""
        try:
            # No text splitter needed for simple document processing
            logger.info("Components initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _load_and_process_data(self):
        """Load and process data from various sources with improved error handling."""
        try:
            # Ensure data directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Load data from different sources
            self._load_compatibility_analysis()
            self._load_sor_history()
            self._load_webserver_data()
            
            # Create vectorstore if we have documents
            if self.documents:
                self._create_vectorstore()
            else:
                logger.warning("No documents loaded - vectorstore not created")
                
        except Exception as e:
            logger.error(f"Error loading and processing data: {e}")
            raise
    
    def _load_compatibility_analysis(self):
        """Load compatibility analysis data."""
        analysis_path = self.data_dir / 'processed' / 'compatibility_analysis.json'
        if analysis_path.exists():
            try:
                logger.info(f"Loading compatibility analysis from {analysis_path}")
                with open(analysis_path, 'r') as f:
                    analysis_data = json.load(f)
                self._process_analysis_data(analysis_data)
            except Exception as e:
                logger.error(f"Error loading compatibility analysis: {e}")
        else:
            logger.info("Compatibility analysis file not found")
    
    def _load_sor_history(self):
        """Load SOR history data."""
        sor_path = self.data_dir / 'raw' / 'sor_hist.csv'
        if sor_path.exists():
            try:
                logger.info(f"Loading SOR history from {sor_path}")
                sor_df = pd.read_csv(sor_path)
                self._process_sor_data(sor_df)
            except Exception as e:
                logger.error(f"Error loading SOR history: {e}")
        else:
            logger.info("SOR history file not found")
    
    def _load_webserver_data(self):
        """Load webserver data."""
        webserver_path = self.data_dir / 'raw' / 'WebServer.csv'
        if webserver_path.exists():
            try:
                logger.info(f"Loading webserver data from {webserver_path}")
                webserver_df = pd.read_csv(webserver_path)
                self._process_webserver_data(webserver_df)
            except Exception as e:
                logger.error(f"Error loading webserver data: {e}")
        else:
            logger.info("Webserver data file not found")
    
    def _create_vectorstore(self):
        """Create and save the vectorstore."""
        try:
            logger.info(f"Creating vectorstore with {len(self.documents)} documents")
            
            if len(self.documents) == 0:
                logger.warning("No documents available to create vectorstore")
                self.is_initialized = False
                return
            
            # Log some sample documents for debugging
            for i, doc in enumerate(self.documents[:3]):
                logger.info(f"Sample document {i+1}: {doc.page_content[:100]}...")
            
            self.vectorstore = SimpleVectorStore(self.documents)
            
            # Save vectorstore
            vectorstore_path = self.data_dir / 'processed' / 'vectorstore'
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(vectorstore_path))
            logger.info(f"Vectorstore saved to {vectorstore_path}")
            
            self.is_initialized = True
            logger.info("Vectorstore created and initialized successfully")
            
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise
    
    def _process_analysis_data(self, analysis_data: Dict[str, Any]):
        """Process compatibility analysis data into documents."""
        try:
            # Process server information
            for server in analysis_data.get('servers', []):
                self._create_server_document(server)
            
            # Process compatibility rules
            for rule in analysis_data.get('compatibility_rules', []):
                self._create_compatibility_rule_document(rule)
            
            # Process SOR history data (including OLD_MAPPED and NEW_MAPPED)
            for sor_record in analysis_data.get('sor_history', []):
                self._create_sor_history_document(sor_record)
                
        except Exception as e:
            logger.error(f"Error processing analysis data: {e}")
    
    def _create_server_document(self, server: Dict[str, Any]):
        """Create a document for server information."""
        server_info = server.get('server_info', {})
        model = server_info.get('model', 'Unknown')
        product_type = server_info.get('product_type', 'Unknown')
        environment = server.get('environment', 'Unknown')
        
        content = (
            f"Server Model: {model}\n"
            f"Product Type: {product_type}\n"
            f"Environment: {environment}\n"
            f"Server ID: {server.get('id', 'Unknown')}\n\n"
            f"Software Information:\n"
            f"{json.dumps(server.get('software_info', {}), indent=2)}\n\n"
            f"Compatibility Status: {server.get('compatibility_status', 'Unknown')}"
        )
        
        metadata = {
            'type': 'server_info',
            'model': model,
            'product_type': product_type,
            'environment': environment,
            'server_id': server.get('id', 'Unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        self.documents.append(Document(
            page_content=content,
            metadata=metadata
        ))
    
    def _create_compatibility_rule_document(self, rule: Dict[str, Any]):
        """Create a document for compatibility rules."""
        content = (
            f"Compatibility Rule:\n"
            f"Software: {rule.get('software', 'Unknown')}\n"
            f"Version: {rule.get('version', 'Unknown')}\n"
            f"OS Compatibility: {rule.get('os_compatibility', 'Unknown')}\n"
            f"Dependencies: {rule.get('dependencies', [])}\n"
            f"Conflicts: {rule.get('conflicts', [])}"
        )
        
        metadata = {
            'type': 'compatibility_rule',
            'software': rule.get('software', 'Unknown'),
            'version': rule.get('version', 'Unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        self.documents.append(Document(
            page_content=content,
            metadata=metadata
        ))
    
    def _create_sor_history_document(self, sor_record: Dict[str, Any]):
        """Create a document for SOR history with OLD_MAPPED and NEW_MAPPED data as a pattern string."""
        # Safely get fields, defaulting to 'Unknown' if missing
        env = sor_record.get("ENVIRONMENT", "Unknown")
        attr = sor_record.get("ATTRIBUTENAME", "Unknown")
        old_val = sor_record.get("OLDVALUE", "Unknown")
        new_val = sor_record.get("NEWVALUE", "Unknown")
        old_mapped = sor_record.get("OLD_MAPPED", "Unknown")
        new_mapped = sor_record.get("NEW_MAPPED", "Unknown")
        osi_user = sor_record.get("VERUMCREATEDBY", "Unknown")
        date = sor_record.get("VERUMCREATEDDATE", "Unknown")

        # Build the pattern string
        pattern_str = (
            f"Environment: {env}. "
            f"User: {osi_user}. "
            f"Date: {date}. "
            f"Change: {attr} from {old_val} ({old_mapped}) to {new_val} ({new_mapped})."
        )

        metadata = {k: sor_record.get(k, None) for k in sor_record.keys()}
        
        self.documents.append(Document(
            page_content=pattern_str,
            metadata=metadata
        ))
    
    def _process_sor_data(self, sor_df: pd.DataFrame):
        """Process SOR history data into documents."""
        try:
            for _, row in sor_df.iterrows():
                content = (
                    f"SOR Change Request:\n"
                    f"Object Name: {row.get('OBJECTNAME', 'Unknown')}\n"
                    f"Attribute Name: {row.get('ATTRIBUTENAME', 'Unknown')}\n"
                    f"Old Value: {row.get('OLDVALUE', 'Unknown')}\n"
                    f"New Value: {row.get('NEWVALUE', 'Unknown')}\n"
                    f"Change Date: {row.get('CHANGEDATE', 'Unknown')}\n"
                    f"Catalog ID: {row.get('CATALOGID', 'Unknown')}"
                )
                
                metadata = {
                    'type': 'sor_change',
                    'object_name': row.get('OBJECTNAME', 'Unknown'),
                    'attribute_name': row.get('ATTRIBUTENAME', 'Unknown'),
                    'catalog_id': row.get('CATALOGID', 'Unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
                
        except Exception as e:
            logger.error(f"Error processing SOR data: {e}")
    
    def _process_webserver_data(self, webserver_df: pd.DataFrame):
        """Process webserver data into documents."""
        try:
            for _, row in webserver_df.iterrows():
                content = (
                    f"Web Server Information:\n"
                    f"Server Name: {row.get('SERVERNAME', 'Unknown')}\n"
                    f"Model: {row.get('MODEL', 'Unknown')}\n"
                    f"Product Type: {row.get('PRODUCTTYPE', 'Unknown')}\n"
                    f"Environment: {row.get('ENVIRONMENT', 'Unknown')}\n"
                    f"OS: {row.get('OS', 'Unknown')}"
                )
                
                metadata = {
                    'type': 'webserver_info',
                    'server_name': row.get('SERVERNAME', 'Unknown'),
                    'model': row.get('MODEL', 'Unknown'),
                    'product_type': row.get('PRODUCTTYPE', 'Unknown'),
                    'environment': row.get('ENVIRONMENT', 'Unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
                
        except Exception as e:
            logger.error(f"Error processing webserver data: {e}")
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the vectorstore for relevant documents and include compatibility recommendations using CheckCompatibility."""
        if not self.is_initialized or not self.vectorstore:
            logger.error("Vectorstore not initialized")
            return {"error": "Vectorstore not available"}
        try:
            logger.info(f"Querying for: '{query_text}' with top_k={top_k}")
            logger.info(f"Vectorstore has {len(self.documents)} documents")
            # Get similar documents
            docs = self.vectorstore.similarity_search(query_text, k=top_k)
            logger.info(f"Found {len(docs)} similar documents")
            # Format results
            results = []
            for i, (doc, score) in enumerate(docs):
                result = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                results.append(result)
            # --- Use CheckCompatibility for recommendations and affected models ---
            comp_result = None
            try:
                # Note: This is a synchronous method, so we can't use async calls here
                # The VectorStore is not used in the main flow, so this is just for reference
                crs = []
                comp_results = []
                comp_result = [
                    {
                        'change_request': cr.__dict__,
                        'compatibility': {
                            'is_compatible': res.is_compatible,
                            'confidence': res.confidence,
                            'affected_servers': res.affected_servers,
                            'conflicts': res.conflicts,
                            'recommendations': res.recommendations,
                            'warnings': res.warnings,
                            'alternative_versions': res.alternative_versions
                        }
                    }
                    for cr, res in comp_results
                ]
            except Exception as e:
                logger.error(f"Error in CheckCompatibility: {e}")
            response = {
                'query': query_text,
                'results': results,
                'total_results': len(results)
            }
            if comp_result:
                response['compatibility_analysis'] = comp_result
            return response
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store."""
        stats = {
            'total_documents': len(self.documents),
            'vectorstore_available': self.vectorstore is not None,
            'is_initialized': self.is_initialized,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'cost': 'Completely Free (Local Only)',
            'privacy': '100% Local - No Data Sent to External Services',
            'embeddings': 'TF-IDF (Local, No External Dependencies)',
            'features': 'Document Search Only - No LLM Required'
        }
        
        # Document type breakdown
        doc_types = {}
        for doc in self.documents:
            doc_type = doc.metadata.get('type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        stats['document_types'] = doc_types
        
        return stats
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """Add a new document to the vectorstore.
        
        Args:
            content: Document content
            metadata: Document metadata
        """
        if not self.vectorstore:
            logger.error("Vectorstore not initialized")
            return False
        
        try:
            # Add timestamp if not provided
            if metadata is None:
                metadata = {}
            if 'timestamp' not in metadata:
                metadata['timestamp'] = datetime.now().isoformat()
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            self.vectorstore.add_documents([doc])
            self.documents.append(doc)
            
            logger.info("Document added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def reload_vectorstore(self):
        """Reload the vectorstore from saved files."""
        try:
            vectorstore_path = self.data_dir / 'processed' / 'vectorstore'
            if vectorstore_path.exists():
                self.vectorstore = SimpleVectorStore.load_local(str(vectorstore_path))
                logger.info("Vectorstore reloaded successfully")
                return True
            else:
                logger.warning("No saved vectorstore found")
                return False
        except Exception as e:
            logger.error(f"Error reloading vectorstore: {e}")
            return False

def main():
    """Test the vector store with pattern documents."""
    try:
        # Build the vector store from compatibility_analysis.json using VectorStore
        store = VectorStore()
        print("\nVector store built. Sample pattern documents:")
        for doc in store.documents[:3]:
            print(doc.page_content)
        
        # Test a sample query
        sample_query = "Upgrade Oracle from 12.1.0.2.0 to 19.3.0.0.0 in Production environment"
        print(f"\nSample query: {sample_query}")
        results = store.vectorstore.similarity_search(sample_query, k=5)
        print("\nTop 3 similar patterns:")
        for doc, score in results:
            print(f"Score: {score:.4f} | Pattern: {doc.page_content}")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 