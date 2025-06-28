#!/usr/bin/env python3
"""
Vector Store for RAG Pipeline
Industry-standard vector store implementation using LangChain
Works completely offline with free local models only
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
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Free local LLM options
try:
    from langchain.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain.llms import CTransformers
    CT_TRANSFORMERS_AVAILABLE = True
except ImportError:
    CT_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define repo root for robust file access
REPO_ROOT = Path(__file__).resolve().parents[2]

class SimpleVectorStore:
    """Simple vector store using TF-IDF for offline operation."""
    
    def __init__(self, documents: List[Document] = None):
        self.documents = documents or []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index from documents."""
        if not self.documents:
            return
        
        # Extract text content
        texts = [doc.page_content for doc in self.documents]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"Built TF-IDF index with {len(self.documents)} documents")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents using TF-IDF."""
        if not self.vectorizer or not self.tfidf_matrix:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                results.append(self.documents[idx])
        
        return results
    
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

class VectorStore:
    """Industry-standard vector store implementation using local components.
    
    Works completely offline with free local models only.
    No API keys or paid services required.
    """
    
    def __init__(self, 
                 data_dir: str = None,
                 use_local_llm: bool = True,
                 local_llm_model: str = "llama2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize the vector store.
        
        Args:
            data_dir: Directory containing source data
            use_local_llm: Whether to use a free local LLM
            local_llm_model: Local LLM model name (llama2, mistral, etc.)
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between text chunks
        """
        self.data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
        self.use_local_llm = use_local_llm
        self.local_llm_model = local_llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.vectorstore = None
        self.text_splitter = None
        self.llm = None
        self.qa_chain = None
        self.conversation_chain = None
        self.documents = []
        self.is_initialized = False
        
        # Initialize the system
        self._initialize_components()
        self._load_and_process_data()
    
    def _initialize_components(self):
        """Initialize components with improved error handling."""
        try:
            # Initialize text splitter with configurable parameters
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize local LLM (free options only)
            self._initialize_local_llm()
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_local_llm(self):
        """Initialize local LLM with fallback options."""
        if not self.use_local_llm:
            logger.info("Local LLM disabled - running with embeddings only")
            self.llm = None
            return
            
        # Try Ollama first
        if OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(model=self.local_llm_model)
                logger.info(f"Ollama LLM initialized with model: {self.local_llm_model}")
                return
            except Exception as e:
                logger.warning(f"Ollama not available: {e}")
        
        # Try CTransformers as fallback
        if CT_TRANSFORMERS_AVAILABLE:
            try:
                self.llm = CTransformers(
                    model="TheBloke/Llama-2-7B-Chat-GGML",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.1}
                )
                logger.info("CTransformers LLM initialized")
                return
            except Exception as e:
                logger.warning(f"CTransformers not available: {e}")
        
        # No LLM available
        logger.info("No local LLM available - running with embeddings only (completely free)")
        self.llm = None
    
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
        sor_path = self.data_dir / 'raw' / 'SOR_History.csv'
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
            self.vectorstore = SimpleVectorStore(self.documents)
            
            # Save vectorstore
            vectorstore_path = self.data_dir / 'processed' / 'vectorstore'
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(vectorstore_path))
            logger.info(f"Vectorstore saved to {vectorstore_path}")
            
            # Initialize QA chain if LLM is available
            if self.llm:
                self._initialize_qa_chain()
            
            self.is_initialized = True
            
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
                
        except Exception as e:
            logger.error(f"Error processing analysis data: {e}")
    
    def _create_server_document(self, server: Dict[str, Any]):
        """Create a document for server information."""
        server_info = server.get('server_info', {})
        model = server_info.get('model', 'Unknown')
        product_type = server_info.get('product_type', 'Unknown')
        environment = server.get('environment', 'Unknown')
        
        content = f"""
        Server Model: {model}
        Product Type: {product_type}
        Environment: {environment}
        Server ID: {server.get('id', 'Unknown')}
        
        Software Information:
        {json.dumps(server.get('software_info', {}), indent=2)}
        
        Compatibility Status: {server.get('compatibility_status', 'Unknown')}
        """
        
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
        content = f"""
        Compatibility Rule:
        Software: {rule.get('software', 'Unknown')}
        Version: {rule.get('version', 'Unknown')}
        OS Compatibility: {rule.get('os_compatibility', 'Unknown')}
        Dependencies: {rule.get('dependencies', [])}
        Conflicts: {rule.get('conflicts', [])}
        """
        
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
    
    def _process_sor_data(self, sor_df: pd.DataFrame):
        """Process SOR history data into documents."""
        try:
            for _, row in sor_df.iterrows():
                content = f"""
                SOR Change Request:
                Object Name: {row.get('OBJECTNAME', 'Unknown')}
                Attribute Name: {row.get('ATTRIBUTENAME', 'Unknown')}
                Old Value: {row.get('OLDVALUE', 'Unknown')}
                New Value: {row.get('NEWVALUE', 'Unknown')}
                Change Date: {row.get('CHANGEDATE', 'Unknown')}
                Catalog ID: {row.get('CATALOGID', 'Unknown')}
                """
                
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
                content = f"""
                Web Server Information:
                Server Name: {row.get('SERVERNAME', 'Unknown')}
                Model: {row.get('MODEL', 'Unknown')}
                Product Type: {row.get('PRODUCTTYPE', 'Unknown')}
                Environment: {row.get('ENVIRONMENT', 'Unknown')}
                OS: {row.get('OS', 'Unknown')}
                """
                
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
    
    def _initialize_qa_chain(self):
        """Initialize the QA chain for conversational responses."""
        try:
            # Create prompt template
            prompt_template = """You are a software compatibility expert. Use the following context to answer the user's question about software compatibility, upgrades, and infrastructure changes.

Context:
{context}

Question: {question}

Answer the question based on the context provided. If the context doesn't contain enough information, say so. Be specific about compatibility issues, affected servers, and recommendations.

Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create simple QA chain without LangChain's RetrievalQA
            # We'll implement a custom one that works with our SimpleVectorStore
            self.qa_chain = {
                'llm': self.llm,
                'prompt': prompt,
                'vectorstore': self.vectorstore
            }
            
            logger.info("QA chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing QA chain: {e}")
    
    def query(self, query_text: str, top_k: int = 5, use_llm: bool = False) -> Dict[str, Any]:
        """Query the vectorstore and optionally get LLM response.
        
        Args:
            query_text: The query text
            top_k: Number of top results to return
            use_llm: Whether to use LLM for response generation
            
        Returns:
            Dictionary containing results and optional LLM response
        """
        if not self.is_initialized or not self.vectorstore:
            logger.error("Vectorstore not initialized")
            return {"error": "Vectorstore not available"}
        
        try:
            # Get similar documents
            docs = self.vectorstore.similarity_search(query_text, k=top_k)
            
            # Format results
            results = []
            for i, doc in enumerate(docs):
                result = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': None  # TF-IDF doesn't return scores in our implementation
                }
                results.append(result)
            
            response = {
                'query': query_text,
                'results': results,
                'total_results': len(results)
            }
            
            # Add LLM response if requested and available
            if use_llm and self.qa_chain and self.llm:
                try:
                    # Create context from top results
                    context = "\n\n".join([doc.page_content for doc in docs[:3]])
                    
                    # Format prompt
                    formatted_prompt = self.qa_chain['prompt'].format(
                        context=context,
                        question=query_text
                    )
                    
                    # Get LLM response
                    llm_response = self.llm(formatted_prompt)
                    response['llm_response'] = llm_response
                except Exception as e:
                    logger.error(f"Error getting LLM response: {e}")
                    response['llm_error'] = str(e)
            
            return response
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {"error": str(e)}
    
    def conversational_query(self, query_text: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Perform a conversational query with memory."""
        if not self.llm:
            logger.error("LLM not available for conversational queries")
            return {"error": "Conversational features not available"}
        
        try:
            # Get relevant documents
            docs = self.vectorstore.similarity_search(query_text, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create conversation prompt
            conversation_prompt = f"""You are a software compatibility expert. Use the following context to answer the user's question.

Context:
{context}

Question: {query_text}

Answer:"""
            
            # Get response
            response = self.llm(conversation_prompt)
            
            return {
                'query': query_text,
                'response': response,
                'source_documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    } for doc in docs
                ],
                'chat_history': chat_history or []
            }
            
        except Exception as e:
            logger.error(f"Error during conversational query: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store."""
        stats = {
            'total_documents': len(self.documents),
            'use_local_llm': self.use_local_llm,
            'local_llm_model': self.local_llm_model,
            'vectorstore_available': self.vectorstore is not None,
            'qa_chain_available': self.qa_chain is not None,
            'is_initialized': self.is_initialized,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'cost': 'Completely Free (Local Models Only)',
            'privacy': '100% Local - No Data Sent to External Services',
            'embeddings': 'TF-IDF (Local, No External Dependencies)'
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
    """Test the vector store."""
    try:
        # Initialize the store (completely free)
        store = VectorStore(use_local_llm=True)
        
        # Test query
        query = "What servers are running Apache HTTPD?"
        results = store.query(query, top_k=3)
        
        print(f"Query: {query}")
        print(f"Results: {json.dumps(results, indent=2)}")
        
        # Test with LLM if available
        if store.llm:
            results_with_llm = store.query(query, top_k=3, use_llm=True)
            print(f"Results with LLM: {json.dumps(results_with_llm, indent=2)}")
        
        # Print stats
        print(f"Stats: {json.dumps(store.get_stats(), indent=2)}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 