#!/usr/bin/env python3
"""
Query Engine for RAG Pipeline
Loads FAISS index and hybrid embedder for similarity search
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryEngine:
    """Production-ready query engine for RAG pipeline using hybrid embedder."""
    
    def __init__(self, 
                 index_path: str = 'data/processed/faiss_index/index.faiss',
                 id_to_chunk_path: str = 'data/processed/faiss_index/id_to_chunk.json',
                 metadata_path: str = 'data/processed/embeddings/metadata.json',
                 chunks_dir: str = 'data/processed/chunks',
                 hybrid_model_path: str = 'data/processed/embeddings/hybrid_model'):
        """Initialize the query engine.
        
        Args:
            index_path: Path to FAISS index
            id_to_chunk_path: Path to id-to-chunk mapping
            metadata_path: Path to embeddings metadata
            chunks_dir: Directory containing chunk files
            hybrid_model_path: Path to hybrid embedder model
        """
        self.index_path = Path(index_path)
        self.id_to_chunk_path = Path(id_to_chunk_path)
        self.metadata_path = Path(metadata_path)
        self.chunks_dir = Path(chunks_dir)
        self.hybrid_model_path = Path(hybrid_model_path)
        
        # Initialize components
        self.embedder = None
        self.index = None
        self.id_to_chunk = {}
        self.metadata = {}
        self.chunks = {}
        
        self._load_components()
    
    def _load_components(self):
        """Load all required components."""
        try:
            # Load hybrid embedder
            logger.info(f"Loading hybrid embedder from {self.hybrid_model_path}")
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from data_processing.hybrid_embedder import HybridEmbedder
            self.embedder = HybridEmbedder()
            self.embedder.load(self.hybrid_model_path)
            logger.info("Hybrid embedder loaded successfully")
            
            # Load FAISS index
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS index loaded with {self.index.ntotal} vectors")
            
            # Load id-to-chunk mapping
            logger.info(f"Loading id-to-chunk mapping from {self.id_to_chunk_path}")
            with open(self.id_to_chunk_path, 'r') as f:
                self.id_to_chunk = json.load(f)
            logger.info(f"Loaded {len(self.id_to_chunk)} id-to-chunk mappings")
            
            # Load metadata
            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata)} chunks")
            
            # Load chunks
            logger.info(f"Loading chunks from {self.chunks_dir}")
            for chunk_file in sorted(self.chunks_dir.glob('chunk_*.json')):
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                    chunk_id = chunk_file.stem
                    self.chunks[chunk_id] = chunk_data
            logger.info(f"Loaded {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading components: {str(e)}")
            raise
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the index for similar chunks.
        
        Args:
            query_text: The query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing chunk info and similarity scores
        """
        try:
            # Embed the query using hybrid embedder
            query_embedding = self.embedder.encode([query_text], convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                    
                chunk_id = self.id_to_chunk[str(idx)]
                chunk_data = self.chunks.get(chunk_id, {})
                chunk_metadata = self.metadata.get(chunk_id, {})
                
                # Convert distance to similarity score (0-1, higher is better)
                similarity_score = 1 / (1 + distance)
                
                result = {
                    'chunk_id': chunk_id,
                    'similarity_score': float(similarity_score),
                    'distance': float(distance),
                    'chunk_type': chunk_data.get('type', 'unknown'),
                    'content': chunk_data,
                    'metadata': chunk_metadata
                }
                results.append(result)
            
            logger.info(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return []
    
    def format_results_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """Format query results for use with an LLM.
        
        Args:
            results: Query results from self.query()
            
        Returns:
            Formatted string for LLM context
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            chunk_type = result['chunk_type']
            content = result['content']
            similarity = result['similarity_score']
            
            context_parts.append(f"--- Result {i} (similarity: {similarity:.3f}) ---")
            
            if chunk_type == 'metadata':
                context_parts.append(f"System Information:\n{json.dumps(content.get('data', {}), indent=2)}")
            
            elif chunk_type == 'server_chunk':
                # Group servers by model and product_type
                model_env_map = {}
                for server in content.get('servers', []):
                    model = server.get('server_info', {}).get('model', 'Unknown')
                    product_type = server.get('server_info', {}).get('product_type', 'Unknown')
                    env = server.get('environment', 'Unknown')
                    key = f"{model} ({product_type})"
                    if key not in model_env_map:
                        model_env_map[key] = set()
                    model_env_map[key].add(env)
                model_lines = []
                for model, envs in model_env_map.items():
                    envs_str = ', '.join(sorted(envs))
                    model_lines.append(f"Model: {model} [Environments: {envs_str}]")
                context_parts.append("Software Models:\n" + "\n".join(model_lines))
            
            elif chunk_type == 'environment_summary':
                context_parts.append(
                    f"Environment Summary for {content.get('environment', 'Unknown')}:\n"
                    f"Total Servers: {content.get('data', {}).get('total_servers', 'Unknown')}\n"
                    f"Server Types: {json.dumps(content.get('data', {}).get('server_types', {}), indent=2)}\n"
                    f"Status Distribution: {json.dumps(content.get('data', {}).get('status_distribution', {}), indent=2)}"
                )
            
            elif chunk_type == 'manufacturer_summary':
                context_parts.append(
                    f"Manufacturer Summary for {content.get('manufacturer', 'Unknown')}:\n"
                    f"Total Servers: {content.get('data', {}).get('total_servers', 'Unknown')}\n"
                    f"Environments: {json.dumps(content.get('data', {}).get('environments', {}), indent=2)}\n"
                    f"Product Types: {json.dumps(content.get('data', {}).get('product_types', {}), indent=2)}"
                )
            
            context_parts.append("")  # Add spacing
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        return {
            'model_type': 'hybrid-nltk-sklearn',
            'index_size': self.index.ntotal if self.index else 0,
            'chunks_loaded': len(self.chunks),
            'metadata_entries': len(self.metadata),
            'embedding_dimension': self.embedder.get_sentence_embedding_dimension() if self.embedder else 300
        }


def main():
    """Test the query engine."""
    try:
        # Initialize query engine
        engine = QueryEngine()
        
        # Get stats
        stats = engine.get_stats()
        print("Query Engine Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test queries
        test_queries = [
            "What servers are compatible with Windows Server 2019?",
            "Which environments have the most Dell servers?",
            "What is the status of servers in the Production environment?",
            "Which manufacturer has the most diverse product types?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            results = engine.query(query, top_k=3)
            
            if results:
                print(f"Found {len(results)} relevant results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Chunk ID: {result['chunk_id']}")
                    print(f"   Type: {result['chunk_type']}")
                    print(f"   Similarity: {result['similarity_score']:.3f}")
                
                # Format for LLM
                llm_context = engine.format_results_for_llm(results)
                print(f"\nLLM Context (first 500 chars):")
                print(llm_context[:500] + "..." if len(llm_context) > 500 else llm_context)
            else:
                print("No results found.")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 