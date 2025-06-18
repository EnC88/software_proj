import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibilityRAG:
    def __init__(
        self,
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        # Use a tiny model for local CPU inference. For better quality, switch to a larger model or API.
        llm_model_name: str = 'sshleifer/tiny-gpt2',
        data_dir: str = 'data/processed',
        top_k: int = 3
    ):
        """Initialize the RAG pipeline."""
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.device = 'cpu'  # Force CPU for stability on Mac
        self.embedding_model.to(self.device)
        
        # Initialize LLM
        try:
            logger.info(f"Loading language model: {llm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float32,
                device_map=None
            ).to(self.device)
        except Exception as e:
            logger.error(f"Error loading language model: {str(e)}")
            logger.info("Falling back to embedding-only mode")
            self.llm = None
            self.tokenizer = None
        
        # Load FAISS index and metadata
        self.load_index()
    
    def load_index(self):
        """Load FAISS index and associated metadata."""
        logger.info("Loading FAISS index and metadata...")
        
        # Load index
        index_path = self.data_dir / 'faiss_index' / 'index.faiss'
        self.index = faiss.read_index(str(index_path))
        
        # Load ID to chunk mapping
        mapping_path = self.data_dir / 'faiss_index' / 'id_to_chunk.json'
        with open(mapping_path, 'r') as f:
            self.id_to_chunk = json.load(f)
        
        # Load chunk metadata
        metadata_path = self.data_dir / 'embeddings' / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load chunks
        self.chunks = {}
        chunks_dir = self.data_dir / 'chunks'
        for chunk_file in chunks_dir.glob('chunk_*.json'):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
                chunk_id = chunk_file.stem
                self.chunks[chunk_id] = chunk_data
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
    
    def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks using FAISS similarity search."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        # Get relevant chunks
        relevant_chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk_id = self.id_to_chunk[str(idx)]
            chunk_data = self.chunks[chunk_id]
            chunk_metadata = self.metadata[chunk_id]
            
            relevant_chunks.append({
                'chunk_id': chunk_id,
                'type': chunk_data['type'],
                'content': chunk_data,
                'metadata': chunk_metadata,
                'similarity_score': float(1 / (1 + distance))  # Convert distance to similarity score
            })
        
        return relevant_chunks
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context for the LLM."""
        context_parts = []
        
        for chunk in chunks:
            chunk_type = chunk['type']
            content = chunk['content']
            
            if chunk_type == 'metadata':
                context_parts.append(f"System Information:\n{json.dumps(content['data'], indent=2)}")
            
            elif chunk_type == 'server_chunk':
                server_info = []
                for server in content['servers']:
                    server_info.append(
                        f"Server: {server['name']}\n"
                        f"Environment: {server['environment']}\n"
                        f"Manufacturer: {server['server_info']['manufacturer']}\n"
                        f"Product Class: {server['server_info']['product_class']}\n"
                        f"Product Type: {server['server_info']['product_type']}\n"
                        f"Model: {server['server_info']['model']}\n"
                        f"Status: {server['server_info']['status']}\n"
                        f"Install Path: {server['deployment_info']['install_path']}\n"
                    )
                context_parts.append("Server Information:\n" + "\n".join(server_info))
            
            elif chunk_type == 'environment_summary':
                context_parts.append(
                    f"Environment Summary for {content['environment']}:\n"
                    f"Total Servers: {content['data']['total_servers']}\n"
                    f"Server Types: {json.dumps(content['data']['server_types'], indent=2)}\n"
                    f"Status Distribution: {json.dumps(content['data']['status_distribution'], indent=2)}"
                )
            
            elif chunk_type == 'manufacturer_summary':
                context_parts.append(
                    f"Manufacturer Summary for {content['manufacturer']}:\n"
                    f"Total Servers: {content['data']['total_servers']}\n"
                    f"Environments: {json.dumps(content['data']['environments'], indent=2)}\n"
                    f"Product Types: {json.dumps(content['data']['product_types'], indent=2)}"
                )
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using the language model."""
        if self.llm is None:
            # If LLM is not available, return a summary of the context
            return f"Based on the retrieved information:\n\n{context}"
        
        # Create prompt
        prompt = f"You are a software compatibility expert. Use the following context to answer the question.\nIf you cannot answer based on the context, say so.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response part (after the prompt)
        response = response[len(prompt):].strip()
        
        return response
    
    def query(self, query: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query)
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Format context
        context = self.format_context(relevant_chunks)
        
        # Generate response
        response = self.generate_response(query, context)
        
        return {
            'query': query,
            'response': response,
            'relevant_chunks': relevant_chunks,
            'context': context
        }

def main():
    # Initialize RAG pipeline
    rag = CompatibilityRAG()
    
    # Example queries
    example_queries = [
        "What servers are compatible with Windows Server 2019?",
        "Which environments have the most Dell servers?",
        "What is the status of servers in the Production environment?",
        "Which manufacturer has the most diverse product types?"
    ]
    
    # Process each query
    for query in example_queries:
        print(f"\nQuery: {query}")
        result = rag.query(query)
        print(f"Response: {result['response']}")
        print("\nRelevant chunks:")
        for chunk in result['relevant_chunks']:
            print(f"- {chunk['type']} (similarity: {chunk['similarity_score']:.3f})")

if __name__ == "__main__":
    main() 