import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibilityEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', model_path: str = None):
        """Initialize the embedder with a local model.
        
        Args:
            model_name: Name of the model (used if model_path is None)
            model_path: Local path to model (for offline usage)
        """
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading local model from: {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
    
    def load_chunks(self, chunks_dir: str = 'data/processed/chunks') -> List[Dict]:
        """Load all chunk files."""
        chunks_path = Path(chunks_dir)
        chunks = []
        
        for chunk_file in sorted(chunks_path.glob('chunk_*.json')):
            with open(chunk_file, 'r') as f:
                chunks.append(json.load(f))
        
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def prepare_text_for_embedding(self, chunk: Dict) -> str:
        """Convert chunk data into a text format suitable for embedding."""
        if chunk['type'] == 'metadata':
            return f"Metadata: {json.dumps(chunk['data'])}"
        
        elif chunk['type'] == 'server_chunk':
            server_texts = []
            for server in chunk['servers']:
                server_info = server['server_info']
                deployment_info = server['deployment_info']
                text = (
                    f"Server {server['name']} in {server['environment']} environment. "
                    f"Manufacturer: {server_info['manufacturer']}, "
                    f"Product Class: {server_info['product_class']}, "
                    f"Product Type: {server_info['product_type']}, "
                    f"Model: {server_info['model']}, "
                    f"Status: {server_info['status']}, "
                    f"Install Path: {deployment_info['install_path']}"
                )
                server_texts.append(text)
            return "\n".join(server_texts)
        
        elif chunk['type'] == 'environment_summary':
            return (
                f"Environment Summary for {chunk['environment']}: "
                f"Total Servers: {chunk['data']['total_servers']}, "
                f"Server Types: {json.dumps(chunk['data']['server_types'])}, "
                f"Status Distribution: {json.dumps(chunk['data']['status_distribution'])}"
            )
        
        elif chunk['type'] == 'manufacturer_summary':
            return (
                f"Manufacturer Summary for {chunk['manufacturer']}: "
                f"Total Servers: {chunk['data']['total_servers']}, "
                f"Environments: {json.dumps(chunk['data']['environments'])}, "
                f"Product Types: {json.dumps(chunk['data']['product_types'])}"
            )
        
        return ""
    
    def create_embeddings(self, chunks: List[Dict]) -> Dict[str, Dict]:
        """Create embeddings for all chunks."""
        logger.info("Creating embeddings...")
        
        embeddings = {}
        for i, chunk in enumerate(chunks):
            # Prepare text for embedding
            text = self.prepare_text_for_embedding(chunk)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Store embedding with chunk info
            chunk_id = f"chunk_{i:04d}"
            embeddings[chunk_id] = {
                'embedding': embedding,
                'type': chunk['type'],
                'metadata': {
                    'chunk_id': chunk_id,
                    'text': text[:200] + '...' if len(text) > 200 else text  # Store truncated text for reference
                }
            }
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
        
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, Dict], output_dir: str = 'data/processed/embeddings'):
        """Save embeddings to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = output_path / 'embeddings.npy'
        metadata_file = output_path / 'metadata.json'
        
        # Extract embeddings and metadata
        embedding_arrays = {k: v['embedding'] for k, v in embeddings.items()}
        metadata = {k: {**v['metadata'], 'type': v['type']} for k, v in embeddings.items()}
        
        # Save embeddings
        np.save(embeddings_file, embedding_arrays)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved embeddings to {embeddings_file}")
        logger.info(f"Saved metadata to {metadata_file}")

def main():
    # Check for local model first
    local_model_path = './models/all-MiniLM-L6-v2'
    
    if os.path.exists(local_model_path):
        logger.info("Using local model for offline processing")
        embedder = CompatibilityEmbedder(model_path=local_model_path)
    else:
        logger.info("No local model found, attempting to download...")
        logger.info("If you're behind a firewall, please:")
        logger.info("1. Run: python download_model_offline.py")
        logger.info("2. Or manually download model to: ./models/all-MiniLM-L6-v2/")
        embedder = CompatibilityEmbedder()
    
    # Load chunks
    chunks = embedder.load_chunks()
    
    # Create embeddings
    embeddings = embedder.create_embeddings(chunks)
    
    # Save embeddings
    embedder.save_embeddings(embeddings)
    
    # Log summary
    chunk_types = {}
    for chunk_id, data in embeddings.items():
        chunk_type = data['type']
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    logger.info("Embedding creation complete")
    logger.info("Chunk type distribution:")
    for chunk_type, count in chunk_types.items():
        logger.info(f"{chunk_type}: {count} chunks")

if __name__ == "__main__":
    main() 