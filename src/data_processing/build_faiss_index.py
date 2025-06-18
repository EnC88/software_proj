import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSIndexBuilder:
    def __init__(self, embeddings_dir: str = 'data/processed/embeddings'):
        """Initialize the FAISS index builder."""
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings = None
        self.metadata = None
        self.index = None
        self.id_to_chunk = {}  # Maps FAISS IDs to chunk metadata
    
    def load_embeddings(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load embeddings and metadata from files."""
        logger.info("Loading embeddings and metadata...")
        
        # Load embeddings
        embeddings_file = self.embeddings_dir / 'embeddings.npy'
        self.embeddings = np.load(embeddings_file, allow_pickle=True).item()
        
        # Load metadata
        metadata_file = self.embeddings_dir / 'metadata.json'
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Convert embeddings to numpy array
        chunk_ids = list(self.embeddings.keys())
        embedding_vectors = np.array([self.embeddings[chunk_id] for chunk_id in chunk_ids])
        
        # Create ID to chunk mapping
        self.id_to_chunk = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        
        logger.info(f"Loaded {len(chunk_ids)} embeddings")
        return embedding_vectors, self.metadata
    
    def build_index(self, embedding_vectors: np.ndarray) -> faiss.Index:
        """Build FAISS index from embedding vectors."""
        logger.info("Building FAISS index...")
        
        # Get embedding dimension
        dimension = embedding_vectors.shape[1]
        
        # Create FAISS index
        # Using L2 (Euclidean) distance for similarity search
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        index.add(embedding_vectors)
        
        logger.info(f"Built index with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, output_dir: str = 'data/processed/faiss_index'):
        """Save FAISS index and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_file = output_path / 'index.faiss'
        faiss.write_index(index, str(index_file))
        
        # Save ID to chunk mapping
        mapping_file = output_path / 'id_to_chunk.json'
        with open(mapping_file, 'w') as f:
            json.dump(self.id_to_chunk, f, indent=2)
        
        logger.info(f"Saved index to {index_file}")
        logger.info(f"Saved ID mapping to {mapping_file}")
    
    def process(self):
        """Process embeddings and build FAISS index."""
        # Load embeddings
        embedding_vectors, metadata = self.load_embeddings()
        
        # Build index
        index = self.build_index(embedding_vectors)
        
        # Save index
        self.save_index(index)
        
        # Log statistics
        logger.info("\nIndex Statistics:")
        logger.info(f"Total vectors: {index.ntotal}")
        logger.info(f"Vector dimension: {index.d}")
        logger.info(f"Index type: {type(index).__name__}")
        
        # Log chunk type distribution
        chunk_types = {}
        for chunk_id in self.id_to_chunk.values():
            chunk_type = self.metadata[chunk_id]['type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        logger.info("\nChunk type distribution:")
        for chunk_type, count in chunk_types.items():
            logger.info(f"{chunk_type}: {count} chunks")

def main():
    # Initialize builder
    builder = FAISSIndexBuilder()
    
    # Process embeddings and build index
    builder.process()

if __name__ == "__main__":
    main() 