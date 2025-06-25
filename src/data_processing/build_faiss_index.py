#!/usr/bin/env python3
"""
Build FAISS index from simple TF-IDF embeddings for RAG pipeline
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path('data/processed/embeddings/embeddings.npy')
METADATA_PATH = Path('data/processed/embeddings/metadata.json')
INDEX_DIR = Path('data/processed/faiss_index')
INDEX_PATH = INDEX_DIR / 'index.faiss'
ID_TO_CHUNK_PATH = INDEX_DIR / 'id_to_chunk.json'


def load_embeddings():
    """Load embeddings from the simple TF-IDF embedder output."""
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    
    logger.info(f"Loading embeddings from {EMBEDDINGS_PATH}")
    embedding_dict = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()
    chunk_ids = list(embedding_dict.keys())
    embeddings = np.stack([embedding_dict[k] for k in chunk_ids])
    logger.info(f"Loaded {len(chunk_ids)} embeddings with shape {embeddings.shape}")
    return embeddings, chunk_ids

def load_metadata():
    """Load metadata for additional context."""
    if not METADATA_PATH.exists():
        logger.warning(f"Metadata file not found: {METADATA_PATH}")
        return {}
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    logger.info(f"Loaded metadata for {len(metadata)} chunks")
    return metadata

def build_faiss_index(embeddings):
    """Build FAISS index for similarity search."""
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index with dimension {dim}")
    
    # Use IndexFlatL2 for exact L2 distance search
    # For larger datasets, consider IndexIVFFlat or IndexHNSW
    index = faiss.IndexFlatL2(dim)
    
    # Add embeddings to index
    index.add(embeddings.astype(np.float32))
    logger.info(f"FAISS index built with {index.ntotal} vectors")
    return index

def save_index(index, chunk_ids, metadata=None):
    """Save FAISS index and mapping files."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    logger.info(f"Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, str(INDEX_PATH))
    
    # Save mapping from FAISS index to chunk_id
    id_to_chunk = {str(i): chunk_id for i, chunk_id in enumerate(chunk_ids)}
    with open(ID_TO_CHUNK_PATH, 'w') as f:
        json.dump(id_to_chunk, f, indent=2)
    logger.info(f"Saved id_to_chunk mapping to {ID_TO_CHUNK_PATH}")
    
    # Save metadata if available
    if metadata:
        metadata_path = INDEX_DIR / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    # Save index info
    index_info = {
        'total_vectors': index.ntotal,
        'dimension': index.d,
        'index_type': 'IndexFlatL2',
        'embedding_model': 'simple-tfidf',
        'created_at': str(Path().cwd())
    }
    info_path = INDEX_DIR / 'index_info.json'
    with open(info_path, 'w') as f:
        json.dump(index_info, f, indent=2)
    logger.info(f"Saved index info to {info_path}")

def build_and_save_faiss_index(
    embeddings_path=EMBEDDINGS_PATH,
    metadata_path=METADATA_PATH,
    index_dir=INDEX_DIR,
    index_path=INDEX_PATH,
    id_to_chunk_path=ID_TO_CHUNK_PATH
):
    """Build and save a FAISS index from embeddings and metadata."""
    # Check if embeddings exist
    if not embeddings_path.exists():
        logger.error(f"Embeddings not found at {embeddings_path}")
        logger.info("Please run the embedder first to create embeddings.")
        return False

    # Load embeddings and metadata
    embeddings, chunk_ids = load_embeddings()
    metadata = load_metadata()

    # Build and save index
    index = build_faiss_index(embeddings)
    save_index(index, chunk_ids, metadata)

    logger.info("✅ FAISS index creation complete!")
    logger.info(f"📁 Index saved to: {index_dir}")
    logger.info(f"🔍 Ready for similarity search with {len(chunk_ids)} chunks")
    return True

def main():
    """Main function to build FAISS index."""
    try:
        build_and_save_faiss_index()
    except Exception as e:
        logger.error(f"❌ Error building FAISS index: {str(e)}")
        logger.info("💡 Make sure embeddings exist by running the embedder first")

if __name__ == "__main__":
    main() 