#!/usr/bin/env python3
"""
Build FAISS index from spaCy embeddings for RAG pipeline
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import faiss
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path('data/processed/embeddings/embeddings.npy')
METADATA_PATH = Path('data/processed/embeddings/metadata.json')
INDEX_DIR = Path('data/processed/faiss_index')
INDEX_PATH = INDEX_DIR / 'index.faiss'
ID_TO_CHUNK_PATH = INDEX_DIR / 'id_to_chunk.json'


def load_embeddings():
    logger.info(f"Loading embeddings from {EMBEDDINGS_PATH}")
    embedding_dict = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()
    chunk_ids = list(embedding_dict.keys())
    embeddings = np.stack([embedding_dict[k] for k in chunk_ids])
    logger.info(f"Loaded {len(chunk_ids)} embeddings with shape {embeddings.shape}")
    return embeddings, chunk_ids

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index with dimension {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    logger.info(f"FAISS index built with {index.ntotal} vectors")
    return index

def save_index(index, chunk_ids):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, str(INDEX_PATH))
    # Save mapping from FAISS index to chunk_id
    id_to_chunk = {str(i): chunk_id for i, chunk_id in enumerate(chunk_ids)}
    with open(ID_TO_CHUNK_PATH, 'w') as f:
        json.dump(id_to_chunk, f, indent=2)
    logger.info(f"Saved id_to_chunk mapping to {ID_TO_CHUNK_PATH}")

def main():
    try:
        embeddings, chunk_ids = load_embeddings()
        index = build_faiss_index(embeddings)
        save_index(index, chunk_ids)
        logger.info("FAISS index creation complete!")
    except Exception as e:
        logger.error(f"Error building FAISS index: {str(e)}")

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    main() 