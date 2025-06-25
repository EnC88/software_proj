#!/usr/bin/env python3
"""
Feedback Loop Script
Retrains the vectorizer/embeddings and rebuilds the index based on user feedback.
"""

import logging
from pathlib import Path
from src.evaluation.feedback_system import FeedbackLogger
from src.data_processing.hybrid_embedder import CompatibilityEmbedder
from src.data_processing.build_faiss_index import build_and_save_faiss_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_feedback_loop():
    logger.info("Starting feedback loop...")
    # 1. Load feedback
    feedback_logger = FeedbackLogger()
    all_feedback = feedback_logger.get_all_feedback()
    logger.info(f"Loaded {len(all_feedback)} feedback entries.")

    # 2. Find queries with negative feedback
    negative_feedback = [f for f in all_feedback if f['feedback_score'] == 0]
    logger.info(f"Found {len(negative_feedback)} negative feedback entries.")

    # 3. Collect all queries for retraining (optionally prioritize negatives)
    all_queries = [f['query'] for f in all_feedback]
    # Optionally, you could upsample negative queries here

    # 4. Retrain vectorizer/embeddings
    logger.info("Retraining hybrid embedder on all feedback queries...")
    embedder = CompatibilityEmbedder()
    embedder.embedder.encode(all_queries)  # Fit on all queries
    embedder.embedder.save(REPO_ROOT / 'data' / 'embeddings' / 'hybrid_model')
    logger.info("Hybrid embedder retrained and saved.")

    # 5. Rebuild the FAISS index
    logger.info("Rebuilding FAISS index...")
    build_and_save_faiss_index()
    logger.info("FAISS index rebuilt.")

    # 6. Log/report improvements
    logger.info(f"Feedback loop complete. Model retrained with {len(all_queries)} queries, including {len(negative_feedback)} negatives.")
    return {
        'total_feedback': len(all_feedback),
        'negative_feedback': len(negative_feedback),
        'status': 'success'
    }


def main():
    result = run_feedback_loop()
    print("\n=== Feedback Loop Summary ===")
    print(f"Total feedback entries: {result['total_feedback']}")
    print(f"Negative feedback entries: {result['negative_feedback']}")
    print(f"Status: {result['status']}")

if __name__ == "__main__":
    main() 