#!/usr/bin/env python3
"""
Example Usage of Vectorizer with Sentence Transformers
Shows both online and offline usage options
"""

import pandas as pd
import logging
from src.vectorizer import Vectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Example usage of the vectorizer."""
    
    # Create sample data
    sample_data = {
        'CHUNK_TEXT': [
            'Upgraded Apache from version 2.4.1 to 2.4.2 successfully in production environment',
            'Failed to upgrade MySQL from 5.7 to 8.0 due to compatibility issues',
            'Successfully patched WebSphere from 9.0.0 to 9.0.1 with minimal downtime',
            'Rolled back Tomcat from 9.0.1 to 8.5.0 after deployment issues',
            'Upgraded PostgreSQL from 11.5 to 12.3 in development environment'
        ],
        'OBJECTNAME': ['Apache', 'MySQL', 'WebSphere', 'Tomcat', 'PostgreSQL'],
        'OLDVALUE': ['2.4.1', '5.7', '9.0.0', '9.0.1', '11.5'],
        'NEWVALUE': ['2.4.2', '8.0', '9.0.1', '8.5.0', '12.3']
    }
    
    df = pd.DataFrame(sample_data)
    
    logger.info("🚀 Example: Using Sentence Transformers")
    
    # Option 1: Online usage (default - will cache locally)
    logger.info("\n" + "="*60)
    logger.info("🌐 ONLINE USAGE (with local caching)")
    logger.info("="*60)
    
    vectorizer_online = Vectorizer(
        use_database=False,  # Disable database for this example
        use_cache=True,
        model_name='all-MiniLM-L6-v2'  # Will download and cache locally
    )
    
    # Set data and vectorize
    vectorizer_online.chunked_df = df
    vectorizer_online.vectorize()
    
    # Get model info
    model_info = vectorizer_online.get_model_info()
    logger.info(f"Model Info: {model_info}")
    
    # Test queries
    test_queries = [
        "How to upgrade Apache?",
        "What are MySQL upgrade issues?",
        "WebSphere patch procedures"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = vectorizer_online.query_upgrades(query, top_k=2)
        
        if results:
            logger.info(f"Found {len(results)} similar upgrades:")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. {result['object_name']} ({result['similarity']:.3f})")
                logger.info(f"     Version: {result['version_info']['old_version']} → {result['version_info']['new_version']}")
        else:
            logger.info("No similar upgrades found")
    
    # Option 2: Offline usage (if you have a local model)
    logger.info("\n" + "="*60)
    logger.info("🔒 OFFLINE USAGE (with local model)")
    logger.info("="*60)
    
    # Check if local model exists
    import os
    local_model_path = './models/all-MiniLM-L6-v2'
    
    if os.path.exists(local_model_path):
        logger.info(f"Using local model: {local_model_path}")
        
        vectorizer_offline = Vectorizer(
            use_database=False,
            use_cache=True,
            model_name='all-MiniLM-L6-v2',
            model_path=local_model_path  # Use local model path
        )
        
        vectorizer_offline.chunked_df = df
        vectorizer_offline.vectorize()
        
        # Test offline query
        results = vectorizer_offline.query_upgrades("How to upgrade Apache?", top_k=2)
        logger.info(f"Offline query found {len(results)} results")
        
    else:
        logger.info("No local model found. To use offline:")
        logger.info("1. Run: python download_model_offline.py")
        logger.info("2. Or run: python offline_alternatives.py")
        logger.info("3. Then use model_path parameter")
    
    # Performance benchmark
    logger.info("\n" + "="*60)
    logger.info("📊 Performance Benchmark")
    logger.info("="*60)
    
    benchmark_results = vectorizer_online.benchmark_model()
    
    logger.info("\n💡 Benefits of Sentence Transformers:")
    logger.info("✅ Fast processing (10-20x faster than alternatives)")
    logger.info("✅ Small model size (~90MB)")
    logger.info("✅ Memory efficient for large datasets")
    logger.info("✅ Good multilingual support")
    logger.info("✅ Easy deployment and maintenance")
    logger.info("✅ Works offline with local models")

if __name__ == "__main__":
    main() 