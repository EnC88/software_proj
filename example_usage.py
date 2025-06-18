#!/usr/bin/env python3
"""
Example Usage of Vectorizer with Sentence Transformers
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
    
    # Initialize vectorizer
    vectorizer = Vectorizer(
        use_database=False,  # Disable database for this example
        use_cache=True,
        model_name='all-MiniLM-L6-v2'  # Fast and efficient
    )
    
    # Set data and vectorize
    vectorizer.chunked_df = df
    vectorizer.vectorize()
    
    # Get model info
    model_info = vectorizer.get_model_info()
    logger.info(f"Model Info: {model_info}")
    
    # Test queries
    test_queries = [
        "How to upgrade Apache?",
        "What are MySQL upgrade issues?",
        "WebSphere patch procedures",
        "Tomcat rollback process"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = vectorizer.query_upgrades(query, top_k=3)
        
        if results:
            logger.info(f"Found {len(results)} similar upgrades:")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. {result['object_name']} ({result['similarity']:.3f})")
                logger.info(f"     Version: {result['version_info']['old_version']} → {result['version_info']['new_version']}")
        else:
            logger.info("No similar upgrades found")
    
    # Performance benchmark
    logger.info("\n" + "="*60)
    logger.info("📊 Performance Benchmark")
    logger.info("="*60)
    
    benchmark_results = vectorizer.benchmark_model()
    
    logger.info("\n💡 Benefits of Sentence Transformers:")
    logger.info("✅ Fast processing (10-20x faster than alternatives)")
    logger.info("✅ Small model size (~90MB)")
    logger.info("✅ Memory efficient for large datasets")
    logger.info("✅ Good multilingual support")
    logger.info("✅ Easy deployment and maintenance")

if __name__ == "__main__":
    main() 