import pandas as pd
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorizer import Vectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create sample data
    sample_data = {
        'LOG_ID': ['log1', 'log2', 'log3'],
        'CHUNK_TEXT': [
            'Upgraded Apache from version 2.4.1 to 2.4.2 successfully',
            'Failed to upgrade MySQL from 5.7 to 8.0 due to compatibility issues',
            'Successfully patched WebSphere from 9.0.0 to 9.0.1'
        ],
        'OBJECT_NAME': ['WEBSERVER', 'DATABASEINSTANCE', 'APPSERVER'],
        'CHANGE_TYPE': ['minor_upgrade', 'major_upgrade', 'patch_update']
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Initialize vectorizer
    vectorizer = Vectorizer(use_database=True)
    vectorizer.chunked_df = df
    
    try:
        # Generate vectors
        logger.info("Generating vectors...")
        vectorizer.vectorize()
        
        # Test queries
        test_queries = [
            "How to upgrade Apache?",
            "What are the issues with MySQL 8.0 upgrade?",
            "Tell me about WebSphere patches"
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            # Get similar upgrades
            results = vectorizer.query_upgrades(query)
            
            # Generate answer
            answer = vectorizer.generate_answer(query, results)
            logger.info(f"Answer:\n{answer}")
        
        # Get statistics
        stats = vectorizer.get_statistics()
        logger.info("\nDatabase Statistics:")
        logger.info(f"Total Records: {stats['total_records']}")
        logger.info("Change Type Distribution:")
        for change_type, count in stats['change_type_distribution'].items():
            logger.info(f"  {change_type}: {count}")
        logger.info("Object Name Distribution:")
        for object_name, count in stats['object_name_distribution'].items():
            logger.info(f"  {object_name}: {count}")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    main() 