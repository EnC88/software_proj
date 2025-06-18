#!/usr/bin/env python3
"""
Offline Model Download Script
Downloads sentence transformer models locally for offline use
"""

import os
import sys
import logging
from pathlib import Path
import sentence_transformers
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model_offline(model_name: str = 'all-MiniLM-L6-v2', 
                          output_dir: str = './models') -> str:
    """
    Download a sentence transformer model for offline use.
    
    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model
    
    Returns:
        Path to the downloaded model
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_dir = output_path / model_name
        
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Output directory: {model_dir}")
        
        # Download the model
        model = sentence_transformers.SentenceTransformer(model_name)
        
        # Save the model locally
        model.save(str(model_dir))
        
        logger.info(f"✅ Model downloaded successfully to: {model_dir}")
        
        # Verify the model can be loaded
        logger.info("Verifying model can be loaded...")
        test_model = sentence_transformers.SentenceTransformer(str(model_dir))
        test_embedding = test_model.encode("Test sentence")
        logger.info(f"✅ Model verification successful. Embedding dimension: {len(test_embedding)}")
        
        return str(model_dir)
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {str(e)}")
        raise

def create_offline_usage_example(model_path: str):
    """Create an example showing how to use the offline model."""
    example_code = f'''#!/usr/bin/env python3
"""
Example: Using Vectorizer with Offline Model
"""

from src.vectorizer import Vectorizer
import pandas as pd

# Use the offline model
vectorizer = Vectorizer(
    use_database=False,
    use_cache=True,
    model_path="{model_path}"  # Use local model path
)

# Create sample data
sample_data = {{
    'CHUNK_TEXT': [
        'Upgraded Apache from version 2.4.1 to 2.4.2 successfully',
        'Failed to upgrade MySQL from 5.7 to 8.0 due to compatibility issues'
    ],
    'OBJECTNAME': ['Apache', 'MySQL'],
    'OLDVALUE': ['2.4.1', '5.7'],
    'NEWVALUE': ['2.4.2', '8.0']
}}

df = pd.DataFrame(sample_data)
vectorizer.chunked_df = df
vectorizer.vectorize()

# Test query
results = vectorizer.query_upgrades("How to upgrade Apache?")
print(f"Found {{len(results)}} similar upgrades")
'''
    
    with open('example_offline_usage.py', 'w') as f:
        f.write(example_code)
    
    logger.info("✅ Created example_offline_usage.py")

def main():
    """Main function to download model and create examples."""
    logger.info("🚀 Starting offline model download...")
    
    # Download the model
    model_path = download_model_offline()
    
    # Create usage example
    create_offline_usage_example(model_path)
    
    logger.info("\\n" + "="*60)
    logger.info("📋 OFFLINE USAGE INSTRUCTIONS")
    logger.info("="*60)
    logger.info("\\n1. Model downloaded to: models/all-MiniLM-L6-v2/")
    logger.info("\\n2. To use offline, initialize vectorizer with:")
    logger.info("   vectorizer = Vectorizer(model_path='./models/all-MiniLM-L6-v2')")
    logger.info("\\n3. Example created: example_offline_usage.py")
    logger.info("\\n4. You can now use the vectorizer without internet access!")
    logger.info("\\n" + "="*60)

if __name__ == "__main__":
    main() 