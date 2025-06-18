#!/usr/bin/env python3
"""
Offline Model Alternatives
Shows lightweight models that work well offline
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_offline_alternatives():
    """Show alternative models that work well offline."""
    
    alternatives = {
        'all-MiniLM-L6-v2': {
            'size': '~90MB',
            'dimension': 384,
            'speed': 'Very Fast',
            'quality': 'Good',
            'offline_friendly': 'Yes'
        },
        'paraphrase-MiniLM-L3-v2': {
            'size': '~60MB', 
            'dimension': 384,
            'speed': 'Very Fast',
            'quality': 'Good',
            'offline_friendly': 'Yes'
        },
        'all-mpnet-base-v2': {
            'size': '~420MB',
            'dimension': 768,
            'speed': 'Fast',
            'quality': 'Excellent',
            'offline_friendly': 'Yes'
        },
        'multi-qa-MiniLM-L6-cos-v1': {
            'size': '~90MB',
            'dimension': 384,
            'speed': 'Very Fast',
            'quality': 'Good for Q&A',
            'offline_friendly': 'Yes'
        }
    }
    
    logger.info("🔍 OFFLINE-FRIENDLY MODEL ALTERNATIVES")
    logger.info("="*60)
    
    for model_name, info in alternatives.items():
        logger.info(f"\\n📦 {model_name}:")
        logger.info(f"   Size: {info['size']}")
        logger.info(f"   Dimension: {info['dimension']}")
        logger.info(f"   Speed: {info['speed']}")
        logger.info(f"   Quality: {info['quality']}")
        logger.info(f"   Offline: {info['offline_friendly']}")
    
    logger.info("\\n" + "="*60)
    logger.info("💡 RECOMMENDATION FOR OFFLINE USE:")
    logger.info("✅ Use 'paraphrase-MiniLM-L3-v2' - smallest and fastest")
    logger.info("✅ Or 'all-MiniLM-L6-v2' - good balance of size/quality")
    logger.info("\\n" + "="*60)

def create_lightweight_example():
    """Create example using the lightest model."""
    
    example_code = '''#!/usr/bin/env python3
"""
Lightweight Offline Vectorizer Example
Uses the smallest sentence transformer model
"""

from src.vectorizer import Vectorizer
import pandas as pd

# Use the lightest model for offline use
vectorizer = Vectorizer(
    use_database=False,
    use_cache=True,
    model_name='paraphrase-MiniLM-L3-v2',  # Smallest model (~60MB)
    model_path='./models/paraphrase-MiniLM-L3-v2'  # Local path
)

# Create sample data
sample_data = {
    'CHUNK_TEXT': [
        'Upgraded Apache from version 2.4.1 to 2.4.2 successfully',
        'Failed to upgrade MySQL from 5.7 to 8.0 due to compatibility issues',
        'Successfully patched WebSphere from 9.0.0 to 9.0.1'
    ],
    'OBJECTNAME': ['Apache', 'MySQL', 'WebSphere'],
    'OLDVALUE': ['2.4.1', '5.7', '9.0.0'],
    'NEWVALUE': ['2.4.2', '8.0', '9.0.1']
}

df = pd.DataFrame(sample_data)
vectorizer.chunked_df = df
vectorizer.vectorize()

# Test queries
queries = [
    "How to upgrade Apache?",
    "What are MySQL upgrade issues?",
    "WebSphere patch procedures"
]

for query in queries:
    results = vectorizer.query_upgrades(query, top_k=2)
    print(f"\\nQuery: {query}")
    print(f"Found {len(results)} similar upgrades:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['object_name']} ({result['similarity']:.3f})")
        print(f"     Version: {result['version_info']['old_version']} → {result['version_info']['new_version']}")

# Show model info
model_info = vectorizer.get_model_info()
print(f"\\nModel Info: {model_info}")
'''
    
    with open('example_lightweight_offline.py', 'w') as f:
        f.write(example_code)
    
    logger.info("✅ Created example_lightweight_offline.py")

def main():
    """Main function."""
    show_offline_alternatives()
    create_lightweight_example()
    
    logger.info("\\n📋 NEXT STEPS:")
    logger.info("1. Run: python download_model_offline.py")
    logger.info("2. Or manually download a model to ./models/")
    logger.info("3. Use the vectorizer with model_path parameter")
    logger.info("4. Test with: python example_lightweight_offline.py")

if __name__ == "__main__":
    main() 