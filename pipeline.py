#!/usr/bin/env python3
"""
Main Pipeline Orchestration
Runs the complete RAG pipeline: chunking -> embedding -> indexing -> querying
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_step(step_name: str, script_path: str, description: str) -> bool:
    """Run a pipeline step with error handling.
    
    Args:
        step_name: Name of the step
        script_path: Path to the script to run
        description: Description of what the step does
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"🚀 Starting {step_name}: {description}")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        logger.info(f"✅ {step_name} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ {step_name} failed with error: {str(e)}")
        return False

def check_prerequisites() -> bool:
    """Check if all required files and dependencies exist."""
    logger.info("🔍 Checking prerequisites...")
    
    # Check required directories
    required_dirs = [
        'src/data_processing',
        'src/rag',
        'data/processed'
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            logger.error(f"❌ Required directory missing: {dir_path}")
            return False
    
    # Check required scripts
    required_scripts = [
        'src/data_processing/chunk_compatibility.py',
        'src/data_processing/hybrid_embedder.py',
        'src/data_processing/build_faiss_index.py',
        'src/rag/query_engine.py'
    ]
    
    for script_path in required_scripts:
        if not Path(script_path).exists():
            logger.error(f"❌ Required script missing: {script_path}")
            return False
    
    logger.info("✅ All prerequisites met")
    return True

def main():
    """Run the complete pipeline."""
    logger.info("🎯 Starting RAG Pipeline")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Define pipeline steps
    steps = [
        {
            'name': 'Chunking',
            'script': 'src/data_processing/chunk_compatibility.py',
            'description': 'Split raw data into meaningful chunks'
        },
        {
            'name': 'Embedding',
            'script': 'src/data_processing/hybrid_embedder.py',
            'description': 'Generate embeddings using hybrid NLTK + TF-IDF approach'
        },
        {
            'name': 'Indexing',
            'script': 'src/data_processing/build_faiss_index.py',
            'description': 'Build FAISS index for fast similarity search'
        }
    ]
    
    # Run each step
    for step in steps:
        success = run_step(step['name'], step['script'], step['description'])
        if not success:
            logger.error(f"❌ Pipeline failed at step: {step['name']}")
            logger.error("Please fix the error and run the pipeline again.")
            sys.exit(1)
    
    logger.info("🎉 Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Test the query engine: python src/rag/query_engine.py")
    logger.info("2. Use the QueryEngine class in your RAG application")
    logger.info("3. Check pipeline.log for detailed execution logs")

def run_query_test():
    """Run a quick test of the query engine."""
    logger.info("🧪 Testing query engine...")
    
    try:
        from src.rag.query_engine import QueryEngine
        
        # Initialize query engine
        engine = QueryEngine()
        
        # Test a simple query
        test_query = "What servers are in the Production environment?"
        results = engine.query(test_query, top_k=3)
        
        if results:
            logger.info(f"✅ Query test successful! Found {len(results)} results")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. {result['chunk_id']} (similarity: {result['similarity_score']:.3f})")
        else:
            logger.warning("⚠️ Query test returned no results")
            
    except Exception as e:
        logger.error(f"❌ Query test failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Pipeline Orchestration')
    parser.add_argument('--test-query', action='store_true', 
                       help='Run a quick query test after pipeline completion')
    
    args = parser.parse_args()
    
    # Run main pipeline
    main()
    
    # Optionally run query test
    if args.test_query:
        run_query_test() 