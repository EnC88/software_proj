#!/usr/bin/env python3
"""
Quick Model Download Script
Downloads the model quickly for offline use
"""

import os
import sys
from pathlib import Path
import sentence_transformers

def quick_download():
    """Quickly download the model for offline use."""
    
    model_name = 'all-MiniLM-L6-v2'
    output_dir = './models'
    
    print(f"🚀 Downloading {model_name} for offline use...")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Download and save model
        print("Downloading model...")
        model = sentence_transformers.SentenceTransformer(model_name)
        
        model_path = os.path.join(output_dir, model_name)
        print(f"Saving to: {model_path}")
        model.save(model_path)
        
        print("✅ Model downloaded successfully!")
        print(f"📁 Location: {model_path}")
        print(f"📊 Size: ~90MB")
        
        # Test loading
        print("Testing model loading...")
        test_model = sentence_transformers.SentenceTransformer(model_path)
        test_embedding = test_model.encode("Test sentence")
        print(f"✅ Test successful! Embedding dimension: {len(test_embedding)}")
        
        print("\n🎉 You can now use the model offline!")
        print("Example usage:")
        print("  from src.vectorizer import Vectorizer")
        print(f"  vectorizer = Vectorizer(model_path='{model_path}')")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 If you're behind a firewall:")
        print("1. Try using a different network")
        print("2. Or manually download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        return False
    
    return True

if __name__ == "__main__":
    success = quick_download()
    if success:
        print("\n✅ Ready for offline use!")
    else:
        print("\n❌ Download failed. Check your internet connection.") 