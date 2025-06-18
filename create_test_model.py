#!/usr/bin/env python3
"""
Create a minimal test embedding model for offline testing
This creates a simple model that can be used to test the embedding pipeline
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

class MinimalEmbeddingModel(nn.Module):
    """A minimal embedding model for testing without internet."""
    
    def __init__(self, embedding_dim=384):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = 1000
        self.max_length = 512
        
        # Simple embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(embedding_dim)
        
        # Initialize weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
    def encode(self, texts, convert_to_numpy=True, **kwargs):
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Simple tokenization (just use character codes)
            tokens = [ord(c) % self.vocab_size for c in text[:self.max_length]]
            if len(tokens) < self.max_length:
                tokens.extend([0] * (self.max_length - len(tokens)))
            
            # Convert to tensor
            token_tensor = torch.tensor(tokens[:self.max_length], dtype=torch.long)
            
            # Get embeddings
            with torch.no_grad():
                embedded = self.embedding(token_tensor)
                pooled = self.pooling(embedded.unsqueeze(0).transpose(1, 2))
                embedding = pooled.squeeze().transpose(0, 1)
                
                if convert_to_numpy:
                    embedding = embedding.numpy()
                
            embeddings.append(embedding)
        
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings
    
    def get_sentence_embedding_dimension(self):
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def save(self, path):
        """Save the model to a directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        
        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'model_type': 'minimal_test_model'
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save tokenizer info
        tokenizer_config = {
            'model_type': 'minimal_test_model',
            'vocab_size': self.vocab_size
        }
        with open(os.path.join(path, 'tokenizer_config.json'), 'w') as f:
            json.dump(tokenizer_config, f, indent=2)

def create_test_model():
    """Create a minimal test model for offline use."""
    
    print("🔧 Creating minimal test embedding model...")
    
    # Create models directory
    model_path = './models/all-MiniLM-L6-v2'
    os.makedirs(model_path, exist_ok=True)
    
    # Create and save model
    model = MinimalEmbeddingModel(embedding_dim=384)  # Same dimension as all-MiniLM-L6-v2
    model.save(model_path)
    
    # Test the model
    print("🧪 Testing the model...")
    test_texts = [
        "This is a test sentence.",
        "Another test sentence for validation."
    ]
    
    embeddings = model.encode(test_texts)
    print(f"✅ Model created successfully!")
    print(f"📁 Location: {model_path}")
    print(f"📊 Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"🧪 Test embeddings shape: {embeddings[0].shape}")
    
    # Test with sentence_transformers interface
    print("\n🔍 Testing compatibility with sentence_transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        test_model = SentenceTransformer(model_path)
        test_embedding = test_model.encode("Test sentence")
        print(f"✅ Compatible with sentence_transformers!")
        print(f"📊 Test embedding shape: {test_embedding.shape}")
    except Exception as e:
        print(f"⚠️  Note: This is a minimal test model, not fully compatible with sentence_transformers")
        print(f"   Error: {str(e)}")
    
    print("\n🎉 Test model ready for offline use!")
    print("Note: This is a minimal model for testing only.")
    print("For production use, download the real all-MiniLM-L6-v2 model.")

if __name__ == "__main__":
    create_test_model() 