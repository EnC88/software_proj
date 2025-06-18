#!/usr/bin/env python3
"""
Hybrid NLTK + scikit-learn Embedder
Uses NLTK for text preprocessing and scikit-learn TF-IDF for vectorization
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridEmbedder:
    def __init__(self, max_features: int = 10000, embedding_dim: int = 300):
        """Initialize hybrid embedder.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            embedding_dim: Embedding dimension (will be padded/truncated)
        """
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set()
        self.vectorizer = None
        self.is_fitted = False
        
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup NLTK components with robust error handling."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            # Test if punkt_tab issue exists and handle it
            try:
                # Try to use word_tokenize to see if punkt_tab error occurs
                test_tokens = word_tokenize("test sentence")
                self.stop_words = set(stopwords.words('english'))
                logger.info("NLTK setup completed successfully")
            except LookupError as e:
                if "punkt_tab" in str(e):
                    logger.warning("NLTK punkt_tab bug detected, using fallback tokenization")
                    # Use a simple fallback tokenization
                    self._use_fallback_tokenization = True
                    self.stop_words = set(stopwords.words('english'))
                else:
                    raise e
                    
        except Exception as e:
            logger.warning(f"NLTK setup warning: {str(e)}")
            logger.info("Using fallback tokenization methods")
            self._use_fallback_tokenization = True
            # Create a basic stop words set
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their'
            }
    
    def _fallback_tokenize(self, text: str) -> List[str]:
        """Fallback tokenization when NLTK punkt_tab fails."""
        # Simple regex-based tokenization
        import re
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _get_wordnet_pos(self, treebank_tag):
        """Convert POS tag to WordNet POS tag."""
        tag_map = {
            'J': 'a',  # Adjective
            'V': 'v',  # Verb
            'N': 'n',  # Noun
            'R': 'r'   # Adverb
        }
        return tag_map.get(treebank_tag[0], 'n')  # Default to noun
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text using NLTK: tokenize, lemmatize, remove stop words."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize - use fallback if NLTK punkt_tab fails
        try:
            if hasattr(self, '_use_fallback_tokenization') and self._use_fallback_tokenization:
                tokens = self._fallback_tokenize(text)
                # Simple processing for fallback mode
                processed_tokens = []
                for token in tokens:
                    if token not in self.stop_words and len(token) > 2:
                        processed_tokens.append(token)
                return ' '.join(processed_tokens)
            else:
                tokens = word_tokenize(text)
        except LookupError as e:
            if "punkt_tab" in str(e):
                logger.warning("punkt_tab error in tokenization, using fallback")
                tokens = self._fallback_tokenize(text)
                # Simple processing for fallback mode
                processed_tokens = []
                for token in tokens:
                    if token not in self.stop_words and len(token) > 2:
                        processed_tokens.append(token)
                return ' '.join(processed_tokens)
            else:
                raise e
        
        # POS tagging for better lemmatization
        try:
            pos_tags = pos_tag(tokens)
        except LookupError as e:
            logger.warning(f"POS tagging failed: {str(e)}, using simple processing")
            # Fallback to simple processing without POS tagging
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    # Simple lemmatization without POS
                    lemmatized = self.lemmatizer.lemmatize(token)
                    if lemmatized.isalpha():
                        processed_tokens.append(lemmatized)
            return ' '.join(processed_tokens)
        
        # Lemmatize and filter
        processed_tokens = []
        for token, pos_tag in pos_tags:
            # Skip stop words and short tokens
            if token in self.stop_words or len(token) < 2:
                continue
            
            # Lemmatize with proper POS
            wordnet_pos = self._get_wordnet_pos(pos_tag)
            lemmatized = self.lemmatizer.lemmatize(token, wordnet_pos)
            
            # Only keep alphabetic tokens
            if lemmatized.isalpha():
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def encode(self, texts, convert_to_numpy=True, **kwargs):
        """Encode texts to embeddings using NLTK preprocessing + TF-IDF."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess all texts with NLTK
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        if not self.is_fitted:
            # Initialize and fit TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=1,  # Include all terms
                max_df=0.95,  # Remove terms that appear in >95% of docs
                stop_words='english'  # Additional stop word removal
            )
            
            # Fit on processed texts
            embeddings = self.vectorizer.fit_transform(processed_texts)
            self.is_fitted = True
        else:
            # Transform using fitted vectorizer
            embeddings = self.vectorizer.transform(processed_texts)
        
        # Convert to dense array and pad/truncate to desired dimension
        embeddings = embeddings.toarray()
        
        # Pad or truncate to desired dimension
        if embeddings.shape[1] < self.embedding_dim:
            # Pad with zeros
            padded = np.zeros((embeddings.shape[0], self.embedding_dim))
            padded[:, :embeddings.shape[1]] = embeddings
            embeddings = padded
        elif embeddings.shape[1] > self.embedding_dim:
            # Truncate
            embeddings = embeddings[:, :self.embedding_dim]
        
        if convert_to_numpy:
            embeddings = np.array(embeddings)
        
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings
    
    def get_sentence_embedding_dimension(self):
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def save(self, path):
        """Save the model."""
        os.makedirs(path, exist_ok=True)
        
        # Save TF-IDF vectorizer
        with open(os.path.join(path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save configuration
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump({
                'max_features': self.max_features,
                'embedding_dim': self.embedding_dim,
                'is_fitted': self.is_fitted,
                'model_type': 'hybrid-nltk-sklearn'
            }, f)
    
    def load(self, path):
        """Load the model."""
        with open(os.path.join(path, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
            self.max_features = config['max_features']
            self.embedding_dim = config['embedding_dim']
            self.is_fitted = config['is_fitted']

class CompatibilityEmbedder:
    """Compatibility embedder using hybrid NLTK + scikit-learn processing."""
    
    def __init__(self, max_features: int = 10000, embedding_dim: int = 300):
        """Initialize with hybrid embedder."""
        self.embedder = HybridEmbedder(max_features, embedding_dim)
        self.device = 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def load_chunks(self, chunks_dir: str = 'data/processed/chunks') -> List[Dict]:
        """Load all chunk files."""
        chunks_path = Path(chunks_dir)
        chunks = []
        
        for chunk_file in sorted(chunks_path.glob('chunk_*.json')):
            with open(chunk_file, 'r') as f:
                chunks.append(json.load(f))
        
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def prepare_text_for_embedding(self, chunk: Dict) -> str:
        """Convert chunk data into a text format suitable for embedding."""
        if chunk['type'] == 'metadata':
            return f"Metadata: {json.dumps(chunk['data'])}"
        
        elif chunk['type'] == 'server_chunk':
            server_texts = []
            for server in chunk['servers']:
                server_info = server['server_info']
                deployment_info = server['deployment_info']
                text = (
                    f"Server {server['name']} in {server['environment']} environment. "
                    f"Manufacturer: {server_info['manufacturer']}, "
                    f"Product Class: {server_info['product_class']}, "
                    f"Product Type: {server_info['product_type']}, "
                    f"Model: {server_info['model']}, "
                    f"Status: {server_info['status']}, "
                    f"Install Path: {deployment_info['install_path']}"
                )
                server_texts.append(text)
            return "\n".join(server_texts)
        
        elif chunk['type'] == 'environment_summary':
            return (
                f"Environment Summary for {chunk['environment']}: "
                f"Total Servers: {chunk['data']['total_servers']}, "
                f"Server Types: {json.dumps(chunk['data']['server_types'])}, "
                f"Status Distribution: {json.dumps(chunk['data']['status_distribution'])}"
            )
        
        elif chunk['type'] == 'manufacturer_summary':
            return (
                f"Manufacturer Summary for {chunk['manufacturer']}: "
                f"Total Servers: {chunk['data']['total_servers']}, "
                f"Environments: {json.dumps(chunk['data']['environments'])}, "
                f"Product Types: {json.dumps(chunk['data']['product_types'])}"
            )
        
        return ""
    
    def create_embeddings(self, chunks: List[Dict]) -> Dict[str, Dict]:
        """Create embeddings for all chunks using hybrid approach."""
        logger.info("Creating embeddings with NLTK preprocessing + TF-IDF...")
        
        # First, prepare all texts to fit the vectorizer
        texts = []
        for chunk in chunks:
            text = self.prepare_text_for_embedding(chunk)
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        
        # Store embeddings with chunk info
        result_embeddings = {}
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i:04d}"
            result_embeddings[chunk_id] = {
                'embedding': embeddings[i] if len(embeddings.shape) > 1 else embeddings,
                'type': chunk['type'],
                'metadata': {
                    'chunk_id': chunk_id,
                    'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                    'model': 'hybrid-nltk-sklearn',
                    'preprocessing': 'NLTK tokenization, lemmatization, stop word removal',
                    'vectorization': 'TF-IDF with bigrams'
                }
            }
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
        
        return result_embeddings
    
    def save_embeddings(self, embeddings: Dict[str, Dict], output_dir: str = 'data/processed/embeddings'):
        """Save embeddings to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = output_path / 'embeddings.npy'
        metadata_file = output_path / 'metadata.json'
        
        # Extract embeddings and metadata
        embedding_arrays = {k: v['embedding'] for k, v in embeddings.items()}
        metadata = {k: {**v['metadata'], 'type': v['type']} for k, v in embeddings.items()}
        
        # Save embeddings
        np.save(embeddings_file, embedding_arrays)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save the embedder model
        model_dir = output_path / 'hybrid_model'
        self.embedder.save(model_dir)
        
        logger.info(f"Saved embeddings to {embeddings_file}")
        logger.info(f"Saved metadata to {metadata_file}")
        logger.info(f"Saved hybrid model to {model_dir}")

def main():
    """Test hybrid embedder."""
    try:
        # Initialize embedder
        embedder = CompatibilityEmbedder(max_features=5000, embedding_dim=300)
        
        # Test embeddings
        test_texts = [
            "This is a test sentence for software compatibility.",
            "Another test sentence for validation with server information."
        ]
        
        embeddings = embedder.embedder.encode(test_texts)
        print(f"✅ Hybrid embedder working!")
        print(f"📊 Embedding dimension: {embedder.embedder.get_sentence_embedding_dimension()}")
        print(f"🧪 Test embeddings shape: {embeddings[0].shape}")
        print(f"🚀 Model: NLTK preprocessing + TF-IDF (best of both worlds)")
        
        # Try to load chunks and create embeddings
        try:
            chunks = embedder.load_chunks()
            if chunks:
                embeddings_data = embedder.create_embeddings(chunks)
                embedder.save_embeddings(embeddings_data)
                
                # Log summary
                chunk_types = {}
                for chunk_id, data in embeddings_data.items():
                    chunk_type = data['type']
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                logger.info("Embedding creation complete with hybrid approach!")
                logger.info("Chunk type distribution:")
                for chunk_type, count in chunk_types.items():
                    logger.info(f"{chunk_type}: {count} chunks")
            else:
                logger.info("No chunks found. Run chunking first.")
                
        except Exception as e:
            logger.warning(f"Could not process chunks: {str(e)}")
            logger.info("This is normal if chunks don't exist yet.")
        
        return embedder
        
    except Exception as e:
        print(f"❌ Error with hybrid embedder: {str(e)}")
        print("\n💡 To install dependencies:")
        print("pip install nltk scikit-learn")
        return None

if __name__ == "__main__":
    main()

import nltk
nltk.data.path.append('/Users/emilycho/nltk_data')
print(nltk.data.path) 