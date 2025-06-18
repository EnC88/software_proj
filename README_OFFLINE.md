# Offline Vectorizer Usage Guide

This guide shows you how to use the vectorizer without internet access.

## 🚀 Quick Start

### Option 1: Download Model Once (Recommended)

```bash
# Download the model locally (requires internet once)
python download_model_offline.py
```

Then use it offline:
```python
from src.vectorizer import Vectorizer

# Use local model
vectorizer = Vectorizer(
    model_path='./models/all-MiniLM-L6-v2'  # Local path
)
```

### Option 2: Use Lightweight Model

```bash
# See lightweight alternatives
python offline_alternatives.py
```

## 📦 Available Models

| Model | Size | Speed | Quality | Offline |
|-------|------|-------|---------|---------|
| `paraphrase-MiniLM-L3-v2` | ~60MB | Very Fast | Good | ✅ |
| `all-MiniLM-L6-v2` | ~90MB | Very Fast | Good | ✅ |
| `multi-qa-MiniLM-L6-cos-v1` | ~90MB | Very Fast | Good for Q&A | ✅ |
| `all-mpnet-base-v2` | ~420MB | Fast | Excellent | ✅ |

## 🔧 Manual Download

If you can't use the download script, manually download:

1. **Find model files** on Hugging Face:
   - Go to: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
   - Download: `config.json`, `pytorch_model.bin`, `sentence_bert_config.json`, `tokenizer_config.json`, `vocab.txt`

2. **Create directory structure**:
   ```
   models/
   └── all-MiniLM-L6-v2/
       ├── config.json
       ├── pytorch_model.bin
       ├── sentence_bert_config.json
       ├── tokenizer_config.json
       └── vocab.txt
   ```

3. **Use in code**:
   ```python
   vectorizer = Vectorizer(model_path='./models/all-MiniLM-L6-v2')
   ```

## 💡 Usage Examples

### Basic Offline Usage
```python
from src.vectorizer import Vectorizer
import pandas as pd

# Initialize with local model
vectorizer = Vectorizer(
    use_database=False,
    use_cache=True,
    model_path='./models/all-MiniLM-L6-v2'
)

# Your data
data = {
    'CHUNK_TEXT': ['Upgraded Apache from 2.4.1 to 2.4.2'],
    'OBJECTNAME': ['Apache'],
    'OLDVALUE': ['2.4.1'],
    'NEWVALUE': ['2.4.2']
}

df = pd.DataFrame(data)
vectorizer.chunked_df = df
vectorizer.vectorize()

# Query
results = vectorizer.query_upgrades("How to upgrade Apache?")
print(f"Found {len(results)} similar upgrades")
```

### Large Dataset Offline Usage
```python
# For couple million data points
vectorizer = Vectorizer(
    use_database=True,  # Use database for large data
    use_cache=True,
    model_path='./models/paraphrase-MiniLM-L3-v2'  # Lightest model
)

# Load your large dataset
vectorizer.chunked_df = your_large_dataframe
vectorizer.vectorize()

# Query with caching
results = vectorizer.query_upgrades("Your query here", top_k=10)
```

## 🔒 Firewall/Network Restrictions

### If you can't access huggingface.co:

1. **Use a machine with internet** to download models
2. **Copy the `models/` directory** to your restricted environment
3. **Use `model_path` parameter** to point to local files

### Alternative: Use smaller models
```python
# Smallest model for restricted environments
vectorizer = Vectorizer(
    model_name='paraphrase-MiniLM-L3-v2',  # Only ~60MB
    model_path='./models/paraphrase-MiniLM-L3-v2'
)
```

## 📊 Performance Comparison

| Model | Download Size | Memory Usage | Speed | Quality |
|-------|---------------|--------------|-------|---------|
| `paraphrase-MiniLM-L3-v2` | 60MB | Low | ⚡⚡⚡ | Good |
| `all-MiniLM-L6-v2` | 90MB | Medium | ⚡⚡⚡ | Good |
| `all-mpnet-base-v2` | 420MB | High | ⚡⚡ | Excellent |

## 🛠️ Troubleshooting

### Model not found error:
```python
# Check if model exists
import os
model_path = './models/all-MiniLM-L6-v2'
if not os.path.exists(model_path):
    print("Model not found. Run: python download_model_offline.py")
```

### Memory issues:
```python
# Use smaller model
vectorizer = Vectorizer(
    model_name='paraphrase-MiniLM-L3-v2',  # Smaller
    model_path='./models/paraphrase-MiniLM-L3-v2'
)
```

### Slow performance:
```python
# Increase cache size for large datasets
vectorizer = Vectorizer(
    model_path='./models/all-MiniLM-L6-v2',
    use_cache=True  # Cache is already optimized for large data
)
```

## ✅ Success Checklist

- [ ] Model downloaded to `./models/` directory
- [ ] Vectorizer initialized with `model_path` parameter
- [ ] Test query returns results
- [ ] Works without internet access

## 🎯 Next Steps

1. **Download a model**: `python download_model_offline.py`
2. **Test offline usage**: `python example_offline_usage.py`
3. **Scale to your data**: Use with your couple million records
4. **Optimize performance**: Adjust cache and database settings

Your vectorizer is now ready for offline use! 🚀 