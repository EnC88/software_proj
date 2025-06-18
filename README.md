# Software Compatibility RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline for software compatibility analysis using spaCy embeddings and FAISS indexing.

## 🏗️ Architecture

```
src/
├── data_processing/          # Data processing pipeline
│   ├── chunk_compatibility.py    # Split data into chunks
│   ├── spacy_embedder.py         # Generate embeddings with spaCy
│   └── build_faiss_index.py      # Build FAISS index
├── rag/                     # RAG components
│   ├── query_engine.py          # Query engine for similarity search
│   └── compatibility_rag.py     # Legacy RAG implementation
models/                      # Model files
data/
└── processed/               # Processed data
    ├── chunks/              # Chunked data
    ├── embeddings/          # Generated embeddings
    └── faiss_index/         # FAISS index files
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf  # Best quality model
```

### 2. Run Complete Pipeline
```bash
python pipeline.py
```

This will run:
1. **Chunking** - Split raw data into meaningful chunks
2. **Embedding** - Generate embeddings using spaCy transformer model
3. **Indexing** - Build FAISS index for fast similarity search

### 3. Test Query Engine
```bash
python src/rag/query_engine.py
```

## 📋 Pipeline Steps

### Step 1: Chunking
```bash
python src/data_processing/chunk_compatibility.py
```
- Splits raw data into server chunks, environment summaries, etc.
- Output: `data/processed/chunks/chunk_*.json`

### Step 2: Embedding
```bash
python src/data_processing/spacy_embedder.py
```
- Uses spaCy `en_core_web_trf` (transformer-based, highest quality)
- Generates 768-dimensional embeddings
- Output: `data/processed/embeddings/embeddings.npy` and `metadata.json`

### Step 3: Indexing
```bash
python src/data_processing/build_faiss_index.py
```
- Builds FAISS index for fast similarity search
- Output: `data/processed/faiss_index/index.faiss` and `id_to_chunk.json`

### Step 4: Querying
```python
from src.rag.query_engine import QueryEngine

# Initialize query engine
engine = QueryEngine()

# Query for similar chunks
results = engine.query("What servers are compatible with Windows Server 2019?", top_k=5)

# Format for LLM
llm_context = engine.format_results_for_llm(results)
```

## 🔧 Configuration

### Model Options
- **`en_core_web_trf`** (default) - Transformer-based, highest quality, 768d
- **`en_core_web_lg`** - Large model, good quality, 300d
- **`en_core_web_md`** - Medium model, balanced, 300d
- **`en_core_web_sm`** - Small model, fast, 96d

### Custom Paths
```python
engine = QueryEngine(
    model_name='en_core_web_trf',
    index_path='custom/path/index.faiss',
    id_to_chunk_path='custom/path/id_to_chunk.json',
    metadata_path='custom/path/metadata.json',
    chunks_dir='custom/path/chunks'
)
```

## 📊 Performance

- **Embedding Quality**: Transformer-based (BERT-style) embeddings
- **Search Speed**: FAISS index for sub-second similarity search
- **Memory Usage**: ~500MB for transformer model + index
- **Offline Capable**: No internet required after model download

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_trf
   ```

2. **FAISS not installed**
   ```bash
   pip install faiss-cpu  # or faiss-gpu for GPU support
   ```

3. **Memory issues with large datasets**
   - Use smaller spaCy model: `en_core_web_sm`
   - Reduce batch size in embedding generation

### Logs
- Check `pipeline.log` for detailed execution logs
- All scripts include comprehensive logging

## 🔄 Development Workflow

1. **Data Changes**: Update raw data, run `pipeline.py`
2. **Model Changes**: Update `spacy_embedder.py`, re-run pipeline
3. **Query Logic**: Modify `query_engine.py`, test with sample queries

## 📝 API Reference

### QueryEngine

```python
class QueryEngine:
    def __init__(self, model_name='en_core_web_trf', ...)
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]
    def format_results_for_llm(self, results: List[Dict]) -> str
    def get_stats(self) -> Dict[str, Any]
```

### Example Usage

```python
from src.rag.query_engine import QueryEngine

# Initialize
engine = QueryEngine()

# Get system stats
stats = engine.get_stats()
print(f"Index size: {stats['index_size']}")

# Query
results = engine.query("Dell servers in production", top_k=3)

# Use with LLM
context = engine.format_results_for_llm(results)
prompt = f"Based on this context: {context}\n\nAnswer the question: ..."
```

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive logging
3. Include error handling
4. Update this README for new features

## 📄 License

[Your License Here]
