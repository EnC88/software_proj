# LangChain Integration Guide

This project now includes LangChain integration for enhanced RAG (Retrieval-Augmented Generation) capabilities. The integration is designed to work **completely offline and free** with no API keys or paid services required.

## 🆓 Completely Free Features (No API Keys Required)

### 1. Vector Store with Local Embeddings
- Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- FAISS vector database for similarity search
- Completely free and runs locally

### 2. Free Local LLM Options
- **Ollama**: Run local LLMs like Llama2, Mistral, etc.
- **CTransformers**: Lightweight local LLM inference
- No API costs, no internet required after setup

### 3. Enhanced Document Processing
- Automatic text splitting and chunking
- Metadata extraction and storage
- Support for multiple data sources (CSV, JSON)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Usage (Completely Free)
```python
from src.rag.vector_store import VectorStore

# Initialize with free local models
store = VectorStore(use_local_llm=True)

# Query the vector store
results = store.query("What servers are running Apache?", top_k=5)
print(results)
```

### 3. Enhanced Integration
```python
from src.rag.langchain_integration import LangChainIntegration

# Initialize integration
integration = LangChainIntegration(use_local_llm=True)

# Enhanced query with compatibility analysis
results = integration.enhanced_query(
    "I want to upgrade Apache to 2.4.50", 
    use_llm=True
)
print(results)
```

## 🔧 Local LLM Setup

### Option 1: Ollama (Recommended)
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Use in code:
```python
store = VectorStore(
    use_local_llm=True,
    local_llm_model="llama2"
)
```

### Option 2: CTransformers
1. Install: `pip install ctransformers`
2. Models are downloaded automatically
3. Use in code:
```python
store = VectorStore(
    use_local_llm=True,
    local_llm_model="llama2"
)
```

## 📊 Features Comparison

| Feature | Free (Local) |
|---------|-------------|
| Vector Search | ✅ |
| Document Processing | ✅ |
| Local LLM Responses | ✅ |
| Conversational Memory | ✅ |
| Cost | $0 |
| Privacy | 100% Local |

## 🔍 Example Queries

### Vector Search Only
```python
results = store.query("Apache HTTPD servers")
```

### With Local LLM Response
```python
results = store.query("Apache HTTPD servers", use_llm=True)
```

### Conversational Query
```python
response = store.conversational_query("What's the latest Apache version?")
```

### Compatibility Analysis
```python
integration = LangChainIntegration()
results = integration.enhanced_query("Upgrade Apache to 2.4.50")
```

## 📁 File Structure

```
src/rag/
├── vector_store.py          # Main vector store implementation
├── langchain_integration.py # Integration with existing systems
├── determine_recs.py        # Existing compatibility analyzer
└── query_engine.py          # Original query engine (still available)
```

## 🛠️ Configuration

### Custom Models
```python
store = VectorStore(
    embeddings_model="sentence-transformers/all-mpnet-base-v2",  # Different embeddings
    local_llm_model="mistral",  # Different local LLM
    use_local_llm=True
)
```

## 🔧 Troubleshooting

### No Local LLM Available
If local LLMs aren't working, the system will fall back to vector search only:
```python
# Check what's available
stats = store.get_stats()
print(stats)
```

### Memory Issues
For large datasets, reduce chunk size:
```python
store = VectorStore(
    text_splitter_chunk_size=500,  # Smaller chunks
    text_splitter_chunk_overlap=100
)
```

### Performance Optimization
```python
store = VectorStore(
    embeddings_model="sentence-transformers/all-MiniLM-L6-v2",  # Faster model
    use_local_llm=False  # Skip LLM for faster queries
)
```

## 🎯 Use Cases

1. **Document Search**: Find relevant information in your data
2. **Compatibility Analysis**: Enhanced software compatibility checking
3. **Conversational Interface**: Chat with your data
4. **Knowledge Base**: Build a searchable knowledge base
5. **Data Exploration**: Discover patterns in your infrastructure data

## 📈 Benefits

- **Cost**: Completely free with local models
- **Privacy**: All data stays local - no data sent to external services
- **Performance**: Fast vector search
- **Flexibility**: Works with or without LLMs
- **Integration**: Seamlessly works with existing systems
- **No Dependencies**: No API keys or external services required

## 🔄 Migration from Old System

The new LangChain integration is designed to work alongside your existing system:

```python
# Old way (still works)
from src.rag.query_engine import QueryEngine
engine = QueryEngine()

# New way (enhanced)
from src.rag.vector_store import VectorStore
store = VectorStore()

# Integration way (best of both)
from src.rag.langchain_integration import LangChainIntegration
integration = LangChainIntegration()
```

The integration provides enhanced capabilities while maintaining backward compatibility with your existing code.

## 🚫 What's NOT Included

- No OpenAI integration (requires API key and costs money)
- No Anthropic integration (requires API key and costs money)
- No other paid LLM services
- No external API calls for LLM responses

Everything runs locally and is completely free! 