# Software Compatibility RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline for software compatibility analysis using spaCy embeddings and FAISS indexing, now with a modern React frontend.

## 🏗️ Architecture

```
src/
├── api/                        # NEW: Flask API backend
│   └── app.py                     # REST API for React frontend
├── data_processing/          # Data processing pipeline
│   ├── chunk_compatibility.py    # Split data into chunks
│   ├── spacy_embedder.py         # Generate embeddings with spaCy
│   └── build_faiss_index.py      # Build FAISS index
├── rag/                     # RAG components
│   ├── vector_store.py          # Vector store for similarity search
│   └── __init__.py
├── evaluation/              # Feedback and evaluation system
│   ├── feedback_system.py       # Consolidated feedback system (43KB)
│   │   ├── FeedbackLogger       # SQLite-based feedback logging
│   │   ├── FeedbackIntegration  # Integration with query engine
│   │   ├── FeedbackLoop         # Automated feedback analysis
│   │   ├── AutomatedScheduler   # Scheduled feedback loop execution
│   │   └── Demo & CLI tools     # Testing and demonstration functions
│   ├── web_interface.py         # Flask API interface (17KB)
│   └── test_feedback.py         # CLI test interface (7.5KB)
templates/                    # NEW: React frontend (Flask templates)
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx    # Main chat interface
│   │   ├── SystemConfiguration.tsx
│   │   └── StatsOverview.tsx
│   ├── lib/
│   │   └── api.ts              # API service
│   └── pages/
│       └── Index.tsx           # Main page
models/                      # Model files
data/
└── processed/               # Processed data
    ├── chunks/              # Chunked data
    ├── embeddings/          # Generated embeddings
    └── faiss_index/         # FAISS index files
```

## 🚀 Quick Start

### Option 1: Integrated Application (Recommended)

The easiest way to run the application with the new React frontend:

```bash
# Run the integrated application
python run_integrated_app.py
```

This will:
1. Check for Node.js and npm
2. Install frontend dependencies
3. Build the React frontend
4. Start the Flask backend
5. Serve the app at http://localhost:5000

### Option 2: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.integrated.yml up --build
```

### Option 3: Legacy Gradio Interface

If you prefer the original Gradio interface:

```bash
# Start the application
./deploy.sh start

# View logs
./deploy.sh logs

# Stop the application
./deploy.sh stop

# Check status
./deploy.sh status
```

**Access the application:**
- **New React Interface**: http://localhost:5000 (Integrated Flask + React)
- **Legacy Gradio Interface**: http://localhost:7860 (Gradio dashboard)
- **API Endpoints**: http://localhost:5000/api (Flask API)

## 🆕 New React Frontend Features

### Modern Chat Interface
- Real-time compatibility analysis
- System configuration panel
- Quick action buttons
- Loading states and error handling
- Message history

### System Configuration
- Operating system selection
- Database selection
- Web server selection
- Real-time configuration summary

### Analytics Dashboard
- Query analytics and performance metrics
- Feedback collection and analysis
- User behavior tracking
- System health monitoring

## 🔧 API Endpoints

The new Flask API provides:

- `GET /api/health` - Health check
- `POST /api/analyze` - Analyze compatibility
- `POST /api/feedback` - Submit feedback
- `GET /api/analytics` - Get analytics data
- `GET /api/suggestions` - Get quick actions

## 🐳 Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available

### Security Setup (Required)
Before starting the application, you must set up secure credentials:

```bash
# Generate secure credentials and create .env file
./deploy.sh setup

# This will create a .env file with:
# - Secure PostgreSQL password
# - Secure Flask secret key
# - Proper database URL
```

**⚠️ Security Note**: Never commit the `.env` file to version control. It contains sensitive credentials.

### Quick Commands
```bash
# Generate secure credentials (first time only)
./deploy.sh setup

# Start integrated application
python run_integrated_app.py

# Or use Docker
docker-compose -f docker-compose.integrated.yml up -d

# View logs
docker-compose -f docker-compose.integrated.yml logs -f

# Stop application
docker-compose -f docker-compose.integrated.yml down

# Check status
docker-compose -f docker-compose.integrated.yml ps
```

### Services Available
- **React Frontend**: http://localhost:5000 (Integrated interface)
- **Legacy Gradio**: http://localhost:7860 (Original interface)
- **API**: http://localhost:5000/api (Flask API)
- **PostgreSQL**: localhost:5432 (Database)
- **Redis**: localhost:6379 (Caching)

## 🧪 Testing

### Integration Tests
```bash
# Test the integrated application
python test_integration.py
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:5000/api/health

# Test analysis endpoint
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Upgrade Apache to 2.4.50"}'
```

## 📊 Analytics Dashboard

### Features
- **Interactive Visualizations**: Real-time charts and graphs using Plotly
- **Filtering**: Filter by date range, OS, feedback score
- **Metrics**: Total feedback, positive/negative rates, session analysis
- **Export**: Download data as CSV or JSON
- **Activity Patterns**: Hourly and daily usage patterns
- **Input Validation**: All inputs are validated to prevent errors

### Access
```bash
# Start the dashboard
./deploy.sh start

# Access at: http://localhost:8501
```

## 🔄 Migration from Gradio

The new React frontend provides a modern alternative to the Gradio interface:

- **Old**: `templates/landing.py` (Gradio)
- **New**: `frontend/` (React + Flask API)

### Migration Steps

1. **Test new interface**
   ```bash
   python run_integrated_app.py
   ```

2. **Update deployment scripts**
   - Replace Gradio references with new endpoints
   - Update Docker configurations
   - Update CI/CD pipelines

3. **Remove old interface** (optional)
   ```bash
   rm -rf templates/
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
from src.rag.vector_store import VectorStore

# Initialize vector store
store = VectorStore()

# Query for similar chunks
results = store.query("What servers are compatible with Windows Server 2019?", top_k=5)

# Get formatted results
formatted_results = store.get_stats()
```

## 🔧 Configuration

### Model Options
- **`en_core_web_trf`** (default) - Transformer-based, highest quality, 768d
- **`en_core_web_lg`** - Large model, good quality, 300d
- **`en_core_web_md`** - Medium model, balanced, 300d
- **`en_core_web_sm`** - Small model, fast, 96d

### Custom Paths
```python
store = VectorStore(
    data_dir='custom/path/data',
    chunk_size=1000,
    chunk_overlap=200
)
```

## 📊 Performance

- **Embedding Quality**: Transformer-based (BERT-style) embeddings
- **Search Speed**: FAISS index for sub-second similarity search
- **Memory Usage**: ~500MB for transformer model + index
- **Offline Capable**: No internet required after model download
- **Frontend Performance**: React 18 with Vite for fast builds
- **Backend Performance**: Flask with rate limiting and caching

## 🐛 Troubleshooting

### Common Issues

1. **Frontend Build Fails**
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

2. **API Connection Issues**
   - Check if Flask server is running
   - Verify API URL in frontend configuration
   - Check CORS settings

3. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_trf
   ```

4. **FAISS not installed**
   ```bash
   pip install faiss-cpu  # or faiss-gpu for GPU support
   ```

5. **Memory issues with large datasets**
   - Use smaller spaCy model: `en_core_web_sm`
   - Reduce batch size in embedding generation

### Logs
- Check `pipeline.log` for detailed execution logs
- All scripts include comprehensive logging
- Frontend: Check browser console
- Backend: Check Flask logs

## 🔄 Development Workflow

1. **Data Changes**: Update raw data, run `pipeline.py`
2. **Model Changes**: Update `spacy_embedder.py`, re-run pipeline
3. **Query Logic**: Modify `vector_store.py`, test with sample queries
4. **Frontend Changes**: Modify React components in `frontend/src/`
5. **API Changes**: Modify Flask routes in `src/api/app.py`

## 📄 API Reference

### VectorStore

```python
class VectorStore:
    def __init__(self, data_dir=None, chunk_size=1000, chunk_overlap=200)
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]
    def get_stats(self) -> Dict[str, Any]
    def add_document(self, content: str, metadata: Dict[str, Any] = None)
    def reload_vectorstore(self)
```

### Example Usage

```python
from src.rag.vector_store import VectorStore

# Initialize
store = VectorStore()

# Get system stats
stats = store.get_stats()
print(f"Documents loaded: {stats['total_documents']}")

# Query
results = store.query("Dell servers in production", top_k=3)

# Add new document
store.add_document("New server information", {"type": "server", "environment": "prod"})
```

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive logging
3. Include error handling
4. Update this README for new features
5. Test both frontend and backend changes

## 📄 License

[Your License Here]
