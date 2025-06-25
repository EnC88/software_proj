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
├── evaluation/              # Feedback and evaluation system
│   ├── feedback_system.py       # Consolidated feedback system (43KB)
│   │   ├── FeedbackLogger       # SQLite-based feedback logging
│   │   ├── FeedbackIntegration  # Integration with query engine
│   │   ├── FeedbackLoop         # Automated feedback analysis
│   │   ├── AutomatedScheduler   # Scheduled feedback loop execution
│   │   └── Demo & CLI tools     # Testing and demonstration functions
│   ├── web_interface.py         # Flask API interface (17KB)
│   └── test_feedback.py         # CLI test interface (7.5KB)
models/                      # Model files
data/
└── processed/               # Processed data
    ├── chunks/              # Chunked data
    ├── embeddings/          # Generated embeddings
    └── faiss_index/         # FAISS index files
```

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

The easiest way to run the application is using Docker:

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
- **Web Interface**: http://localhost:7860 (Gradio dashboard)
- **API Endpoints**: http://localhost:5000 (Flask API)

### Option 2: Local Development

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

# Start application
./deploy.sh start

# View logs
./deploy.sh logs

# Stop application
./deploy.sh stop

# Restart application
./deploy.sh restart

# Check status
./deploy.sh status

# Clean up (removes all containers and images)
./deploy.sh cleanup
```

### Manual Docker Commands
```bash
# Build and start
docker-compose up --build -d

# View logs
docker-compose logs -f app

# Stop
docker-compose down

# Rebuild
docker-compose up --build --force-recreate -d
```

### Services Available
- **Main App**: http://localhost:7860 (Gradio interface)
- **Analytics Dashboard**: http://localhost:8501 (Streamlit dashboard)
- **API**: http://localhost:5000 (Flask API)
- **PostgreSQL**: localhost:5432 (Database)
- **Redis**: localhost:6379 (Caching)

### Production Deployment
The system now includes PostgreSQL and Redis for production use with enhanced security:

```yaml
# Services automatically included:
postgres:
  image: postgres:15
  # Production database with JSONB support and secure credentials

redis:
  image: redis:7-alpine
  # Caching and session management

# Security features:
# - Rate limiting on all API endpoints
# - Security headers (XSS protection, CSRF, etc.)
# - Input validation and sanitization
# - Secure credential management
```

### Environment Variables
- `PYTHONPATH=/app` - Python path configuration
- `GRADIO_SERVER_NAME=0.0.0.0` - Allow external connections
- `GRADIO_SERVER_PORT=7860` - Gradio server port
- `FLASK_ENV=production` - Flask environment
- `DATABASE_URL` - Database connection (auto-generated)
- `FLASK_SECRET_KEY` - Flask secret key (auto-generated)
- `POSTGRES_PASSWORD` - Database password (auto-generated)

### Data Persistence
- `./data:/app/data` - Application data is persisted
- `./logs:/app/logs` - Application logs are persisted
- `postgres_data:/var/lib/postgresql/data` - Database data is persisted

### Security Features
- **Rate Limiting**: API endpoints are rate-limited to prevent abuse
- **Input Validation**: All inputs are validated and sanitized
- **Security Headers**: XSS protection, CSRF protection, content type validation
- **Secure Credentials**: Auto-generated secure passwords and keys
- **Error Handling**: Proper error handling without information leakage
- **Database Security**: Parameterized queries to prevent SQL injection

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

### Dashboard Sections
1. **Overview Metrics**: Key performance indicators
2. **Feedback Trends**: Time-series analysis
3. **Activity Patterns**: Usage patterns by hour/day
4. **OS Analysis**: Performance by operating system
5. **Session Analysis**: User session statistics
6. **Recent Feedback**: Latest feedback entries

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
The project includes a comprehensive CI/CD pipeline with security scanning:

```yaml
# Triggered on:
# - Push to main/develop branches
# - Pull requests to main branch
```

### Pipeline Stages
1. **Test**: Multi-Python version testing (3.9, 3.10, 3.11)
2. **Security**: Bandit security scanning with high-severity failure
3. **Build**: Docker image building and pushing
4. **Deploy**: Staging and production deployments
5. **Performance**: Load testing with Locust
6. **Notify**: Success/failure notifications

### Features
- **Code Quality**: Flake8 linting, Black formatting
- **Test Coverage**: Pytest with coverage reporting
- **Security**: Automated security scanning with failure on high-severity issues
- **Multi-Platform**: Docker images for AMD64 and ARM64
- **Caching**: Optimized build caching
- **Notifications**: Slack/webhook notifications

### Setup
1. **Enable GitHub Actions** in your repository
2. **Set up environments** (staging, production) in GitHub
3. **Configure secrets** for deployment credentials
4. **Add notification webhooks** (optional)

### Manual Testing
```bash
# Run tests locally
pytest tests/ --cov=src/evaluation/

# Run linting
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run security scan
bandit -r src/ -f json -o bandit-report.json
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
