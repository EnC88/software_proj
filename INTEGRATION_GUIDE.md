# System Compatibility Assistant - Integration Guide

This guide explains how to integrate and run the new React frontend with your existing RAG pipeline backend.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   Flask API     │    │   RAG Pipeline  │
│   (TypeScript)   │◄──►│   (Python)      │◄──►│   (Python)      │
│   - Chat UI      │    │   - API Routes  │    │   - Vector Store│
│   - System Config│    │   - Rate Limiting│   │   - Embeddings  │
│   - Analytics    │    │   - CORS        │    │   - FAISS Index │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Integrated Runner (Recommended)

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

### Option 3: Manual Setup

#### 1. Install Frontend Dependencies

```bash
cd templates
npm install
npm run build
```

#### 2. Start Backend

```bash
# Set environment variables
export FLASK_ENV=development
export PORT=5000

# Run the Flask app
python src/api/app.py
```

## 📁 Project Structure

```
software_compatibility_project/
├── src/
│   ├── api/
│   │   └── app.py                 # Flask API backend
│   ├── rag/                       # Existing RAG components
│   ├── data_processing/           # Existing data processing
│   └── evaluation/                # Existing evaluation system
├── templates/                     # React frontend (Flask templates)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx  # Main chat interface
│   │   │   ├── SystemConfiguration.tsx
│   │   │   └── StatsOverview.tsx
│   │   ├── lib/
│   │   │   └── api.ts            # API service
│   │   └── pages/
│   │       └── Index.tsx         # Main page
│   ├── package.json
│   └── vite.config.ts
├── data/                          # Existing data
├── templates/                     # Old Gradio interface
├── run_integrated_app.py         # Integrated runner
├── docker-compose.integrated.yml # Docker setup
└── Dockerfile.integrated         # Docker build
```

## 🔧 API Endpoints

### Core Endpoints

- `GET /api/health` - Health check
- `POST /api/analyze` - Analyze compatibility
- `POST /api/feedback` - Submit feedback
- `GET /api/analytics` - Get analytics data
- `GET /api/suggestions` - Get quick actions

### Frontend Routes

- `GET /` - Main application
- `GET /<static>` - Static assets

## 💬 Chat Interface Features

### System Configuration
- Operating system selection
- Database selection
- Web server selection
- Real-time configuration summary

### Chat Features
- Real-time compatibility analysis
- Quick action buttons
- Loading states
- Error handling
- Message history

### Analytics Integration
- Query analytics
- Feedback collection
- Performance metrics
- User behavior tracking

## 🔄 Data Flow

1. **User Input**: User types query in chat interface
2. **System Config**: Configuration is passed to API
3. **API Processing**: Flask processes request with RAG pipeline
4. **Analysis**: Compatibility analysis performed
5. **Response**: Formatted results returned to frontend
6. **Display**: Results displayed in chat interface
7. **Feedback**: User can provide feedback on results

## 🛠️ Development

### Frontend Development

```bash
cd templates
npm run dev  # Start development server
```

### Backend Development

```bash
# Run Flask in development mode
export FLASK_ENV=development
python src/api/app.py
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

## 🔒 Security Features

- Rate limiting on all endpoints
- CORS configuration
- Input validation
- Error handling without information leakage
- Secure credential management

## 📊 Analytics Dashboard

The integrated application includes:

- **Query Analytics**: Track query patterns and performance
- **Feedback System**: Collect and analyze user feedback
- **System Metrics**: Monitor application health
- **User Behavior**: Understand usage patterns

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build and run production container
docker-compose -f docker-compose.integrated.yml up -d

# View logs
docker-compose -f docker-compose.integrated.yml logs -f

# Stop services
docker-compose -f docker-compose.integrated.yml down
```

### Environment Variables

```bash
# Required for production
export FLASK_SECRET_KEY=your-secure-secret-key
export POSTGRES_PASSWORD=your-secure-password

# Optional
export FLASK_ENV=production
export PORT=5000
```

## 🔧 Configuration

### Frontend Configuration

Create `.env` file in `templates/` directory:

```env
VITE_API_URL=http://localhost:5000/api
```

### Backend Configuration

Environment variables for the Flask app:

```bash
FLASK_ENV=development|production
FLASK_SECRET_KEY=your-secret-key
PORT=5000
```

## 🚨 Troubleshooting

### Common Issues

1. **Frontend Build Fails**
   ```bash
   cd templates
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

2. **API Connection Issues**
   - Check if Flask server is running
   - Verify API URL in frontend configuration
   - Check CORS settings

3. **RAG Pipeline Issues**
   - Ensure data pipeline has been run
   - Check FAISS index exists
   - Verify embeddings are generated

4. **Docker Issues**
   ```bash
   # Clean up Docker
   docker-compose -f docker-compose.integrated.yml down -v
   docker system prune -f
   docker-compose -f docker-compose.integrated.yml up --build
   ```

### Logs

- **Frontend**: Check browser console
- **Backend**: Check Flask logs
- **Docker**: `docker-compose logs -f integrated-app`

## 🔄 Migration from Gradio

The new React frontend replaces the existing Gradio interface:

- **Old**: `templates/landing.py` (Gradio)
- **New**: `templates/` (React + Flask API)

### Migration Steps

1. **Backup existing data**
   ```bash
   cp -r data data_backup
   ```

2. **Test new interface**
   ```bash
   python run_integrated_app.py
   ```

3. **Update deployment scripts**
   - Replace Gradio references with new endpoints
   - Update Docker configurations
   - Update CI/CD pipelines

4. **Remove old interface** (optional)
   ```bash
   rm -rf templates/
   ```

## 📈 Performance

### Frontend Performance
- React 18 with concurrent features
- Vite for fast development and builds
- Tailwind CSS for optimized styling
- Lazy loading for components

### Backend Performance
- Flask with rate limiting
- FAISS for fast similarity search
- PostgreSQL for data persistence
- Redis for caching

## 🔮 Future Enhancements

- Real-time WebSocket communication
- Advanced analytics dashboard
- Multi-language support
- Mobile app version
- Enterprise SSO integration
- Advanced ML model integration

## 📞 Support

For issues or questions:

1. Check the troubleshooting section
2. Review logs for error messages
3. Test individual components
4. Verify configuration settings

## 📄 License

[Your License Here] 