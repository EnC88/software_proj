# Software Compatibility Project

This project provides tools for analyzing and querying software upgrade experiences using natural language processing and vector embeddings.

## Project Structure

```
software_compatibility_project/
├── src/
│   ├── __init__.py
│   ├── vectorizer.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── query_parser.py
│   └── database/
│       ├── __init__.py
│       └── upgrade_db.py
├── test_vectorizer.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To test the vectorizer with sample data:

```bash
python test_vectorizer.py
```

## Components

- **QueryParser**: Handles natural language query parsing and intent classification
- **UpgradeDatabase**: Manages database operations for storing and retrieving upgrade vectors
- **UpgradeVectorizer**: Main class that coordinates the components and provides the vectorization functionality

## Features

- Natural language query processing
- Vector embeddings for upgrade experiences
- Similarity search for finding relevant upgrade cases
- Version information extraction and analysis
- Database storage for persistent vector storage
- Comprehensive logging and error handling
