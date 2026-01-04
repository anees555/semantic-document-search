# Semantic Document Search & Question Answering System

A comprehensive semantic document search system with question-answering capabilities. Features advanced document processing, vector embeddings, and multiple query interfaces for academic and research documents.

## Project Overview

This system provides semantic search and intelligent question-answering across document collections using:
- ChromaDB vector database for semantic storage
- sentence-transformers for high-quality embeddings  
- Groq API integration for natural language responses
- PDF processing with Grobid for academic papers

## Features

### Core Capabilities
- **Semantic Search**: Natural language queries with similarity scoring
- **Question Answering**: AI-powered responses using retrieved document context
- **Multi-Format Support**: PDF, TXT, Markdown document processing
- **Academic Focus**: Structure-aware processing for research papers
- **Interactive Interfaces**: Command-line and programmatic access
- **Persistent Storage**: ChromaDB vector database with metadata

### Performance
- **Search Speed**: Sub-second semantic queries  
- **Embedding Model**: all-MiniLM-L6-v2 (384-dimensional vectors)
- **Processing Rate**: ~43 embeddings/second (CPU)
- **API Integration**: Groq for fast language model inference
- **Memory Efficient**: Optimized embedding generation and storage



## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv vectorenv
vectorenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Document Processing
```bash
# Load and process documents
python app.py load

# Start interactive search
python app.py search
```

### 3. Question Answering with AI

**Setup API Key:**
```bash
# Copy template and add your API key
copy .env.template .env
# Edit .env file and add your Groq API key from https://console.groq.com/
```

**Option A: Advanced QA with AI (Recommended)**
```bash
# Ask questions with AI responses (uses .env configuration)
python semantic_qa.py "explain transformer architecture"

# Interactive mode
python semantic_qa.py interactive
```

**Option B: Simple Document-based QA**
```bash
# Fast responses without external API
python document_qa.py "what is attention mechanism?"

# Interactive mode
python document_qa.py interactive
```

### 4. Optional: Enhanced PDF Processing
```bash
# Start Grobid server for academic papers
docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0
```

##  System Performance

**Test Results** (2 documents, 49 chunks processed):
- **Embedding Generation**: 384-dimensional vectors using all-MiniLM-L6-v2
- **Vector Database**: ChromaDB with persistent storage 
- **Search Performance**: < 50ms semantic queries with similarity scoring
- **Memory Usage**: Single model instance shared across pipeline (optimized)

### Example Search Results
```
Query: "attention mechanism"
├── Score: 0.832 - "...attention mechanism allows the model to focus on relevant parts..."
├── Score: 0.798 - "...self-attention computes attention weights for each position..."
└── Score: 0.765 - "...multi-head attention provides multiple representation subspaces..."
```
## Usage Examples

### Semantic Search
```python
from src.integrated_pipeline import DocumentSearchPipeline

# Initialize pipeline
pipeline = DocumentSearchPipeline(
    use_grobid=True,
    chunk_size=1000,
    chunk_overlap=150
)

# Process documents
results = pipeline.process_documents_directory("documents/")

# Search documents
search_results = pipeline.search_documents(
    "machine learning algorithms", 
    n_results=5
)
```

### Question Answering
```python
# Using Groq-powered QA system
from semantic_qa import GroqRAGSystem

qa_system = GroqRAGSystem()
response = qa_system.ask_question("What is transformer architecture?")
print(response['answer'])
```

### Document Processing
```python
from src.document_loader import DocumentLoader
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore

# Process individual documents
loader = DocumentLoader(use_grobid=True)
chunks = loader.load_pdf("research_paper.pdf")

# Generate embeddings
embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings(chunks)

# Store in vector database
store = VectorStore(embedder, persist_directory="data/chroma_db")
results = store.search("neural networks", n_results=3)
```

## Project Structure

```
semantic-document-search/
├── src/
│   ├── document_loader.py       # PDF and document processing
│   ├── embedding_generator.py   # Semantic embeddings generation
│   ├── vector_store.py          # ChromaDB vector operations
│   ├── integrated_pipeline.py   # Complete processing pipeline
│   ├── groq_llm_interface.py   # Groq API integration
│   └── rag_*.py                # Question-answering components
├── documents/                   # Input documents directory
├── data/                       # ChromaDB persistent storage
├── app.py                      # Main application interface
├── semantic_qa.py              # AI-powered question answering
├── document_qa.py              # Document-based question answering
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## System Architecture

### Core Components
- **Document Processing**: Multi-format support with Grobid integration for academic papers
- **Vector Embeddings**: sentence-transformers with 384-dimensional semantic vectors
- **Vector Storage**: ChromaDB for persistent, searchable document storage
- **Question Answering**: Groq API integration for natural language responses
- **Search Interface**: Multiple query modes with similarity scoring

### Technical Specifications
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions, cosine similarity)
- **Vector Database**: ChromaDB with persistent SQLite storage
- **API Integration**: Groq for fast language model inference (14,400 requests/day free)
- **Document Support**: PDF, TXT, Markdown with metadata preservation
- **Performance**: Sub-second queries, ~43 embeddings/second processing

## Configuration Options

### Document Processing
```python
# Academic papers (recommended)
pipeline = DocumentSearchPipeline(
    use_grobid=True,
    chunk_size=1000,
    chunk_overlap=150
)

# General documents
pipeline = DocumentSearchPipeline(
    use_grobid=False,
    chunk_size=500,
    chunk_overlap=100
)
```

### Question Answering Models
```python
# High-quality responses (recommended)
qa_system = GroqRAGSystem(model_name="llama-3.1-70b-versatile")

# Fast responses
qa_system = GroqRAGSystem(model_name="llama-3.1-8b-instant")
```

## Docker Integration

### Grobid Server (Optional)
For enhanced academic PDF processing:

```bash
# Start Grobid server
docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0

# Check status
docker ps | grep grobid

# Stop server
docker stop grobid-server && docker rm grobid-server
```

## API Integration

### Environment Configuration
1. Copy the template: `copy .env.template .env`
2. Edit `.env` file and add your API key:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```
3. Get free API key from https://console.groq.com/
4. Usage limits: 14,400 requests per day (free tier)

### Configuration Options
The `.env` file supports these settings:
```bash
GROQ_API_KEY=your_api_key_here     # Required for AI Q&A
DEFAULT_MODEL=llama-3.1-8b-instant # Optional: Default model
MAX_TOKENS=1024                     # Optional: Response length
TEMPERATURE=0.1                     # Optional: Response creativity
```

### Available Models
- **llama-3.1-70b-versatile**: Best for complex reasoning
- **llama-3.1-8b-instant**: Fastest responses
- **mixtral-8x7b-32768**: Excellent for academic content

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
- Optional: Docker for Grobid PDF processing
- Optional: Groq API key for AI question answering

### Installation
```bash
pip install -r requirements.txt
```

Core dependencies:
- chromadb: Vector database
- sentence-transformers: Semantic embeddings
- torch: ML framework (CPU version)
- transformers: Model loading
- requests: API communication
- pypdf2: PDF processing