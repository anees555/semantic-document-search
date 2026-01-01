# Semantic Document Search

A document question-answering system using Retrieval Augmented Generation (RAG) with enhanced PDF processing via Grobid and semantic embedding generation.

##  Completed Features

1. ** Enhanced PDF Processing**: Grobid integration for academic papers with structure-aware chunking
2. ** Embedding Generation**: sentence-transformers with all-MiniLM-L6-v2 model (384-dimensional vectors)
3. ** Vector Store Integration**: ChromaDB setup (in progress)
4. ** RAG Q&A System**: Question-answering interface (pending)



### 1. Setup Environment
```bash
# Create and activate virtual environment  
python -m venv vectorenv
vectorenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Grobid Server (For PDF Processing)
```bash
# Using Docker
docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0
```

### 3. Generate Embeddings
```python
from src.embedding_generator import EmbeddingGenerator

# Initialize embedding generator
generator = EmbeddingGenerator()

# Generate 384-dimensional vectors
embeddings = generator.generate_embeddings(chunks)
print(f"Generated {len(embeddings)} embeddings")
```

### 4. Process Documents
```python
from src.document_loader import DocumentLoader

loader = DocumentLoader(chunk_size=1000, use_grobid=True)
documents = loader.load_directory("documents/")
chunks = loader.process_documents(documents)
```

## 📁 Project Structure

```
semantic-document-search/
├── src/
│   ├── document_loader.py     # Enhanced PDF processing with Grobid
│   ├── embedding_generator.py # Semantic embeddings with sentence-transformers
│   └── vector_store.py        # ChromaDB vector operations
├── documents/                # Input documents (PDF, TXT, MD)
├── examples/                 # Demo scripts and pipeline examples
├── app.py                    # Main application
├── requirements.txt          # Dependencies
└── README.md             
```

##  Features

- **Enhanced PDF Processing**: Grobid integration for academic papers
- **Semantic Embeddings**: 384-dimensional vectors using all-MiniLM-L6-v2
- **Structure-Aware Chunking**: Preserves document hierarchy 
- **Multi-Format Support**: PDF, TXT, Markdown files
- **Academic Section Detection**: Abstract, Methods, Results, etc.
- **Automatic Fallback**: PyPDF2 backup if Grobid unavailable
- **Batch Processing**: Optimized embedding generation with progress tracking

##  Usage

### Document Processing + Embedding Pipeline
```python
from src.document_loader import DocumentLoader
from src.embedding_generator import EmbeddingGenerator

# Initialize components
loader = DocumentLoader(
    chunk_size=1000,
    use_grobid=True,
    preserve_academic_structure=True
)
generator = EmbeddingGenerator()

# Process documents and generate embeddings
documents = loader.load_directory("documents/")
chunks = loader.process_documents(documents)
embeddings = generator.generate_embeddings(chunks)

# Display results
loader.display_statistics(chunks)
generator.display_embedding_stats()
```

### Configuration Options
```python
# For Academic Papers
loader = DocumentLoader(
    chunk_size=1000,
    use_grobid=True,
    preserve_academic_structure=True
)

# For General Documents  
loader = DocumentLoader(
    chunk_size=500,
    use_grobid=False
)
```

##  Docker Commands

```bash
# Start Grobid server
docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0

# Check status
docker ps

# Stop server
docker stop grobid-server
```

##  Academic Section Types

- **Abstract**: Paper summary
- **Introduction**: Background  
- **Methods**: Methodology
- **Results**: Findings
- **Discussion**: Analysis
- **Conclusion**: Summary
- **References**: Citations

##  Examples

Run demo scripts:
```bash
python examples/test_pdf_chunking.py           # Test PDF processing
python examples/grobid_example.py              # Grobid demonstration  
python examples/embedding_pipeline_example.py  # Complete processing pipeline
python examples/setup_grobid.py                # Setup helper
```

##  Requirements

- Python 3.8+
- Docker (for Grobid)
- Dependencies: sentence-transformers, torch, chromadb, PyPDF2, requests
- See requirements.txt for complete list

##  Performance

- **Document Processing**: ~90-95% success rate with Grobid for academic papers
- **Embedding Generation**: ~100-240 embeddings/second (CPU), ~1000+/second (GPU)
- **Model**: all-MiniLM-L6-v2 (384 dimensions, L2-normalized)
- **Memory Usage**: ~400MB for embedding model