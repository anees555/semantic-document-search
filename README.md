# Semantic Document Search

A document question-answering system using Retrieval Augmented Generation (RAG) with enhanced PDF processing via Grobid.



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

### 3. Process Documents
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
│   ├── document_loader.py    # Enhanced PDF processing with Grobid
│   └── vector_store.py       # ChromaDB vector operations
├── documents/               # Input documents (PDF, TXT, MD)
├── examples/               # Demo scripts
├── app.py                 # Main application
├── requirements.txt       # Dependencies
└── README.md             
```

## 🔬 Features

- **Enhanced PDF Processing**: Grobid integration for academic papers
- **Structure-Aware Chunking**: Preserves document hierarchy 
- **Multi-Format Support**: PDF, TXT, Markdown files
- **Academic Section Detection**: Abstract, Methods, Results, etc.
- **Automatic Fallback**: PyPDF2 backup if Grobid unavailable

##  Usage

### Basic Document Processing
```python
from src.document_loader import DocumentLoader

# Initialize with Grobid support
loader = DocumentLoader(
    chunk_size=1000,
    use_grobid=True,
    preserve_academic_structure=True
)

# Process documents
documents = loader.load_directory("documents/")
chunks = loader.process_documents(documents)
loader.display_statistics(chunks)
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
python examples/test_pdf_chunking.py    # Test PDF processing
python examples/grobid_example.py       # Grobid demonstration
python examples/setup_grobid.py         # Setup helper
```

##  Requirements

- Python 3.8+
- Docker (for Grobid)
- Dependencies in requirements.txt