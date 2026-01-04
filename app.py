#!/usr/bin/env python3
"""
Semantic Document Search - Main Application

Integrated pipeline with DocumentLoader → EmbeddingGenerator → VectorStore
for complete semantic document search functionality.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from integrated_pipeline import DocumentSearchPipeline
from rag_qa_pipeline import RAGQuestionAnsweringPipeline


def main():
    """Main application entry point with integrated pipeline and RAG Q&A"""
    print("🔍 Semantic Document Search - Enhanced with RAG Q&A")
    print("=" * 60)
    
    # Check for command line arguments
    mode = 'search'  # default mode
    if len(sys.argv) > 1:
        if sys.argv[1] in ['qa', 'rag', 'question']:
            mode = 'qa'
        elif sys.argv[1] in ['search', 'semantic']:
            mode = 'search'
        elif sys.argv[1] in ['help', '--help', '-h']:
            display_help()
            return
    
    print(f"\n📋 Mode: {'RAG Q&A' if mode == 'qa' else 'Semantic Search'}")
    
    # Initialize integrated pipeline
    try:
        pipeline = DocumentSearchPipeline(
            persist_directory="./data/chroma_db",
            collection_name="semantic_documents", 
            use_grobid=True,
            chunk_size=1000,
            chunk_overlap=150
        )
        
        print_system_status()
        
        # Process documents
        documents_dir = "documents"
        if not os.path.exists(documents_dir):
            print(f"\n Creating '{documents_dir}' directory...")
            os.makedirs(documents_dir)
            print(f"   Add your PDF, TXT, or MD files to '{documents_dir}/' and run again.")
            return
        
        # Check for documents
        document_files = list(Path(documents_dir).glob("*.*"))
        supported_extensions = {'.pdf', '.txt', '.md'}
        valid_files = [f for f in document_files if f.suffix.lower() in supported_extensions]
        
        if not valid_files:
            print(f"\n No supported documents found in '{documents_dir}/'")
            print(f"   Supported formats: PDF, TXT, MD")
            print(f"   Add documents and run again.")
            return
        
        print(f"\n Found {len(valid_files)} supported documents:")
        for file in valid_files:
            print(f"   - {file.name} ({file.suffix.upper()})")
        
        # Process documents through integrated pipeline
        print(f"\n Processing documents through integrated pipeline...")
        results = pipeline.process_documents_directory(documents_dir)
        
        if results.get("success"):
            print(f"\nPipeline processing complete!")
            print(f"   Documents: {results['documents_processed']}")
            print(f"   Chunks: {results['chunks_generated']} ")
            print(f"   Embeddings: {results['embeddings_generated']}")
            print(f"   Stored: {results['documents_stored']}")
            
            if mode == 'qa':
                # Initialize and start RAG Q&A system
                print(f"\n🤖 Initializing RAG Q&A system...")
                try:
                    rag_pipeline = RAGQuestionAnsweringPipeline(
                        document_pipeline=pipeline,
                        max_context_tokens=3000,
                        use_gpu=True,
                        citation_style="academic"
                    )
                    print(f"\n🚀 Starting interactive Q&A session...")
                    rag_pipeline.interactive_qa_session()
                except Exception as e:
                    print(f"❌ RAG Q&A initialization failed: {e}")
                    print(f"\n🔄 Falling back to semantic search mode...")
                    pipeline.interactive_search()
            else:
                # Start interactive search
                print(f"\n🔍 Starting interactive search...")
                pipeline.interactive_search()
        else:
            print(f" Processing failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f" Application error: {str(e)}")
        import traceback
        traceback.print_exc()


def display_help():
    """Display help information"""
    help_text = """
 Semantic Document Search - Enhanced with RAG Q&A
==================================================

USAGE:
  python app.py [mode]

MODES:
  search, semantic  - Semantic document search (default)
  qa, rag, question - RAG question-answering system
  help, --help, -h  - Display this help message

FEATURES:
📚 Document Processing:
  • Enhanced PDF processing with Grobid integration
  • Multi-format support (PDF, TXT, Markdown)
  • Structure-aware chunking for academic papers
  • ChromaDB vector storage with embeddings

 Semantic Search Mode:
  • Interactive semantic search interface
  • Similarity scoring and ranking
  • Real-time document retrieval

 RAG Q&A Mode:
  • Intelligent question answering
  • Academic citation formatting
  • Confidence scoring and source attribution
  • Free local LLM models (Phi-3, Mistral, etc.)

REQUIREMENTS:
  • Python 3.8+ with required packages
  • Optional: Docker with Grobid for enhanced PDF processing
  • Optional: GPU for faster LLM inference

EXAMPLES:
  python app.py search    # Semantic search mode
  python app.py qa        # Q&A mode
  python app.py help      # Show this help

FIRST TIME SETUP:
  1. Add documents to 'documents/' directory
  2. Run the application to process documents
  3. Choose your preferred interaction mode
    """
    print(help_text)


def print_system_status():
    """Print system status and requirements"""
    print("\n System Status:")
    print("-" * 40)
    
    # Check Docker/Grobid
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=grobid", "--format", "table {{.Names}}\\t{{.Status}}"],
            capture_output=True, text=True, timeout=10
        )
        
        if "grobid" in result.stdout:
            print("    Grobid Docker container: Running")
        else:
            print("     Grobid Docker container: Not running")
            print("      Run: docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0")
    except:
        print("    Docker/Grobid status: Unknown (Docker not accessible)")
    
    # Check ChromaDB directory
    chroma_dir = Path("./data/chroma_db")
    if chroma_dir.exists():
        print(f"    ChromaDB directory: {chroma_dir}")
    else:
        print(f"    ChromaDB directory: Will be created at {chroma_dir}")
    
    # Check Python environment  
    try:
        import chromadb, sentence_transformers
        print("    Dependencies: ChromaDB, sentence-transformers available")
    except ImportError as e:
        print(f"    Dependencies missing: {e}")
    
    print("-" * 40)


if __name__ == "__main__":
    main()