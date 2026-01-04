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


def main():
    """Main application entry point with integrated pipeline"""
    print(" Semantic Document Search - Integrated Pipeline")
    print("=" * 60)
    
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
            
            # Start interactive search
            print(f"\n Starting interactive search...")
            pipeline.interactive_search()
        else:
            print(f" Processing failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f" Application error: {str(e)}")
        import traceback
        traceback.print_exc()


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