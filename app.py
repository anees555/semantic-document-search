#!/usr/bin/env python3
"""
Semantic Document Search - Main Application

Enhanced document processing with Grobid support for PDF chunking.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from document_loader import DocumentLoader
from vector_store import VectorStore

def main():
    """Main application entry point"""
    print("🔬 Semantic Document Search - Enhanced PDF Processing")
    print("=" * 60)
    
    # Initialize document loader with Grobid support
    loader = DocumentLoader(
        chunk_size=1000,           # Optimized for academic content
        chunk_overlap=150,         # Good context preservation
        min_chunk_size=200,        # Minimum meaningful chunk size
        use_grobid=True,          # Enable Grobid for PDF processing
        preserve_academic_structure=True  # Structure-aware chunking
    )
    
    # Check system status
    print_system_status(loader)
    
    # Process documents
    documents_dir = "documents"
    if not os.path.exists(documents_dir):
        print(f"\n Creating '{documents_dir}' directory...")
        os.makedirs(documents_dir)
        print(f"   Add your PDF, TXT, or MD files to '{documents_dir}/' and run again.")
        return
    
    print(f"\n Processing documents from '{documents_dir}/'...")
    documents = loader.load_directory(documents_dir)
    
    if not documents:
        print(f"   No supported documents found in '{documents_dir}/'")
        print(f"   Supported formats: {loader.supported_extension}")
        return
    
    # Process into chunks
    chunks = loader.process_documents(documents)
    
    # Display comprehensive statistics
    loader.display_statistics(chunks)
    
    # Show sample chunks
    show_sample_chunks(chunks)
    
    # Optional: Initialize vector store for search
    setup_vector_store_option(chunks)

def print_system_status(loader):
    """Print current system configuration and status"""
    status = loader.get_grobid_status()
    
    print(f"\n System Status:")
    print(f"   • Grobid Support: {'True' if status['grobid_support'] else 'False'}")
    print(f"   • Grobid Server: {' Running' if status['server_available'] else ' Not Available'}")
    print(f"   • Academic Structure: {' Enabled' if status['preserve_academic_structure'] else ' Disabled'}")
    print(f"   • Supported Formats: PDF, TXT, MD")
    
    if not status['server_available'] and status['use_grobid']:
        print(f"\n To enable Grobid PDF processing:")
        print(f"   docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0")

def show_sample_chunks(chunks, max_samples=3):
    """Display sample chunks from processing"""
    if not chunks:
        return
    
    print(f"\n Sample Chunks:")
    print("-" * 40)
    
    # Group chunks by processing method
    grobid_chunks = [c for c in chunks if 'grobid' in c.metadata.get('processing_method', '')]
    regular_chunks = [c for c in chunks if 'grobid' not in c.metadata.get('processing_method', '')]
    
    sample_count = min(max_samples, len(chunks))
    for i, chunk in enumerate(chunks[:sample_count], 1):
        print(f"\n--- Sample {i} ---")
        print(f"File: {chunk.metadata.get('filename', 'Unknown')}")
        print(f"Type: {chunk.metadata.get('section_type', 'unknown')}")
        print(f"Method: {chunk.metadata.get('processing_method', 'unknown')}")
        print(f"Size: {len(chunk.text)} characters")
        print(f"Preview: {chunk.text[:200]}...")
    
    if grobid_chunks:
        print(f"\n Academic Sections Found:")
        section_types = {}
        for chunk in grobid_chunks:
            section_type = chunk.metadata.get('section_type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        for section_type, count in section_types.items():
            if section_type != 'unknown':
                print(f"   • {section_type.title()}: {count} chunks")

def setup_vector_store_option(chunks):
    """Optionally setup vector store for search capabilities"""
    print(f"\n Vector Store Setup:")
    print(f"   Processed {len(chunks)} chunks ready for vector storage")
    print(f"   To enable search capabilities, chunks can be added to ChromaDB")
    
    # Uncomment below to automatically setup vector store
    # try:
    #     vector_store = VectorStore()
    #     vector_store.add_documents(chunks)
    #     print(f"    Added {len(chunks)} chunks to vector store")
    # except Exception as e:
    #     print(f"     Vector store setup failed: {str(e)}")

def interactive_mode():
    """Interactive mode for document processing"""
    print(f"\n Interactive Mode")
    print(f"Commands:")
    print(f"  'process' - Process all documents in documents/ folder")
    print(f"  'status'  - Show system status")  
    print(f"  'quit'    - Exit")
    
    loader = DocumentLoader(
        chunk_size=1000,
        use_grobid=True,
        preserve_academic_structure=True
    )
    
    while True:
        try:
            command = input(f"\n> ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'status':
                print_system_status(loader)
            elif command == 'process':
                documents = loader.load_directory("documents")
                if documents:
                    chunks = loader.process_documents(documents)
                    loader.display_statistics(chunks)
                else:
                    print(f"No documents found in 'documents/' folder")
            else:
                print(f"Unknown command: {command}")
                
        except KeyboardInterrupt:
            print(f"\n\nGoodbye!")
            break

if __name__ == "__main__":
    try:
        # Check for command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            main()
    except KeyboardInterrupt:
        print(f"\n Operation cancelled by user")
    except Exception as e:
        print(f"\n Error: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"1. Ensure virtual environment is activated")
        print(f"2. Check if documents/ folder exists with files")
        print(f"3. For PDF processing, start Grobid server:")
        print(f"   docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0")