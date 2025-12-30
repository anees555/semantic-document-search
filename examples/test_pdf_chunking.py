#!/usr/bin/env python3
"""
Test PDF Chunking with Grobid

This script demonstrates PDF processing and chunking with Grobid support.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from document_loader import DocumentLoader

def main():
    print("🔬 PDF Chunking Test with Grobid")
    print("=" * 50)
    
    # Initialize loader optimized for academic papers
    loader = DocumentLoader(
        chunk_size=1000,          # Larger chunks for academic content
        chunk_overlap=150,        # More overlap for context
        min_chunk_size=200,       # Minimum chunk size
        use_grobid=True,          # Enable Grobid
        preserve_academic_structure=True  # Structure-aware chunking
    )
    
    # Check Grobid status
    status = loader.get_grobid_status()
    print(f"Grobid Server Status: {'✅ Running' if status['server_available'] else '❌ Not Available'}")
    
    if not status['server_available']:
        print("⚠️  Please start Grobid server first:")
        print("   docker run -d --name grobid-server -p 8070:8070 lfoppiano/grobid:0.8.0")
        return
    
    # Look for PDF files
    documents_dir = Path("documents")
    pdf_files = list(documents_dir.glob("*.pdf")) if documents_dir.exists() else []
    
    if not pdf_files:
        print(f"\n📁 No PDF files found in 'documents/' folder")
        print(f"   Please add some PDF files (preferably research papers) to test Grobid processing")
        
        # Demo with any files available
        if documents_dir.exists():
            print(f"\n🔍 Processing available files...")
            documents = loader.load_directory("documents")
            if documents:
                process_and_show_results(loader, documents)
    else:
        print(f"\n📄 Found {len(pdf_files)} PDF file(s):")
        for pdf_file in pdf_files:
            print(f"   • {pdf_file.name}")
        
        print(f"\n🔄 Processing PDFs with Grobid...")
        documents = loader.load_directory("documents")
        process_and_show_results(loader, documents)

def process_and_show_results(loader, documents):
    """Process documents and show detailed results"""
    if not documents:
        print("No documents to process")
        return
    
    # Process documents
    chunks = loader.process_documents(documents)
    
    # Show statistics
    loader.display_statistics(chunks)
    
    # Show examples of different chunk types
    show_chunk_examples(chunks)

def show_chunk_examples(chunks):
    """Show examples of different types of chunks"""
    print(f"\n📊 Chunk Examples:")
    print("-" * 40)
    
    # Group chunks by processing method
    grobid_chunks = [c for c in chunks if 'grobid' in c.metadata.get('processing_method', '')]
    regular_chunks = [c for c in chunks if 'grobid' not in c.metadata.get('processing_method', '')]
    
    print(f"Grobid-processed chunks: {len(grobid_chunks)}")
    print(f"Regular chunks: {len(regular_chunks)}")
    
    # Show examples from each type
    if grobid_chunks:
        print(f"\n🔬 Grobid Chunk Example:")
        chunk = grobid_chunks[0]
        print(f"   File: {chunk.metadata.get('filename')}")
        print(f"   Section Type: {chunk.metadata.get('section_type', 'unknown')}")
        print(f"   Size: {len(chunk.text)} characters")
        print(f"   Preview: {chunk.text[:300]}...")
        
        # Show section types
        section_types = {}
        for chunk in grobid_chunks:
            section_type = chunk.metadata.get('section_type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        print(f"\n📚 Section Types Found:")
        for section_type, count in section_types.items():
            print(f"   • {section_type.title()}: {count} chunks")
    
    if regular_chunks:
        print(f"\n📄 Regular Chunk Example:")
        chunk = regular_chunks[0]
        print(f"   File: {chunk.metadata.get('filename')}")
        print(f"   Size: {len(chunk.text)} characters")
        print(f"   Preview: {chunk.text[:300]}...")

def demo_configuration_options():
    """Show different configuration options"""
    print(f"\n⚙️  Configuration Options:")
    print("-" * 40)
    
    configs = [
        {
            "name": "Academic Papers",
            "settings": {
                "chunk_size": 1000,
                "use_grobid": True,
                "preserve_academic_structure": True
            }
        },
        {
            "name": "General Documents", 
            "settings": {
                "chunk_size": 500,
                "use_grobid": False,
                "preserve_academic_structure": False
            }
        }
    ]
    
    for config in configs:
        print(f"\n📋 {config['name']}:")
        for key, value in config['settings'].items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    try:
        main()
        demo_configuration_options()
    except KeyboardInterrupt:
        print(f"\n❌ Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"\nMake sure:")
        print(f"1. Grobid server is running (docker ps)")
        print(f"2. Documents folder exists with PDF files")
        print(f"3. Virtual environment is activated")