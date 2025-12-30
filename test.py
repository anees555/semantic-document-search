#!/usr/bin/env python3
"""
Quick Test Script for PDF Chunking

Test the document loader functionality directly from project root.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from document_loader import DocumentLoader

def quick_test():
    """Quick test of document processing functionality"""
    print(" Quick PDF Chunking Test")
    print("=" * 40)
    
    # Initialize loader
    loader = DocumentLoader(
        chunk_size=800,
        use_grobid=True,
        preserve_academic_structure=True
    )
    
    # Check Grobid status
    status = loader.get_grobid_status()
    print(f"Grobid Server: {' Running' if status['server_available'] else '❌ Not Available'}")
    
    # Process documents
    documents = loader.load_directory("documents")
    if documents:
        chunks = loader.process_documents(documents)
        loader.display_statistics(chunks)
        
        # Show first chunk as example
        if chunks:
            print(f"\n First Chunk Example:")
            chunk = chunks[0]
            print(f"File: {chunk.metadata.get('filename')}")
            print(f"Method: {chunk.metadata.get('processing_method')}")
            print(f"Content: {chunk.text[:300]}...")
    else:
        print(f"No documents found in 'documents/' folder")

if __name__ == "__main__":
    quick_test()