#!/usr/bin/env python3
"""
Grobid-Enhanced Document Processing Example

This example demonstrates how to use the enhanced DocumentLoader 
with Grobid support for processing academic papers and research documents.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from document_loader import DocumentLoader, AcademicSection

def main():
    print("🔬 Grobid-Enhanced Document Processing Demo")
    print("=" * 60)
    
    # Initialize loader with Grobid support
    loader = DocumentLoader(
        chunk_size=1000,           # Larger chunks for academic content
        chunk_overlap=150,         # More overlap for better context
        min_chunk_size=100,
        grobid_server_url="http://localhost:8070",  # Default Grobid server
        use_grobid=True,           # Enable Grobid processing
        preserve_academic_structure=True  # Maintain paper structure
    )
    
    # Check Grobid status
    loader.print_grobid_status()
    
    # Demo 1: Process documents directory
    documents_dir = "documents"
    if os.path.exists(documents_dir):
        print(f"\n📁 Processing documents from '{documents_dir}/'...")
        
        # Load all documents
        documents = loader.load_directory(documents_dir)
        
        if documents:
            print(f"Found {len(documents)} document(s)")
            
            # Process into chunks
            all_chunks = loader.process_documents(documents)
            
            # Display comprehensive statistics
            loader.display_statistics(all_chunks)
            
            # Show examples of different chunk types
            demo_chunk_types(all_chunks)
            
        else:
            print("No supported documents found!")
    else:
        print(f"📁 Documents directory '{documents_dir}' not found")
        print("   Create the directory and add PDF research papers to test Grobid!")
    
    # Demo 2: Show configuration options
    demonstrate_configurations()
    
    print("\n✅ Demo completed!")

def demo_chunk_types(chunks):
    """Demonstrate different types of chunks created"""
    print(f"\n📊 Chunk Analysis:")
    print("-" * 40)
    
    # Group chunks by processing method
    grobid_chunks = [c for c in chunks if 'grobid' in c.metadata.get('processing_method', '')]
    regular_chunks = [c for c in chunks if 'grobid' not in c.metadata.get('processing_method', '')]
    
    print(f"Grobid-processed chunks: {len(grobid_chunks)}")
    print(f"Regular chunks: {len(regular_chunks)}")
    
    # Show academic section types
    if grobid_chunks:
        section_types = {}
        for chunk in grobid_chunks:
            section_type = chunk.metadata.get('section_type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        print(f"\n📚 Academic Section Types Found:")
        for section_type, count in sorted(section_types.items()):
            print(f"  • {section_type.title()}: {count} chunks")
    
    # Show sample chunks
    print(f"\n📄 Sample Chunks:")
    sample_count = min(3, len(chunks))
    for i, chunk in enumerate(chunks[:sample_count]):
        print(f"\n--- Sample {i+1} ---")
        print(f"File: {chunk.metadata.get('filename', 'Unknown')}")
        print(f"Type: {chunk.metadata.get('section_type', 'unknown')}")
        print(f"Method: {chunk.metadata.get('processing_method', 'unknown')}")
        print(f"Size: {len(chunk.text)} characters")
        print(f"Preview: {chunk.text[:150]}...")

def demonstrate_configurations():
    """Show different configuration options"""
    print(f"\n⚙️  Configuration Examples:")
    print("-" * 40)
    
    configs = [
        {
            "name": "Academic Papers (Recommended)",
            "config": {
                "chunk_size": 1000,
                "chunk_overlap": 150,
                "use_grobid": True,
                "preserve_academic_structure": True
            },
            "description": "Best for research papers, scientific articles"
        },
        {
            "name": "General Documents",
            "config": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "use_grobid": False,
                "preserve_academic_structure": False
            },
            "description": "For general PDFs, books, articles"
        },
        {
            "name": "Large Context Analysis",
            "config": {
                "chunk_size": 2000,
                "chunk_overlap": 200,
                "use_grobid": True,
                "preserve_academic_structure": True
            },
            "description": "For detailed analysis requiring more context"
        }
    ]
    
    for config in configs:
        print(f"\n🔧 {config['name']}:")
        print(f"   Description: {config['description']}")
        print(f"   Settings: {config['config']}")

def setup_grobid_instructions():
    """Display Grobid setup instructions"""
    print(f"\n🐳 Grobid Setup Instructions:")
    print("-" * 40)
    print("1. Using Docker (Recommended):")
    print("   docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
    print()
    print("2. Manual Installation:")
    print("   git clone https://github.com/kermitt2/grobid.git")
    print("   cd grobid")
    print("   ./gradlew run")
    print()
    print("3. Verify Installation:")
    print("   curl http://localhost:8070/api/isalive")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        setup_grobid_instructions()