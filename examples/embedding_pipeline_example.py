"""
Example: Document Processing Pipeline with Embedding Generation

This example demonstrates how to:
1. Load and chunk documents using DocumentLoader with Grobid
2. Generate embeddings using EmbeddingGenerator
3. Store embeddings for semantic search
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from document_loader import DocumentLoader, DocumentChunk
from embedding_generator import EmbeddingGenerator

def test_full_pipeline():
    """Test the complete document processing pipeline."""
    print("="*60)
    print("DOCUMENT PROCESSING PIPELINE TEST")
    print("="*60)
    
    # Initialize components
    print("1. Initializing Document Loader and Embedding Generator...")
    doc_loader = DocumentLoader()
    embedding_gen = EmbeddingGenerator(show_progress=True)
    
    # Check if we have a sample PDF
    sample_pdf = Path("../documents/sample_ml.txt")  # Using .txt for simplicity in this example
    if sample_pdf.exists():
        print(f"2. Loading document: {sample_pdf}")
        chunks = doc_loader.load_document(sample_pdf)
        print(f"   Loaded {len(chunks)} chunks")
    else:
        print("2. Creating sample text chunks...")
        # Create sample chunks for demonstration
        chunks = [
            DocumentChunk(
                text="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                metadata={"section": "introduction", "source": "sample", "page": 1},
                chunk_id="chunk_001"
            ),
            DocumentChunk(
                text="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
                metadata={"section": "deep_learning", "source": "sample", "page": 2},
                chunk_id="chunk_002"
            ),
            DocumentChunk(
                text="Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
                metadata={"section": "nlp", "source": "sample", "page": 3},
                chunk_id="chunk_003"
            ),
            DocumentChunk(
                text="The transformer architecture has revolutionized natural language processing. It relies entirely on attention mechanisms to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
                metadata={"section": "transformers", "source": "sample", "page": 4},
                chunk_id="chunk_004"
            ),
            DocumentChunk(
                text="Semantic search refers to search functionality that understands the intent and contextual meaning of search terms, rather than just matching keywords. This is achieved through vector embeddings and similarity calculations.",
                metadata={"section": "semantic_search", "source": "sample", "page": 5},
                chunk_id="chunk_005"
            )
        ]
    
    print(f"   Processing {len(chunks)} chunks")
    
    # Display chunk information
    print("\\n3. Document Chunks Preview:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"   Chunk {i+1} ({chunk.chunk_id}):")
        print(f"     Text: {chunk.text[:100]}...")
        print(f"     Metadata: {chunk.metadata}")
    
    # Generate embeddings
    print("\\n4. Generating embeddings...")
    embeddings = embedding_gen.generate_embeddings(chunks)
    
    print(f"   Generated {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
    
    # Validate embeddings
    validation = embedding_gen.validate_embeddings(embeddings)
    print(f"\\n5. Embedding Validation:")
    print(f"   Valid: {validation['valid']}")
    print(f"   All normalized: {validation['normalized']}")
    if validation['errors']:
        print(f"   Errors: {len(validation['errors'])}")
        for error in validation['errors'][:3]:  # Show first 3 errors
            print(f"     - {error}")
    
    # Test semantic similarity
    print("\\n6. Testing Semantic Similarity:")
    test_queries = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain attention mechanisms"
    ]
    
    for query in test_queries:
        print(f"\\n   Query: '{query}'")
        similarities = []
        
        for i, chunk in enumerate(chunks):
            similarity = embedding_gen.compare_embeddings(query, chunk.text)
            similarities.append((i, chunk.chunk_id, similarity, chunk.text[:50]))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        print("   Top 2 most similar chunks:")
        for j, (idx, chunk_id, sim, text_preview) in enumerate(similarities[:2]):
            print(f"     {j+1}. {chunk_id} (similarity: {sim:.3f})")
            print(f"        Text: {text_preview}...")
    
    # Display final statistics
    print("\\n7. Performance Statistics:")
    embedding_gen.display_embedding_stats()
    
    print("\\n" + "="*60)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return chunks, embeddings

def demonstrate_batch_processing():
    """Demonstrate batch processing with larger datasets."""
    print("\\n" + "="*60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Create a larger dataset for batch processing test
    print("Creating large dataset for batch processing test...")
    
    base_texts = [
        "Machine learning algorithms can learn from data",
        "Deep neural networks process information in layers", 
        "Natural language processing understands human text",
        "Computer vision analyzes visual information",
        "Reinforcement learning learns through rewards",
        "Supervised learning uses labeled training data",
        "Unsupervised learning finds hidden patterns",
        "Transfer learning adapts pre-trained models",
        "Attention mechanisms focus on relevant information",
        "Transformer models revolutionized AI"
    ]
    
    # Create 100 chunks by combining and modifying base texts
    large_chunks = []
    for i in range(100):
        base_idx = i % len(base_texts)
        text = f"{base_texts[base_idx]} in various applications and research contexts. " \
               f"This demonstrates the versatility and importance of these concepts in AI. " \
               f"Research paper #{i+1} explores these topics in depth."
        
        chunk = DocumentChunk(
            text=text,
            metadata={
                "section": f"section_{(i//10)+1}",
                "paper_id": f"paper_{i+1:03d}",
                "category": ["ML", "DL", "NLP", "CV", "RL"][i % 5]
            },
            chunk_id=f"batch_chunk_{i+1:03d}"
        )
        large_chunks.append(chunk)
    
    print(f"Created {len(large_chunks)} chunks for batch processing")
    
    # Test with different batch sizes
    embedding_gen = EmbeddingGenerator(
        cpu_batch_size=16,
        gpu_batch_size=32,
        show_progress=True
    )
    
    print("\\nGenerating embeddings for large dataset...")
    import time
    start_time = time.time()
    
    embeddings = embedding_gen.generate_embeddings(large_chunks, batch_size=16)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processed {len(embeddings)} chunks in {processing_time:.2f} seconds")
    print(f"Throughput: {len(embeddings)/processing_time:.1f} chunks/second")
    
    # Validate batch results
    validation = embedding_gen.validate_embeddings(embeddings)
    print(f"\\nBatch processing validation:")
    print(f"  All embeddings valid: {validation['valid']}")
    print(f"  Total embeddings: {validation['count']}")
    print(f"  Embedding dimension: {validation['dimension']}")
    
    embedding_gen.display_embedding_stats()
    
    return large_chunks, embeddings

if __name__ == "__main__":
    print("Running Document Processing Pipeline Example...")
    
    try:
        # Test basic pipeline
        chunks, embeddings = test_full_pipeline()
        
        # Test batch processing
        large_chunks, large_embeddings = demonstrate_batch_processing()
        
        print("\\n All tests completed successfully!")
        print(f"   Processed {len(chunks) + len(large_chunks)} total chunks")
        print(f"   Generated {len(embeddings) + len(large_embeddings)} total embeddings")
        
    except Exception as e:
        print(f"\\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)