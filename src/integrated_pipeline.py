"""
Integrated Document Processing Pipeline

Connects DocumentLoader → EmbeddingGenerator → VectorStore for complete
semantic document search functionality.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from document_loader import DocumentLoader, DocumentChunk
from embedding_generator import EmbeddingGenerator  
from vector_store import VectorStore


class DocumentSearchPipeline:
    """
    Integrated pipeline for document processing and semantic search.
    
    Workflow:
    1. DocumentLoader: Load and chunk documents (with Grobid for PDFs)
    2. EmbeddingGenerator: Generate semantic embeddings (single model instance)
    3. VectorStore: Store embeddings in ChromaDB for search
    4. Search: Query interface for semantic document retrieval
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        use_grobid: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ):
        """
        Initialize the integrated pipeline.
        
        Args:
            persist_directory: ChromaDB storage directory
            collection_name: ChromaDB collection name
            use_grobid: Enable Grobid for academic PDF processing
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
        """
        print("🚀 Initializing Integrated Document Search Pipeline")
        print("="*60)
        
        # Initialize components with dependency injection
        print("1. Initializing DocumentLoader...")
        self.document_loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=200,
            use_grobid=use_grobid,
            preserve_academic_structure=True
        )
        
        print("2. Initializing EmbeddingGenerator...")
        self.embedding_generator = EmbeddingGenerator(show_progress=True)
        
        print("3. Initializing VectorStore with dependency injection...")
        self.vector_store = VectorStore(
            embedding_generator=self.embedding_generator,  # Inject generator
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        print("✅ Pipeline initialization complete!")
        print("="*60)
    
    def process_documents_directory(self, documents_dir: str) -> Dict:
        """
        Process all documents in a directory through the complete pipeline.
        
        Args:
            documents_dir: Directory containing documents to process
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n📂 Processing documents from: {documents_dir}")
        
        # Step 1: Load and chunk documents
        print("\nStep 1: Loading and chunking documents...")
        documents = self.document_loader.load_directory(documents_dir)
        
        if not documents:
            print(f"❌ No supported documents found in {documents_dir}")
            return {"error": "No documents found"}
        
        chunks = self.document_loader.process_documents(documents)
        
        if not chunks:
            print("❌ No chunks generated from documents")
            return {"error": "No chunks generated"}
        
        print(f"✅ Generated {len(chunks)} chunks from {len(documents)} documents")
        
        # Step 2: Generate embeddings (using single model instance)
        print("\\nStep 2: Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        print(f"✅ Generated {len(embeddings)} embeddings ({self.embedding_generator.embedding_dim}-dimensional)")
        
        # Step 3: Store in vector database
        print("\\nStep 3: Storing in vector database...")
        added_count = self.vector_store.add_documents_with_embeddings(chunks, embeddings)
        
        # Display statistics
        print("\\nStep 4: Processing complete!")
        self.display_pipeline_stats()
        
        return {
            "documents_processed": len(documents),
            "chunks_generated": len(chunks),
            "embeddings_generated": len(embeddings),
            "documents_stored": added_count,
            "success": True
        }
    
    def search_documents(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: float = 0.0
    ) -> Dict:
        """
        Search documents using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Dictionary with search results
        """
        print(f"\\n🔍 Searching for: '{query}'")
        
        if similarity_threshold > 0:
            results = self.vector_store.search_by_similarity_threshold(
                query=query,
                threshold=similarity_threshold,
                max_results=n_results
            )
        else:
            results = self.vector_store.search(
                query=query,
                n_results=n_results
            )
        
        print(f"✅ Found {len(results['results'])} results")
        
        return results
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0):
        """
        Alias for search_documents to maintain compatibility with RAG components
        """
        return self.search_documents(query, n_results=top_k, similarity_threshold=similarity_threshold)
    
    def display_pipeline_stats(self):
        """Display comprehensive pipeline statistics."""
        print("\\n" + "="*60)
        print("INTEGRATED PIPELINE STATISTICS")
        print("="*60)
        
        # Document Loader stats
        print("📄 Document Processing:")
        print(f"   Supported formats: {', '.join(self.document_loader.supported_extension)}")
        print(f"   Grobid enabled: {self.document_loader.use_grobid}")
        print(f"   Chunk size: {self.document_loader.chunk_size}")
        print(f"   Chunk overlap: {self.document_loader.chunk_overlap}")
        
        # Embedding Generator stats  
        print("\\n🧠 Embedding Generation:")
        print(f"   Model: {self.embedding_generator.model_name}")
        print(f"   Dimension: {self.embedding_generator.embedding_dim}")
        print(f"   Device: {self.embedding_generator.device}")
        print(f"   Total embeddings: {self.embedding_generator.total_embeddings_generated}")
        
        if self.embedding_generator.total_processing_time > 0:
            throughput = self.embedding_generator.total_embeddings_generated / self.embedding_generator.total_processing_time
            print(f"   Throughput: {throughput:.1f} embeddings/second")
        
        # Vector Store stats
        print("\\n🗄️ Vector Storage:")
        vs_stats = self.vector_store.get_collection_stats()
        print(f"   Collection: {vs_stats['collection_name']}")
        print(f"   Total documents: {vs_stats['total_documents']}")
        print(f"   Storage location: {vs_stats['persist_directory']}")
        print(f"   Has data: {vs_stats['has_data']}")
        
        print("="*60)
    
    def interactive_search(self):
        """Interactive search interface."""
        print("\\n🔍 Interactive Search Mode")
        print("Type 'quit' to exit, 'stats' for statistics")
        print("-" * 40)
        
        while True:
            try:
                query = input("\\nEnter search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                    
                if query.lower() in ['stats', 'statistics']:
                    self.display_pipeline_stats()
                    continue
                    
                if not query:
                    print("Please enter a query.")
                    continue
                
                # Search with default parameters
                results = self.search_documents(query, n_results=3)
                
                # Display results
                if results['results']:
                    print(f"\\nTop {len(results['results'])} results:")
                    for i, result in enumerate(results['results'], 1):
                        print(f"\\n{i}. Score: {result['similarity']:.4f}")
                        print(f"   Text: {result['text'][:200]}...")
                        if result['metadata']:
                            print(f"   Metadata: {result['metadata']}")
                else:
                    print("No results found.")
                    
            except KeyboardInterrupt:
                print("\\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def get_collection_info(self) -> Dict:
        """Get information about the current document collection."""
        return self.vector_store.get_collection_stats()
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        print("🗑️ Clearing document collection...")
        self.vector_store.delete_collection()
        print("✅ Collection cleared!")


def main():
    """Demo of the integrated pipeline."""
    print("🔬 Integrated Document Search Pipeline - Demo")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = DocumentSearchPipeline(
            persist_directory="./data/demo_chroma_db",
            collection_name="demo_documents",
            use_grobid=True
        )
        
        # Check if documents directory exists
        documents_dir = "documents"
        if Path(documents_dir).exists():
            # Process documents
            results = pipeline.process_documents_directory(documents_dir)
            
            if results.get("success"):
                print(f"\\n✅ Successfully processed {results['documents_processed']} documents")
                
                # Start interactive search
                pipeline.interactive_search()
            else:
                print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
        else:
            print(f"\\n📁 Documents directory '{documents_dir}' not found.")
            print(f"Create the directory and add PDF/TXT/MD files to test the pipeline.")
            
            # Demo with sample data
            print("\\n🧪 Running demo with sample data...")
            demo_pipeline(pipeline)
    
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()


def demo_pipeline(pipeline: DocumentSearchPipeline):
    """Demo with sample chunks."""
    print("Creating sample document chunks...")
    
    sample_chunks = [
        DocumentChunk(
            text="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            metadata={"source": "ml_guide", "section": "introduction", "page": 1},
            chunk_id="ml_guide_001"
        ),
        DocumentChunk(
            text="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            metadata={"source": "dl_basics", "section": "overview", "page": 1},
            chunk_id="dl_basics_001"
        ),
        DocumentChunk(
            text="Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            metadata={"source": "nlp_intro", "section": "definition", "page": 1},
            chunk_id="nlp_intro_001"
        )
    ]
    
    # Generate embeddings and store
    print("\\nGenerating embeddings...")
    embeddings = pipeline.embedding_generator.generate_embeddings(sample_chunks)
    
    print("\\nStoring in vector database...")
    pipeline.vector_store.add_documents_with_embeddings(sample_chunks, embeddings)
    
    # Demo searches
    demo_queries = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain machine learning"
    ]
    
    print("\\n🔍 Demo searches:")
    for query in demo_queries:
        print(f"\\nQuery: '{query}'")
        results = pipeline.search_documents(query, n_results=2)
        
        for i, result in enumerate(results['results'], 1):
            print(f"  {i}. Score: {result['similarity']:.3f} - {result['text'][:80]}...")


if __name__ == "__main__":
    main()