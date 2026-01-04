"""
vector_store.py
Vector database management using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
from dataclasses import dataclass

# Import EmbeddingGenerator for dependency injection
from embedding_generator import EmbeddingGenerator


class VectorStore:
    """
    Manages vector embeddings and similarity search using ChromaDB
    """
    
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents"
    ):
        """
        Initialize VectorStore with dependency injection
        
        Args:
            embedding_generator: EmbeddingGenerator instance for embeddings
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Use injected embedding generator or create default
        if embedding_generator is not None:
            self.embedding_generator = embedding_generator
            print(f"Using injected EmbeddingGenerator: {embedding_generator.model_name}")
        else:
            print("No EmbeddingGenerator provided - creating default instance")
            self.embedding_generator = EmbeddingGenerator()
        
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create new one
        
        Returns:
            ChromaDB collection object
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            count = collection.count()
            print(f"Loaded existing collection '{self.collection_name}' with {count} documents")
        except:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks with embeddings"}
            )
            print(f"Created new collection '{self.collection_name}'")
        
        return collection
    
    
    def add_documents(
        self,
        chunks: List,
        batch_size: int = 100
    ) -> int:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of DocumentChunk objects from document_loader
            batch_size: Number of documents to process at once
            
        Returns:
            Number of documents added
        """
        if not chunks:
            print("No chunks to add")
            return 0
        
        print(f"\nAdding {len(chunks)} chunks to vector store...")
        
        # Prepare data
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings using injected generator
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        
        # Add to ChromaDB in batches
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                documents=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            
            total_added += (batch_end - i)
            print(f"  Added batch {i//batch_size + 1}: {total_added}/{len(chunks)} chunks")
        
        print(f"Successfully added {total_added} chunks to collection")
        return total_added
    
    
    def add_documents_with_embeddings(
        self,
        chunks: List,
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> int:
        """
        Add document chunks with pre-computed embeddings to the vector store
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: Pre-computed embeddings (same order as chunks)
            batch_size: Number of documents to process at once
            
        Returns:
            Number of documents added
        """
        if not chunks or not embeddings:
            print("No chunks or embeddings to add")
            return 0
            
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")
        
        print(f"\nAdding {len(chunks)} chunks with pre-computed embeddings...")
        
        # Prepare data
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add to ChromaDB in batches
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                documents=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            
            total_added += (batch_end - i)
            print(f"  Added batch {i//batch_size + 1}: {total_added}/{len(chunks)} chunks")
        
        print(f"Successfully added {total_added} chunks to collection")
        return total_added
    
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents using cosine similarity
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {'filename': 'doc.txt'})
            
        Returns:
            Dictionary containing results with texts, distances, and metadata
        """
        # Create temporary DocumentChunk for query (using duck typing)
        class QueryChunk:
            def __init__(self, text, chunk_id):
                self.text = text
                self.metadata = {"query": True}
                self.chunk_id = chunk_id
        
        query_chunk = QueryChunk(query, "temp_query")
        
        # Generate query embedding using injected generator
        query_embeddings = self.embedding_generator.generate_embeddings([query_chunk])
        query_embedding = query_embeddings[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = {
            'query': query,
            'results': []
        }
        
        # ChromaDB returns results in lists
        for i in range(len(results['ids'][0])):
            # Convert distance to similarity score (ChromaDB uses L2 distance)
            # Similarity = 1 / (1 + distance)
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)
            
            formatted_results['results'].append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': distance,
                'similarity': similarity
            })
        
        return formatted_results
    
    
    def search_by_similarity_threshold(
        self,
        query: str,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> Dict:
        """
        Search and return only results above similarity threshold
        
        Args:
            query: Search query text
            threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results to retrieve
            
        Returns:
            Dictionary containing filtered results
        """
        # Get more results than needed to filter
        results = self.search(query, n_results=max_results)
        
        # Filter by threshold
        filtered_results = {
            'query': query,
            'threshold': threshold,
            'results': [
                r for r in results['results'] 
                if r['similarity'] >= threshold
            ]
        }
        
        return filtered_results
    
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the current collection
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get a sample to check
        if count > 0:
            sample = self.collection.peek(limit=1)
            has_data = len(sample['ids']) > 0
        else:
            has_data = False
        
        stats = {
            'collection_name': self.collection_name,
            'total_documents': count,
            'has_data': has_data,
            'persist_directory': self.persist_directory,
            'embedding_model': self.embedding_generator.model_name,
            'embedding_dimension': self.embedding_generator.embedding_dim
        }
        
        return stats
    
    
    def delete_collection(self):
        """
        Delete the current collection
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection '{self.collection_name}'")
            # Recreate empty collection
            self.collection = self._get_or_create_collection()
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
    
    
    def get_documents_by_metadata(
        self,
        metadata_filter: Dict,
        limit: int = 10
    ) -> List[Dict]:
        """
        Retrieve documents by metadata filtering
        
        Args:
            metadata_filter: Metadata to filter by (e.g., {'filename': 'doc.txt'})
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        results = self.collection.get(
            where=metadata_filter,
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        documents = []
        for i in range(len(results['ids'])):
            documents.append({
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return documents
    
    
    def update_document(
        self,
        doc_id: str,
        new_text: Optional[str] = None,
        new_metadata: Optional[Dict] = None
    ):
        """
        Update an existing document
        
        Args:
            doc_id: Document ID to update
            new_text: New text content (will regenerate embedding)
            new_metadata: New metadata to merge with existing
        """
        if new_text:
            # Generate new embedding using injected generator (duck typing)
            class TempChunk:
                def __init__(self, text, metadata, chunk_id):
                    self.text = text
                    self.metadata = metadata or {}
                    self.chunk_id = chunk_id
            
            temp_chunk = TempChunk(new_text, new_metadata or {}, doc_id)
            new_embeddings = self.embedding_generator.generate_embeddings([temp_chunk])
            
            self.collection.update(
                ids=[doc_id],
                documents=[new_text],
                embeddings=new_embeddings,
                metadatas=[new_metadata] if new_metadata else None
            )
        elif new_metadata:
            self.collection.update(
                ids=[doc_id],
                metadatas=[new_metadata]
            )
        
        print(f"Updated document: {doc_id}")
    
    
    def delete_documents(self, doc_ids: List[str]):
        """
        Delete specific documents by ID
        
        Args:
            doc_ids: List of document IDs to delete
        """
        self.collection.delete(ids=doc_ids)
        print(f"Deleted {len(doc_ids)} documents")
    
    
    def display_stats(self):
        """
        Display collection statistics in readable format
        """
        stats = self.get_collection_stats()
        
        print("\n" + "="*50)
        print("VECTOR STORE STATISTICS")
        print("="*50)
        print(f"Collection Name: {stats['collection_name']}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Embedding Model: {stats['embedding_model']}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        print(f"Storage Location: {stats['persist_directory']}")
        print(f"Has Data: {stats['has_data']}")
        print("="*50)
    
    
    def compare_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Use injected embedding generator for consistency
        return self.embedding_generator.compare_embeddings(text1, text2)


# Example usage and testing
if __name__ == "__main__":
    print("Vector Store - Test Run\n")
    
    # Import required classes
    from embedding_generator import EmbeddingGenerator
    from document_loader import DocumentChunk
    
    # Initialize components with dependency injection
    generator = EmbeddingGenerator(show_progress=True)
    store = VectorStore(
        embedding_generator=generator,
        persist_directory="./data/test_chroma_db",
        collection_name="test_collection"
    )
    
    # Create sample chunks
    sample_chunks = [
        DocumentChunk(
            text="Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
            metadata={'filename': 'ml_intro.txt', 'chunk_index': 0},
            chunk_id="ml_intro_chunk_0"
        ),
        DocumentChunk(
            text="Gradient descent is an optimization algorithm used to minimize the cost function in machine learning.",
            metadata={'filename': 'ml_intro.txt', 'chunk_index': 1},
            chunk_id="ml_intro_chunk_1"
        ),
        DocumentChunk(
            text="Neural networks are composed of layers of interconnected nodes that process information.",
            metadata={'filename': 'neural_nets.txt', 'chunk_index': 0},
            chunk_id="neural_nets_chunk_0"
        )
    ]
    
    # Add documents
    print("\n1. Adding documents to vector store...")
    store.add_documents(sample_chunks)
    
    # Display stats
    print("\n2. Collection statistics:")
    store.display_stats()
    
    # Test search
    print("\n3. Testing similarity search...")
    query = "How does gradient descent work?"
    results = store.search(query, n_results=2)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results['results'])} results:\n")
    
    for i, result in enumerate(results['results'], 1):
        print(f"Result {i}:")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Text: {result['text'][:100]}...")
        print(f"  Metadata: {result['metadata']}\n")
    
    # Test similarity comparison
    print("\n4. Testing direct similarity comparison...")
    text_a = "Machine learning algorithms"
    text_b = "Artificial intelligence systems"
    similarity = store.compare_similarity(text_a, text_b)
    print(f"Similarity between:")
    print(f"  '{text_a}'")
    print(f"  '{text_b}'")
    print(f"  Score: {similarity:.4f}")
    
    # Clean up test collection
    print("\n5. Cleaning up test collection...")
    store.delete_collection()
    
    print("\nTest completed!")