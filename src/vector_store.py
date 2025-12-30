"""
vector_store.py
Vector database management using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path


class VectorStore:
    """
    Manages vector embeddings and similarity search using ChromaDB
    """
    
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize VectorStore
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
            embedding_model: Sentence Transformer model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
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
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Model loaded. Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        
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
    
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    
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
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
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
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
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
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension()
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
            # Generate new embedding
            new_embedding = self.embedding_model.encode([new_text]).tolist()
            
            self.collection.update(
                ids=[doc_id],
                documents=[new_text],
                embeddings=new_embedding,
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
        embeddings = self.embedding_model.encode([text1, text2])
        
        # Calculate cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        
        similarity = dot(embeddings[0], embeddings[1]) / (
            norm(embeddings[0]) * norm(embeddings[1])
        )
        
        return float(similarity)


# Example usage and testing
if __name__ == "__main__":
    print("Vector Store - Test Run\n")
    
    # Initialize vector store
    store = VectorStore(
        persist_directory="./data/test_chroma_db",
        collection_name="test_collection"
    )
    
    # Create sample chunks (mimicking document_loader output)
    from dataclasses import dataclass
    
    @dataclass
    class DocumentChunk:
        text: str
        metadata: Dict
        chunk_id: str
    
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