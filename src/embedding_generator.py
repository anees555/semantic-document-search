"""
Embedding Generator Module
Generates semantic embeddings from DocumentChunk objects using sentence-transformers.
Optimized for academic document processing with comprehensive error handling.
"""

import os
import re
import time
import logging
import warnings
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    warnings.warn("PyTorch not available. CPU-only mode will be used.")

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    st_available = True
except ImportError:
    st_available = False
    raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

from document_loader import DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    High-performance embedding generator for academic documents using sentence-transformers.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        cpu_batch_size: int = 32,
        gpu_batch_size: int = 64,
        max_seq_length: int = 256,
        normalize_embeddings: bool = True,
        show_progress: bool = True
    ):
        """
        Initialize the embedding generator.
        """
        self.model_name = model_name
        self.cpu_batch_size = cpu_batch_size
        self.gpu_batch_size = gpu_batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.show_progress = show_progress
        
        # Device detection
        self.device = self._detect_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Model initialization
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Performance tracking
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0
        self.model_load_time = 0.0
        
        # Initialize model
        self._load_model()
    
    def _detect_device(self, device: Optional[str]) -> str:
        """Detect the best available device for processing."""
        if device is not None and device != 'auto':
            return device
            
        if torch_available and torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def _load_model(self) -> None:
        """Load the sentence transformer model with error handling."""
        try:
            start_time = time.time()
            logger.info(f"Loading model: {self.model_name}")
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Configure model settings
            self.model.max_seq_length = self.max_seq_length
            
            # Get actual embedding dimension
            sample_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dim = len(sample_embedding)
            
            self.model_load_time = time.time() - start_time
            logger.info(f"Model loaded in {self.model_load_time:.2f}s. Embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding generation.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not text.strip():
            return ""
        
        # Normalize whitespace but preserve academic symbols
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Keep important academic symbols and formatting
        # Remove excessive punctuation but keep mathematical notation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\_\+\=\<\>\@\#\$\%\^\&\*\/\\]', '', text)
        
        # Ensure reasonable length bounds
        if len(text) < 10:  # Very short text might not be meaningful
            logger.warning(f"Very short text after preprocessing: '{text[:50]}...'")
        
        return text
    
    def _validate_chunk(self, chunk) -> bool:
        """
        Validate a document chunk for embedding generation.
        
        Args:
            chunk: DocumentChunk to validate
            
        Returns:
            True if chunk is valid for processing
        """
        # Check if object has required attributes (duck typing)
        if not hasattr(chunk, 'text') or not hasattr(chunk, 'chunk_id'):
            logger.warning(f"Invalid chunk: missing text or chunk_id attributes")
            return False
        
        if not chunk.text or not chunk.text.strip():
            logger.warning(f"Empty chunk text for chunk_id: {getattr(chunk, 'chunk_id', 'unknown')}")
            return False
        
        preprocessed = self._preprocess_text(chunk.text)
        if len(preprocessed) < 5:  # Minimum meaningful length
            logger.warning(f"Chunk too short after preprocessing: {getattr(chunk, 'chunk_id', 'unknown')}")
            return False
        
        if len(preprocessed) > 2000:  # Very long chunks
            logger.info(f"Long chunk will be truncated: {getattr(chunk, 'chunk_id', 'unknown')} ({len(preprocessed)} chars)")
        
        return True
    
    def _prepare_texts(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Prepare and validate texts from chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of preprocessed text strings
        """
        texts = []
        valid_indices = []
        
        for i, chunk in enumerate(chunks):
            if self._validate_chunk(chunk):
                preprocessed = self._preprocess_text(chunk.text)
                texts.append(preprocessed)
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping invalid chunk at index {i}")
        
        logger.info(f"Prepared {len(texts)} valid texts from {len(chunks)} chunks")
        return texts, valid_indices
    
    def generate_embeddings(
        self, 
        chunks,  # Remove type hint to allow duck typing
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of DocumentChunk objects or objects with text/chunk_id attributes
            batch_size: Override default batch size
            
        Returns:
            List of L2-normalized embedding vectors (384-dimensional)
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        start_time = time.time()
        
        # Prepare texts
        texts, valid_indices = self._prepare_texts(chunks)
        if not texts:
            logger.error("No valid texts found for embedding generation")
            return []
        
        # Determine batch size
        if batch_size is None:
            batch_size = self.gpu_batch_size if self.device == 'cuda' else self.cpu_batch_size
        
        logger.info(f"Generating embeddings for {len(texts)} texts with batch_size={batch_size}")
        
        try:
            # Generate embeddings with progress bar
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )
            
            # Validate embeddings
            if embeddings.shape[1] != self.embedding_dim:
                logger.error(f"Unexpected embedding dimension: {embeddings.shape[1]} vs {self.embedding_dim}")
                raise ValueError("Embedding dimension mismatch")
            
            # Convert to list format and ensure proper ordering
            embedding_list = []
            valid_idx = 0
            
            for i in range(len(chunks)):
                if i in valid_indices:
                    embedding = embeddings[valid_idx].tolist()
                    
                    # Verify L2 normalization
                    if self.normalize_embeddings:
                        norm = np.linalg.norm(embedding)
                        if not np.isclose(norm, 1.0, rtol=1e-5):
                            logger.warning(f"Embedding not properly normalized: norm={norm}")
                    
                    embedding_list.append(embedding)
                    valid_idx += 1
                else:
                    # Provide zero embedding for invalid chunks
                    embedding_list.append([0.0] * self.embedding_dim)
            
            # Clean up GPU memory if using CUDA
            if self.device == 'cuda' and torch_available:
                torch.cuda.empty_cache()
            
            # Update statistics
            processing_time = time.time() - start_time
            self.total_embeddings_generated += len(texts)
            self.total_processing_time += processing_time
            
            logger.info(f"Generated {len(embedding_list)} embeddings in {processing_time:.2f}s")
            
            return embedding_list
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise RuntimeError(f"Embedding generation error: {str(e)}")
    
    def compare_embeddings(self, text1: str, text2: str) -> float:
        """
        Compare two texts using cosine similarity of their embeddings.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Create temporary chunks (using duck typing)
            class TempChunk:
                def __init__(self, text, chunk_id):
                    self.text = text
                    self.metadata = {}
                    self.chunk_id = chunk_id
            
            chunk1 = TempChunk(text1, "temp1")
            chunk2 = TempChunk(text2, "temp2")
            
            # Generate embeddings
            embeddings = self.generate_embeddings([chunk1, chunk2])
            
            if len(embeddings) != 2:
                logger.error("Failed to generate embeddings for comparison")
                return 0.0
            
            # Calculate cosine similarity
            emb1 = torch.tensor(embeddings[0])
            emb2 = torch.tensor(embeddings[1])
            
            similarity = cos_sim(emb1, emb2).item()
            
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Embedding comparison failed: {str(e)}")
            return 0.0
    
    def display_embedding_stats(self) -> None:
        """Display comprehensive embedding generation statistics."""
        print("\n" + "="*60)
        print("EMBEDDING GENERATOR STATISTICS")
        print("="*60)
        
        # Model information
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Max Sequence Length: {self.max_seq_length}")
        print(f"Normalization: {'L2' if self.normalize_embeddings else 'None'}")
        
        # Performance statistics
        print(f"\nModel Load Time: {self.model_load_time:.2f}s")
        print(f"Total Embeddings Generated: {self.total_embeddings_generated:,}")
        print(f"Total Processing Time: {self.total_processing_time:.2f}s")
        
        if self.total_embeddings_generated > 0:
            avg_time_per_embedding = (self.total_processing_time / self.total_embeddings_generated) * 1000
            throughput = self.total_embeddings_generated / self.total_processing_time if self.total_processing_time > 0 else 0
            
            print(f"Average Time per Embedding: {avg_time_per_embedding:.2f}ms")
            print(f"Throughput: {throughput:.1f} embeddings/second")
        
        # Batch size configuration
        print(f"\nBatch Sizes:")
        print(f"  CPU: {self.cpu_batch_size}")
        print(f"  GPU: {self.gpu_batch_size}")
        
        # Memory information
        if torch_available and torch.cuda.is_available():
            print(f"\nGPU Memory:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        print("="*60)
    
    def validate_embeddings(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Validate generated embeddings for correctness.
        
        Args:
            embeddings: List of embedding vectors to validate
            
        Returns:
            Dictionary with validation results
        """
        if not embeddings:
            return {"valid": False, "error": "No embeddings provided"}
        
        validation_results = {
            "valid": True,
            "count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0,
            "expected_dimension": self.embedding_dim,
            "normalized": True,
            "errors": []
        }
        
        for i, embedding in enumerate(embeddings):
            # Check dimension
            if len(embedding) != self.embedding_dim:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Embedding {i}: Wrong dimension {len(embedding)}, expected {self.embedding_dim}"
                )
            
            # Check for NaN or infinite values
            if any(not np.isfinite(val) for val in embedding):
                validation_results["valid"] = False
                validation_results["errors"].append(f"Embedding {i}: Contains NaN or infinite values")
            
            # Check L2 normalization if enabled
            if self.normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if not np.isclose(norm, 1.0, rtol=1e-4):
                    validation_results["normalized"] = False
                    validation_results["errors"].append(
                        f"Embedding {i}: Not properly normalized, norm={norm:.6f}"
                    )
        
        return validation_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.max_seq_length,
            "normalization_enabled": self.normalize_embeddings,
            "batch_sizes": {
                "cpu": self.cpu_batch_size,
                "gpu": self.gpu_batch_size
            },
            "model_loaded": self.model is not None,
            "torch_available": torch_available,
            "cuda_available": torch_available and torch.cuda.is_available() if torch_available else False
        }


def test_embedding_generator():
    """Test function demonstrating EmbeddingGenerator usage."""
    print("Testing EmbeddingGenerator...")
    
    # Create sample chunks
    sample_chunks = [
        DocumentChunk(
            text="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata={"section": "introduction", "page": 1},
            chunk_id="chunk_1"
        ),
        DocumentChunk(
            text="Deep learning uses neural networks with multiple layers to process data.",
            metadata={"section": "methods", "page": 3},
            chunk_id="chunk_2"
        ),
        DocumentChunk(
            text="Natural language processing enables computers to understand human language.",
            metadata={"section": "background", "page": 2},
            chunk_id="chunk_3"
        ),
        DocumentChunk(
            text="", # Empty chunk to test validation
            metadata={"section": "empty", "page": 0},
            chunk_id="chunk_4"
        ),
        DocumentChunk(
            text="Transformers revolutionized NLP with attention mechanisms and parallel processing.",
            metadata={"section": "results", "page": 5},
            chunk_id="chunk_5"
        )
    ]
    
    try:
        # Initialize generator
        generator = EmbeddingGenerator(show_progress=True)
        
        # Display model info
        print("\nModel Information:")
        model_info = generator.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Generate embeddings
        print(f"\nGenerating embeddings for {len(sample_chunks)} chunks...")
        embeddings = generator.generate_embeddings(sample_chunks)
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        
        # Validate embeddings
        validation = generator.validate_embeddings(embeddings)
        print(f"\nValidation Results:")
        print(f"  Valid: {validation['valid']}")
        print(f"  Count: {validation['count']}")
        print(f"  Dimension: {validation['dimension']}")
        print(f"  Normalized: {validation['normalized']}")
        if validation['errors']:
            print("  Errors:")
            for error in validation['errors']:
                print(f"    - {error}")
        
        # Test embedding comparison
        print(f"\nTesting embedding comparison...")
        similarity1 = generator.compare_embeddings(
            "Machine learning uses algorithms",
            "Deep learning uses neural networks"
        )
        similarity2 = generator.compare_embeddings(
            "Machine learning uses algorithms", 
            "Machine learning applies algorithms"
        )
        print(f"  Similarity (ML vs DL): {similarity1:.4f}")
        print(f"  Similarity (ML vs ML-similar): {similarity2:.4f}")
        
        # Display statistics
        generator.display_embedding_stats()
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test when module is executed directly
    success = test_embedding_generator()
    exit(0 if success else 1)