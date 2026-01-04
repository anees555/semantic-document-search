"""
Context Assembler for RAG Q&A System
Handles intelligent context window assembly from retrieved document chunks
Optimized for academic/scientific research papers
"""

from typing import List, Dict, Any, Optional
import re
import logging
from collections import Counter
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """Represents a processed context chunk with metadata"""
    text: str
    source: str
    similarity_score: float
    chunk_id: str
    section_type: Optional[str] = None
    page_number: Optional[int] = None
    token_count: int = 0


class RAGContextAssembler:
    """
    Assembles optimal context from retrieved chunks for academic Q&A
    
    Features:
    - Token limit management
    - Content deduplication 
    - Relevance-based ranking
    - Academic section prioritization
    - Source diversity optimization
    """
    
    def __init__(self, max_tokens: int = 3000, min_similarity_threshold: float = 0.6):
        """
        Initialize context assembler
        
        Args:
            max_tokens: Maximum tokens for context window
            min_similarity_threshold: Minimum similarity score to include chunks
        """
        self.max_tokens = max_tokens
        self.min_similarity_threshold = min_similarity_threshold
        
        # Academic section priorities for scientific papers
        self.section_priorities = {
            'abstract': 10,
            'introduction': 8,
            'methods': 7,
            'methodology': 7,
            'results': 9,
            'discussion': 8,
            'conclusion': 9,
            'related work': 6,
            'background': 6,
            'references': 2,
            'acknowledgments': 1,
            'default': 5
        }
    
    def assemble_context(self, search_results: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Assemble optimal context from search results
        
        Args:
            search_results: Results from vector search
            question: Original user question for context optimization
            
        Returns:
            Dictionary containing assembled context and metadata
        """
        if not search_results.get('results'):
            return {'context': '', 'chunks': [], 'total_tokens': 0}
        
        # Convert search results to ContextChunk objects
        chunks = self._convert_to_context_chunks(search_results['results'])
        logger.info(f"Converted {len(chunks)} chunks from search results")
        
        if chunks:
            avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)
            logger.info(f"Average similarity: {avg_similarity:.3f}, min threshold: {self.min_similarity_threshold}")
        
        # Filter by similarity threshold
        chunks = [c for c in chunks if c.similarity_score >= self.min_similarity_threshold]
        logger.info(f"After filtering by threshold ({self.min_similarity_threshold}): {len(chunks)} chunks")
        
        if not chunks:
            return {'context': '', 'chunks': [], 'total_tokens': 0}
        
        # Deduplicate similar content
        chunks = self._deduplicate_chunks(chunks)
        
        # Rank chunks by relevance for the specific question
        chunks = self._rank_chunks_for_question(chunks, question)
        
        # Select optimal subset within token limit
        selected_chunks = self._select_optimal_chunks(chunks)
        
        # Assemble final context
        context_text = self._build_context_text(selected_chunks)
        
        return {
            'context': context_text,
            'chunks': selected_chunks,
            'total_tokens': sum(c.token_count for c in selected_chunks),
            'num_sources': len(set(c.source for c in selected_chunks)),
            'avg_similarity': sum(c.similarity_score for c in selected_chunks) / len(selected_chunks) if selected_chunks else 0
        }
    
    def _convert_to_context_chunks(self, search_results: List[Dict]) -> List[ContextChunk]:
        """Convert search results to ContextChunk objects"""
        chunks = []
        
        # Handle different search result formats
        if isinstance(search_results, dict) and 'results' in search_results:
            results_list = search_results['results']
        else:
            results_list = search_results
        
        for result in results_list:
            # Extract metadata from result
            metadata = result.get('metadata', {})
            
            # Handle different text field names ('text' or 'chunk')
            text_content = result.get('text', result.get('chunk', ''))
            
            chunk = ContextChunk(
                text=text_content,
                source=metadata.get('source', 'Unknown'),
                similarity_score=result.get('similarity', 0.0),
                chunk_id=metadata.get('chunk_id', 'unknown'),
                section_type=metadata.get('section_type'),
                page_number=metadata.get('page_number')
            )
            
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            chunk.token_count = len(chunk.text) // 4
            
            if chunk.text:  # Only add chunks with content
                chunks.append(chunk)
        
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[ContextChunk], similarity_threshold: float = 0.85) -> List[ContextChunk]:
        """Remove highly similar chunks to reduce redundancy"""
        if len(chunks) <= 1:
            return chunks
        
        deduplicated = []
        
        for chunk in chunks:
            is_duplicate = False
            
            for existing in deduplicated:
                # Simple text similarity check using common words
                similarity = self._text_similarity(chunk.text, existing.text)
                
                if similarity > similarity_threshold:
                    # Keep chunk with higher similarity score
                    if chunk.similarity_score > existing.similarity_score:
                        deduplicated.remove(existing)
                        deduplicated.append(chunk)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(chunk)
        
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity based on common words"""
        # Simple word-based similarity for deduplication
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _rank_chunks_for_question(self, chunks: List[ContextChunk], question: str) -> List[ContextChunk]:
        """Rank chunks based on relevance to specific question"""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        def calculate_relevance_score(chunk: ContextChunk) -> float:
            base_score = chunk.similarity_score
            
            # Academic section priority bonus
            section_priority = self.section_priorities.get(
                chunk.section_type.lower() if chunk.section_type else 'default',
                self.section_priorities['default']
            )
            section_bonus = (section_priority / 10) * 0.1  # Up to 10% bonus
            
            # Question word overlap bonus
            chunk_words = set(re.findall(r'\b\w+\b', chunk.text.lower()))
            overlap = len(question_words.intersection(chunk_words))
            overlap_bonus = min(overlap / len(question_words), 0.2) if question_words else 0  # Up to 20% bonus
            
            return base_score + section_bonus + overlap_bonus
        
        # Sort by relevance score (descending)
        chunks.sort(key=calculate_relevance_score, reverse=True)
        
        return chunks
    
    def _select_optimal_chunks(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Select optimal subset of chunks within token limit"""
        selected = []
        total_tokens = 0
        sources_used = set()
        
        for chunk in chunks:
            # Check token limit
            if total_tokens + chunk.token_count > self.max_tokens:
                continue
            
            # Prefer source diversity (but don't enforce strict limits)
            source_diversity_bonus = 0.1 if chunk.source not in sources_used else 0
            
            selected.append(chunk)
            total_tokens += chunk.token_count
            sources_used.add(chunk.source)
            
            # Stop if we have good coverage and are approaching token limit
            if len(selected) >= 3 and total_tokens > self.max_tokens * 0.8:
                break
        
        return selected
    
    def _build_context_text(self, chunks: List[ContextChunk]) -> str:
        """Build formatted context text from selected chunks"""
        if not chunks:
            return ""
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Format chunk with metadata
            source_info = f"[Source {i}: {chunk.source}"
            if chunk.section_type:
                source_info += f", {chunk.section_type.title()}"
            if chunk.page_number:
                source_info += f", Page {chunk.page_number}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{chunk.text.strip()}")
        
        return "\n\n" + "\n\n---\n\n".join(context_parts) + "\n\n"
    
    def get_context_summary(self, context_data: Dict[str, Any]) -> str:
        """Generate summary of assembled context"""
        if not context_data['chunks']:
            return "No relevant context found."
        
        summary = f"Context assembled from {len(context_data['chunks'])} chunks "
        summary += f"({context_data['num_sources']} sources, "
        summary += f"{context_data['total_tokens']} tokens, "
        summary += f"avg similarity: {context_data['avg_similarity']:.3f})"
        
        return summary