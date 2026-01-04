"""
Response Formatter for RAG Q&A System
Formats LLM responses with academic citations, confidence scores, and source attribution
Optimized for scientific research paper citations and academic presentation
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
from rag_context_assembler import ContextChunk


@dataclass
class RAGResponse:
    """Complete RAG response with all metadata"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    response_metadata: Dict[str, Any]
    generation_info: Dict[str, Any]
    timestamp: str


class RAGResponseFormatter:
    """
    Formats RAG responses with academic-style citations and confidence assessment
    
    Features:
    - Academic citation formatting
    - Confidence score calculation
    - Source attribution with page numbers
    - Response quality assessment
    - Research-appropriate presentation
    """
    
    def __init__(self, citation_style: str = "academic", include_confidence: bool = True):
        """
        Initialize response formatter
        
        Args:
            citation_style: Citation format ('academic', 'numbered', 'brief')
            include_confidence: Whether to include confidence scores
        """
        self.citation_style = citation_style
        self.include_confidence = include_confidence
        
        # Confidence assessment weights
        self.confidence_weights = {
            'similarity_score': 0.4,      # How well chunks match the query
            'source_diversity': 0.2,      # Number of different sources
            'content_length': 0.1,        # Sufficient context provided
            'llm_confidence': 0.2,        # LLM generation success
            'citation_coverage': 0.1       # How well sources support the answer
        }
    
    def format_response(self, 
                       llm_response: Dict[str, Any], 
                       context_data: Dict[str, Any], 
                       original_question: str) -> RAGResponse:
        """
        Format complete RAG response with citations and confidence assessment
        
        Args:
            llm_response: Response from LLM interface
            context_data: Context assembly results
            original_question: Original user question
            
        Returns:
            Formatted RAGResponse object
        """
        # Extract basic response
        answer = llm_response.get('response', '')
        
        # Format sources with citations
        formatted_sources = self._format_sources(context_data.get('chunks', []))
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(llm_response, context_data, answer)
        
        # Enhance answer with citations
        enhanced_answer = self._add_citations_to_answer(answer, context_data.get('chunks', []))
        
        # Create response metadata
        response_metadata = self._create_response_metadata(context_data, confidence)
        
        # Generation info
        generation_info = {
            'model_used': llm_response.get('model_used', 'unknown'),
            'generation_time': llm_response.get('generation_time', 0),
            'tokens_generated': llm_response.get('tokens_generated', 0),
            'success': llm_response.get('success', False),
            'context_tokens': context_data.get('total_tokens', 0),
            'num_sources': context_data.get('num_sources', 0)
        }
        
        return RAGResponse(
            answer=enhanced_answer,
            sources=formatted_sources,
            confidence_score=confidence,
            response_metadata=response_metadata,
            generation_info=generation_info,
            timestamp=datetime.now().isoformat()
        )
    
    def _format_sources(self, chunks: List[ContextChunk]) -> List[Dict[str, Any]]:
        """Format source citations in academic style"""
        formatted_sources = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = {
                'citation_id': i,
                'source_file': chunk.source,
                'similarity_score': round(chunk.similarity_score, 3),
                'section_type': chunk.section_type,
                'page_number': chunk.page_number,
                'chunk_id': chunk.chunk_id,
                'text_preview': self._create_text_preview(chunk.text),
                'token_count': chunk.token_count
            }
            
            # Create formatted citation
            citation = self._create_citation(chunk, i)
            source_info['formatted_citation'] = citation
            
            formatted_sources.append(source_info)
        
        return formatted_sources
    
    def _create_citation(self, chunk: ContextChunk, citation_id: int) -> str:
        """Create formatted citation based on style preference"""
        base_citation = f"[{citation_id}] {chunk.source}"
        
        if self.citation_style == "academic":
            # Academic style with full details
            citation_parts = [base_citation]
            
            if chunk.section_type:
                citation_parts.append(f"({chunk.section_type.title()} section)")
            
            if chunk.page_number:
                citation_parts.append(f"Page {chunk.page_number}")
            
            if self.include_confidence:
                citation_parts.append(f"Relevance: {chunk.similarity_score:.3f}")
            
            return ", ".join(citation_parts)
        
        elif self.citation_style == "numbered":
            # Simple numbered citations
            extras = []
            if chunk.page_number:
                extras.append(f"p.{chunk.page_number}")
            if chunk.section_type:
                extras.append(chunk.section_type.title())
            
            if extras:
                return f"{base_citation} ({', '.join(extras)})"
            return base_citation
        
        else:  # brief
            return base_citation
    
    def _create_text_preview(self, text: str, max_length: int = 150) -> str:
        """Create preview of source text"""
        if len(text) <= max_length:
            return text
        
        # Try to end at a sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.7:  # If period is reasonably close to end
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def _add_citations_to_answer(self, answer: str, chunks: List[ContextChunk]) -> str:
        """Add citation markers to the answer text"""
        if not chunks:
            return answer
        
        # For now, add citations at the end
        # In advanced implementation, could analyze text to place citations contextually
        enhanced_answer = answer.strip()
        
        if not enhanced_answer.endswith('.'):
            enhanced_answer += '.'
        
        # Add general citation reference
        citation_ids = [str(i) for i in range(1, len(chunks) + 1)]
        citation_text = f" [{', '.join(citation_ids)}]"
        
        return enhanced_answer + citation_text
    
    def _calculate_confidence_score(self, 
                                   llm_response: Dict[str, Any], 
                                   context_data: Dict[str, Any], 
                                   answer: str) -> float:
        """Calculate confidence score based on multiple factors"""
        confidence_factors = {}
        
        # 1. Similarity score factor (average of chunks)
        avg_similarity = context_data.get('avg_similarity', 0)
        confidence_factors['similarity_score'] = min(avg_similarity / 0.8, 1.0)  # Normalize to 0.8 max
        
        # 2. Source diversity factor
        num_sources = context_data.get('num_sources', 0)
        max_sources = min(len(context_data.get('chunks', [])), 3)  # Ideal: 2-3 sources
        confidence_factors['source_diversity'] = min(num_sources / max_sources, 1.0) if max_sources > 0 else 0
        
        # 3. Content length factor (sufficient context)
        total_tokens = context_data.get('total_tokens', 0)
        confidence_factors['content_length'] = min(total_tokens / 2000, 1.0)  # Normalize to 2000 tokens
        
        # 4. LLM confidence factor
        llm_success = 1.0 if llm_response.get('success', False) else 0.5
        generation_time = llm_response.get('generation_time', 0)
        time_factor = 1.0 if generation_time > 0 else 0.8  # Penalize very fast/failed generations
        confidence_factors['llm_confidence'] = llm_success * time_factor
        
        # 5. Citation coverage (simple heuristic)
        answer_length = len(answer.split())
        context_tokens = context_data.get('total_tokens', 1)
        coverage_ratio = min(answer_length / (context_tokens / 4), 1.0)  # Rough word estimation
        confidence_factors['citation_coverage'] = coverage_ratio
        
        # Calculate weighted confidence score
        total_confidence = 0
        for factor, score in confidence_factors.items():
            weight = self.confidence_weights.get(factor, 0)
            total_confidence += score * weight
        
        return round(min(total_confidence, 1.0), 3)
    
    def _create_response_metadata(self, context_data: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Create metadata about the response quality"""
        chunks = context_data.get('chunks', [])
        
        # Analyze source types
        source_types = {}
        section_types = {}
        
        for chunk in chunks:
            # Count source files
            source_name = chunk.source
            source_types[source_name] = source_types.get(source_name, 0) + 1
            
            # Count section types
            if chunk.section_type:
                section_type = chunk.section_type.lower()
                section_types[section_type] = section_types.get(section_type, 0) + 1
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "High"
        elif confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': confidence,
            'sources_used': list(source_types.keys()),
            'section_types_covered': list(section_types.keys()),
            'total_chunks_used': len(chunks),
            'avg_chunk_similarity': context_data.get('avg_similarity', 0),
            'context_summary': context_data.get('total_tokens', 0)
        }
    
    def format_for_display(self, rag_response: RAGResponse) -> str:
        """Format response for console/terminal display"""
        display_parts = []
        
        # Header
        display_parts.append("=" * 80)
        display_parts.append("📚 RESEARCH ASSISTANT RESPONSE")
        display_parts.append("=" * 80)
        
        # Answer
        display_parts.append("\n📖 ANSWER:")
        display_parts.append("-" * 40)
        display_parts.append(rag_response.answer)
        
        # Confidence
        if self.include_confidence:
            confidence = rag_response.confidence_score
            level = rag_response.response_metadata['confidence_level']
            confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
            
            display_parts.append(f"\n{confidence_emoji} CONFIDENCE: {level} ({confidence:.3f})")
        
        # Sources
        display_parts.append("\n📋 SOURCES:")
        display_parts.append("-" * 40)
        
        for source in rag_response.sources:
            display_parts.append(f"{source['formatted_citation']}")
            display_parts.append(f"   Preview: {source['text_preview']}")
            display_parts.append("")
        
        # Generation info
        gen_info = rag_response.generation_info
        display_parts.append("⚙️ GENERATION INFO:")
        display_parts.append("-" * 40)
        display_parts.append(f"Model: {gen_info['model_used']}")
        display_parts.append(f"Generation time: {gen_info['generation_time']:.2f}s")
        display_parts.append(f"Sources used: {gen_info['num_sources']}")
        display_parts.append(f"Context tokens: {gen_info['context_tokens']}")
        
        display_parts.append("=" * 80)
        
        return "\n".join(display_parts)
    
    def format_for_json(self, rag_response: RAGResponse) -> Dict[str, Any]:
        """Format response for JSON serialization"""
        return {
            'answer': rag_response.answer,
            'confidence': {
                'score': rag_response.confidence_score,
                'level': rag_response.response_metadata['confidence_level']
            },
            'sources': [
                {
                    'citation': source['formatted_citation'],
                    'file': source['source_file'],
                    'similarity': source['similarity_score'],
                    'preview': source['text_preview']
                }
                for source in rag_response.sources
            ],
            'metadata': rag_response.response_metadata,
            'generation': rag_response.generation_info,
            'timestamp': rag_response.timestamp
        }