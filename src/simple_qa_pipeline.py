"""
Simple Q&A Pipeline - Fast Context-Based Question Answering
Uses retrievals context to generate responses without heavy LLM inference
Perfect for quick testing and demonstrating RAG capabilities
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleQAPipeline:
    """
    Simple context-based Q&A system that provides intelligent responses
    using retrieved document chunks without requiring heavy LLM inference.
    """
    
    def __init__(self, integrated_pipeline, max_context_chunks: int = 5):
        """
        Initialize Simple Q&A Pipeline
        
        Args:
            integrated_pipeline: The integrated document search pipeline
            max_context_chunks: Maximum context chunks to use for answers
        """
        self.pipeline = integrated_pipeline
        self.max_context_chunks = max_context_chunks
        logger.info("Simple Q&A Pipeline initialized successfully")
    
    def answer_question(self, question: str, similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Answer a question using retrieved context and intelligent text processing
        
        Args:
            question: The question to answer
            similarity_threshold: Minimum similarity for relevant chunks
            
        Returns:
            Dict containing answer, sources, and metadata
        """
        # Step 1: Retrieve relevant context
        logger.info(f"Processing question: {question}")
        search_results = self.pipeline.search(
            query=question, 
            top_k=self.max_context_chunks, 
            similarity_threshold=similarity_threshold
        )
        
        if not search_results['results']:
            return {
                'answer': "I couldn't find relevant information in the documents to answer your question.",
                'confidence': 0.0,
                'sources': [],
                'context_used': [],
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 2: Extract and analyze context
        context_chunks = []
        sources = []
        
        for result in search_results['results']:
            context_chunks.append({
                'text': result.get('text', result.get('chunk', '')),  # Handle both field names
                'similarity': result['similarity'],
                'source': result['metadata'].get('source', 'Unknown')
            })
            
            source_info = {
                'document': result['metadata'].get('source', 'Unknown'),
                'similarity': result['similarity']
            }
            if source_info not in sources:
                sources.append(source_info)
        
        # Step 3: Generate intelligent response
        answer = self._generate_contextual_answer(question, context_chunks)
        confidence = self._calculate_confidence(context_chunks)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources': sources,
            'context_used': context_chunks,
            'total_chunks_analyzed': len(context_chunks),
            'avg_similarity': sum(c['similarity'] for c in context_chunks) / len(context_chunks) if context_chunks else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_contextual_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Generate an intelligent answer based on context analysis
        """
        if not context_chunks:
            return "No relevant context found to answer the question."
        
        # Analyze question type
        question_type = self._classify_question(question)
        
        # Extract key information from context
        key_info = self._extract_key_information(context_chunks, question)
        
        # Generate structured answer based on question type
        if question_type == 'definition':
            return self._generate_definition_answer(question, key_info, context_chunks)
        elif question_type == 'how':
            return self._generate_how_answer(question, key_info, context_chunks)
        elif question_type == 'why':
            return self._generate_why_answer(question, key_info, context_chunks)
        elif question_type == 'what':
            return self._generate_what_answer(question, key_info, context_chunks)
        else:
            return self._generate_general_answer(question, key_info, context_chunks)
    
    def _classify_question(self, question: str) -> str:
        """Classify question type to generate appropriate response structure"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'define', 'definition']):
            return 'definition'
        elif question_lower.startswith('how'):
            return 'how'
        elif question_lower.startswith('why'):
            return 'why'
        elif question_lower.startswith('what'):
            return 'what'
        else:
            return 'general'
    
    def _extract_key_information(self, context_chunks: List[Dict], question: str) -> Dict[str, List[str]]:
        """Extract key information from context chunks"""
        key_info = {
            'definitions': [],
            'processes': [],
            'explanations': [],
            'examples': [],
            'numbers': [],
            'entities': []
        }
        
        for chunk in context_chunks:
            text = chunk['text']
            
            # Extract definitions (sentences with "is", "are", "refers to")
            definition_patterns = [
                r'([A-Z][^.]*(?:is|are|refers to|means|denotes)[^.]*\.)',
                r'([A-Z][^.]*(?:defined as|known as)[^.]*\.)'
            ]
            
            for pattern in definition_patterns:
                matches = re.findall(pattern, text)
                key_info['definitions'].extend(matches)
            
            # Extract process descriptions (sentences with process words)
            process_words = ['algorithm', 'method', 'approach', 'technique', 'process', 'procedure']
            for word in process_words:
                if word in text.lower():
                    sentences = text.split('.')
                    for sentence in sentences:
                        if word in sentence.lower():
                            key_info['processes'].append(sentence.strip() + '.')
            
            # Extract numbers and statistics
            numbers = re.findall(r'\b\d+\.?\d*\%?\b', text)
            key_info['numbers'].extend(numbers)
            
            # Extract capitalized entities (likely important terms)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            key_info['entities'].extend(entities)
        
        # Remove duplicates
        for key in key_info:
            key_info[key] = list(set(key_info[key]))
        
        return key_info
    
    def _generate_definition_answer(self, question: str, key_info: Dict, context_chunks: List[Dict]) -> str:
        """Generate definition-type answer"""
        answer_parts = []
        
        if key_info['definitions']:
            answer_parts.append("Based on the documents:")
            answer_parts.extend(key_info['definitions'][:2])  # Top 2 definitions
        
        if key_info['entities']:
            entities_str = ', '.join(key_info['entities'][:5])
            answer_parts.append(f"\\nKey terms mentioned: {entities_str}")
        
        if not answer_parts:
            # Fallback: use most relevant chunk
            best_chunk = context_chunks[0]
            answer_parts.append(f"According to the most relevant source: {best_chunk['text'][:300]}...")
        
        return '\\n'.join(answer_parts)
    
    def _generate_how_answer(self, question: str, key_info: Dict, context_chunks: List[Dict]) -> str:
        """Generate how-type answer"""
        answer_parts = ["Based on the document analysis:"]
        
        if key_info['processes']:
            answer_parts.append("\\nProcess/Method described:")
            answer_parts.extend(key_info['processes'][:2])
        
        if key_info['numbers']:
            numbers_str = ', '.join(key_info['numbers'][:5])
            answer_parts.append(f"\\nRelevant metrics: {numbers_str}")
        
        # Add most relevant context if no specific processes found
        if not key_info['processes']:
            best_chunk = context_chunks[0]
            answer_parts.append(f"\\nFrom the most relevant section: {best_chunk['text'][:250]}...")
        
        return '\\n'.join(answer_parts)
    
    def _generate_why_answer(self, question: str, key_info: Dict, context_chunks: List[Dict]) -> str:
        """Generate why-type answer"""
        answer_parts = ["Based on the document analysis:"]
        
        # Look for explanatory content
        explanations = []
        for chunk in context_chunks[:3]:  # Top 3 chunks
            text = chunk['text']
            sentences = text.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['because', 'since', 'due to', 'reason', 'cause']):
                    explanations.append(sentence.strip() + '.')
        
        if explanations:
            answer_parts.append("\\nExplanations found:")
            answer_parts.extend(explanations[:2])
        else:
            # Fallback to most relevant content
            best_chunk = context_chunks[0]
            answer_parts.append(f"\\nRelevant context: {best_chunk['text'][:300]}...")
        
        return '\\n'.join(answer_parts)
    
    def _generate_what_answer(self, question: str, key_info: Dict, context_chunks: List[Dict]) -> str:
        """Generate what-type answer"""
        return self._generate_definition_answer(question, key_info, context_chunks)
    
    def _generate_general_answer(self, question: str, key_info: Dict, context_chunks: List[Dict]) -> str:
        """Generate general answer"""
        answer_parts = ["Based on the document analysis:"]
        
        # Combine best information
        if key_info['definitions']:
            answer_parts.append("\\nDefinitions:")
            answer_parts.extend(key_info['definitions'][:1])
        
        if key_info['processes']:
            answer_parts.append("\\nProcess information:")
            answer_parts.extend(key_info['processes'][:1])
        
        # Always include most relevant chunk
        best_chunk = context_chunks[0]
        answer_parts.append(f"\\nMost relevant context: {best_chunk['text'][:250]}...")
        
        return '\\n'.join(answer_parts)
    
    def _calculate_confidence(self, context_chunks: List[Dict]) -> float:
        """Calculate confidence score based on context quality"""
        if not context_chunks:
            return 0.0
        
        # Base confidence on average similarity
        avg_similarity = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
        
        # Boost confidence if we have multiple high-quality chunks
        high_quality_chunks = sum(1 for chunk in context_chunks if chunk['similarity'] > 0.5)
        quality_bonus = min(high_quality_chunks * 0.1, 0.3)  # Max 30% bonus
        
        final_confidence = min(avg_similarity + quality_bonus, 1.0)
        return round(final_confidence, 3)
    
    def interactive_session(self):
        """Run an interactive Q&A session"""
        print("🤖 Simple Q&A System Ready!")
        print("📚 Documents loaded and processed")
        print("💡 Ask questions about your documents (type 'quit' to exit)")
        print("=" * 60)
        
        while True:
            try:
                question = input("\\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\\n👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\\n🔍 Searching and analyzing...")
                result = self.answer_question(question)
                
                print(f"\\n📝 **Answer** (Confidence: {result['confidence']:.1%}):")
                print(result['answer'])
                
                print(f"\\n📊 **Analysis Summary:**")
                print(f"   • Chunks analyzed: {result.get('total_chunks_analyzed', 0)}")
                print(f"   • Average similarity: {result.get('avg_similarity', 0):.3f}")
                sources_str = ', '.join([s['document'] for s in result.get('sources', [])]) if result.get('sources') else 'None'
                print(f"   • Sources: {sources_str}")
                
                if result['confidence'] < 0.4:
                    print("\\n⚠️  Low confidence - answer may be incomplete or speculative")
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Session terminated by user")
                break
            except Exception as e:
                print(f"\\n❌ Error: {str(e)}")
                print("Please try a different question.")