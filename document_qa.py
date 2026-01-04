#!/usr/bin/env python3
"""
Document-Based Question Answering System
Provides intelligent answers directly from document content without external API dependencies
"""

import sys
import os
from typing import Dict, List, Any
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from document_loader import DocumentLoader
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class DocumentQASystem:
    """
    Document-based question answering system with intelligent text processing.
    Provides natural language responses without external API dependencies.
    """
    
    def __init__(self):
        """Initialize the document QA system."""
        print("Initializing Document Q&A System...")
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(
            collection_name="semantic_documents",  # Use the same collection as app.py
            embedding_generator=self.embedding_generator,
            persist_directory="data/chroma_db"  # Use the same data directory
        )
        
        print("System ready - Fast responses from document analysis!")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get instant formatted answer from documents.
        
        Args:
            question: User's question
            
        Returns:
            Formatted response with answer and sources
        """
        try:
            print(f"\n🔍 Searching for: '{question}'")
            
            # Search for relevant documents
            results = self.vector_store.search(
                query=question,
                n_results=5
            )
            
            if not results or not results.get('results'):
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant information in the documents.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Extract the best matches from formatted results
            documents = []
            similarities = []
            metadatas = []
            
            for result in results['results']:
                documents.append(result['text'])
                similarities.append(result['similarity'])
                metadatas.append(result['metadata'])
            
            # Calculate confidence (similarity = 1 - distance)
            avg_confidence = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Format the answer based on question type
            formatted_answer = self._format_answer(question, documents, similarities)
            
            # Prepare sources
            sources = []
            for i, (doc, similarity, metadata) in enumerate(zip(documents, similarities, metadatas)):
                source_info = {
                    'index': i + 1,
                    'content': doc[:200] + "..." if len(doc) > 200 else doc,
                    'similarity': similarity,
                    'filename': metadata.get('filename', 'Unknown') if metadata else 'Unknown'
                }
                sources.append(source_info)
            
            return {
                'question': question,
                'answer': formatted_answer,
                'sources': sources,
                'confidence': avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    def _format_answer(self, question: str, documents: List[str], similarities: List[float]) -> str:
        """Format a natural answer based on the question and retrieved documents."""
        
        # Get the most relevant document
        if not documents:
            return "No relevant information found."
        
        primary_doc = documents[0]
        
        # Question-specific formatting
        if any(term in question.lower() for term in ['attention', 'attend']):
            return self._format_attention_answer(primary_doc, documents)
        elif any(term in question.lower() for term in ['transformer', 'model']):
            return self._format_transformer_answer(primary_doc, documents)
        elif any(term in question.lower() for term in ['what', 'how', 'why']):
            return self._format_general_answer(primary_doc, documents)
        else:
            return f"Based on the research documents: {primary_doc[:300]}..."
    
    def _format_attention_answer(self, primary_doc: str, all_docs: List[str]) -> str:
        """Format answer specifically about attention mechanisms."""
        answer_parts = ["The attention mechanism is a key component in modern neural networks. "]
        
        # Extract attention-related content
        for doc in all_docs[:2]:  # Use top 2 most relevant
            if any(term in doc.lower() for term in ['attention', 'query', 'key', 'value']):
                # Find sentences about attention
                sentences = doc.split('.')
                for sentence in sentences:
                    if 'attention' in sentence.lower() and len(sentence.strip()) > 20:
                        answer_parts.append(sentence.strip() + ". ")
                        break
                break
        
        answer_parts.append("\nThis mechanism allows models to focus on different parts of the input when processing information, ")
        answer_parts.append("which is crucial for handling sequential data and long-range dependencies.")
        
        return ''.join(answer_parts)
    
    def _format_transformer_answer(self, primary_doc: str, all_docs: List[str]) -> str:
        """Format answer about transformer models."""
        answer_parts = ["Transformer models represent a significant advancement in neural architecture. "]
        
        # Extract transformer-related content
        for doc in all_docs[:2]:
            if any(term in doc.lower() for term in ['transformer', 'encoder', 'decoder']):
                sentences = doc.split('.')
                for sentence in sentences:
                    if any(term in sentence.lower() for term in ['transformer', 'model']) and len(sentence.strip()) > 20:
                        answer_parts.append(sentence.strip() + ". ")
                        break
                break
        
        answer_parts.append("\nThe transformer architecture has become the foundation for many state-of-the-art models in natural language processing.")
        
        return ''.join(answer_parts)
    
    def _format_general_answer(self, primary_doc: str, all_docs: List[str]) -> str:
        """Format a general answer for other questions."""
        answer_parts = ["According to the research literature, "]
        
        # Extract the most informative sentences
        sentences = primary_doc.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 30:  # Skip very short sentences
                relevant_sentences.append(sentence.strip())
                if len(relevant_sentences) >= 2:  # Take first 2 good sentences
                    break
        
        if relevant_sentences:
            for sentence in relevant_sentences:
                answer_parts.append(sentence + ". ")
        else:
            answer_parts.append(primary_doc[:200] + "...")
        
        answer_parts.append("\nThis information provides insights into the current research in this area.")
        
        return ''.join(answer_parts)
    
    def print_response(self, response: Dict[str, Any]):
        """Print a nicely formatted response."""
        print("\n" + "="*80)
        print("DOCUMENT RESEARCH ASSISTANT")
        print("="*80)
        
        print(f"\n❓ QUESTION:")
        print(f"{response['question']}")
        
        print(f"\n📖 ANSWER:")
        print("-" * 50)
        print(response['answer'])
        
        confidence_emoji = "🟢" if response['confidence'] > 0.7 else "🟡" if response['confidence'] > 0.4 else "🔴"
        print(f"\n{confidence_emoji} CONFIDENCE: {response['confidence']:.1%}")
        
        if response['sources']:
            print(f"\n📋 SOURCES:")
            print("-" * 50)
            for source in response['sources']:
                print(f"[{source['index']}] {source['filename']} (Similarity: {source['similarity']:.3f})")
                print(f"    {source['content'][:150]}...")
                print()
        
        print("="*80)


def main():
    """Main function to run the instant Q&A system."""
    if len(sys.argv) < 2:
        print("Usage: python instant_qa.py 'your question here'")
        print("   or: python instant_qa.py interactive")
        return
    
    # Initialize system
    qa_system = DocumentQASystem()
    
    if sys.argv[1].lower() == 'interactive':
        # Interactive mode
        print("Interactive mode - Type 'quit' to exit")
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question:
                    response = qa_system.ask_question(question)
                    qa_system.print_response(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    else:
        # Single question mode
        question = ' '.join(sys.argv[1:])
        response = qa_system.ask_question(question)
        qa_system.print_response(response)


if __name__ == "__main__":
    main()