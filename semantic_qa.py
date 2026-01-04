#!/usr/bin/env python3
"""
Semantic Question Answering System with AI Integration
Combines semantic search with natural language generation for intelligent document Q&A
"""

import sys
import os
from typing import Dict, List, Any
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from document_loader import DocumentLoader
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from groq_llm_interface import GroqLLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class SemanticQASystem:
    """
    Advanced question answering system using semantic search and AI language models
    
    Features:
    - Semantic document retrieval
    - Natural language generation for comprehensive answers
    - Fast response times with cloud-based inference
    - Free usage with generous daily limits
    """
    
    def __init__(self, groq_api_key: str = None, model_name: str = None):
        """Initialize the semantic QA system"""
        print("Initializing Semantic Q&A System...")
        
        # Initialize document processing components
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(
            collection_name="semantic_documents",
            embedding_generator=self.embedding_generator,
            persist_directory="data/chroma_db"
        )
        
        # Initialize Groq LLM interface (will load from .env automatically)
        default_model = model_name or os.getenv('DEFAULT_MODEL', 'llama-3.1-8b-instant')
        print(f"Initializing AI model: {default_model}")
        self.groq_llm = GroqLLMInterface(
            model_name=default_model,
            api_key=groq_api_key  # Will fall back to .env if None
        )
        
        # Test API connection
        if not self.groq_llm.test_connection():
            raise Exception("Failed to connect to AI service. Please check your API key in .env file!")
        
        print("System ready for question answering!")
        print("Available models:", len(self.groq_llm.get_available_models()))
    
    def ask_question(self, question: str, max_context_chunks: int = 5) -> Dict[str, Any]:
        """
        Ask a question and get LLM-generated answer with document context
        
        Args:
            question: User's question
            max_context_chunks: Maximum number of document chunks to use for context
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        try:
            print(f"\n🔍 Processing question: '{question}'")
            
            # Step 1: Semantic search for relevant documents
            search_results = self.vector_store.search(
                query=question,
                n_results=max_context_chunks
            )
            
            if not search_results or not search_results.get('results'):
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant information in the documents to answer your question.",
                    'sources': [],
                    'confidence': 0.0,
                    'model_used': self.groq_llm.model_name,
                    'generation_time': 0,
                    'success': False
                }
            
            # Step 2: Prepare context from retrieved documents
            context_chunks = []
            sources = []
            
            for i, result in enumerate(search_results['results']):
                chunk_text = result['text']
                similarity = result['similarity']
                metadata = result.get('metadata', {})
                
                context_chunks.append(f"[Source {i+1}] {chunk_text}")
                sources.append({
                    'index': i + 1,
                    'content': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    'similarity': similarity,
                    'filename': metadata.get('filename', 'Unknown')
                })
            
            # Combine context
            full_context = "\n\n".join(context_chunks)
            avg_confidence = sum(s['similarity'] for s in sources) / len(sources)
            
            print(f"📚 Retrieved {len(sources)} relevant chunks (avg similarity: {avg_confidence:.3f})")
            
            # Step 3: Generate answer using Groq LLM
            print("🤖 Generating response with Groq LLM...")
            llm_response = self.groq_llm.generate_response(
                question=question,
                context=full_context
            )
            
            # Step 4: Format final response
            if llm_response['success']:
                print(f"✅ Response generated in {llm_response['generation_time']:.2f}s")
                
                return {
                    'question': question,
                    'answer': llm_response['response'],
                    'sources': sources,
                    'confidence': avg_confidence,
                    'model_used': llm_response['model'],
                    'generation_time': llm_response['generation_time'],
                    'token_usage': llm_response['usage'],
                    'success': True
                }
            else:
                print(f"❌ LLM generation failed: {llm_response.get('error', 'Unknown error')}")
                return {
                    'question': question,
                    'answer': llm_response['response'],
                    'sources': sources,
                    'confidence': avg_confidence,
                    'model_used': llm_response['model'],
                    'generation_time': 0,
                    'success': False,
                    'error': llm_response.get('error')
                }
                
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'question': question,
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def print_response(self, response: Dict[str, Any]):
        """Print a beautifully formatted response"""
        print("\n" + "="*80)
        print("SEMANTIC RESEARCH ASSISTANT")
        print("="*80)
        
        print(f"\n❓ QUESTION:")
        print(f"{response['question']}")
        
        print(f"\n📖 ANSWER:")
        print("-" * 50)
        print(response['answer'])
        
        # Confidence indicator
        confidence = response.get('confidence', 0)
        if confidence > 0.7:
            conf_emoji = "🟢 High"
        elif confidence > 0.4:
            conf_emoji = "🟡 Medium" 
        else:
            conf_emoji = "🔴 Low"
            
        print(f"\n{conf_emoji} CONFIDENCE: {confidence:.1%}")
        
        # Model info
        model_used = response.get('model_used', 'Unknown')
        gen_time = response.get('generation_time', 0)
        print(f"🤖 MODEL: {model_used}")
        print(f"⚡ RESPONSE TIME: {gen_time:.2f}s")
        
        # Token usage (if available)
        if 'token_usage' in response:
            usage = response['token_usage']
            if usage:
                print(f"🔢 TOKENS: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total")
        
        # Sources
        if response.get('sources'):
            print(f"\n📋 SOURCES:")
            print("-" * 50)
            for source in response['sources']:
                print(f"[{source['index']}] {source['filename']} (Similarity: {source['similarity']:.3f})")
                print(f"    {source['content']}")
                print()
        
        print("="*80)


def main():
    """Main function to run the semantic QA system"""
    if len(sys.argv) < 2:
        print("Usage: python semantic_qa.py 'your question here'")
        print("   or: python semantic_qa.py interactive")
        print("\nNote: API key should be configured in .env file")
        print("Create .env file with: GROQ_API_KEY=your_api_key_here")
        print("Get your free API key at: https://console.groq.com/")
        return
    
    try:
        # Initialize system (will load API key from .env automatically)
        qa_system = SemanticQASystem()
        
        if sys.argv[1].lower() == 'interactive':
            # Interactive mode
            print("Interactive Semantic Q&A mode - Type 'quit' to exit")
            print("Ask questions about your documents!")
            
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
    except Exception as e:
        print(f"Failed to initialize system: {str(e)}")
        print("\nPlease ensure:")
        print("1. .env file exists with GROQ_API_KEY=your_api_key_here")
        print("2. Documents are loaded (run: python app.py load)")
        print("3. Internet connection is available")


if __name__ == "__main__":
    main()