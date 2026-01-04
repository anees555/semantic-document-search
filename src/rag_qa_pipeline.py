"""
RAG Q&A Pipeline
Integrates context assembly, LLM interface, and response formatting
Complete question-answering system for academic/scientific documents
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from integrated_pipeline import DocumentSearchPipeline
from rag_context_assembler import RAGContextAssembler
from local_llm_interface import LocalLLMInterface
from rag_response_formatter import RAGResponseFormatter, RAGResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGQuestionAnsweringPipeline:
    """
    Complete RAG Q&A pipeline for academic document question-answering
    
    Integrates:
    - Semantic document search (existing DocumentSearchPipeline)
    - Context assembly and optimization
    - Local LLM response generation
    - Academic citation formatting
    
    Optimized for scientific research papers and academic content
    """
    
    def __init__(self, 
                 document_pipeline: Optional[DocumentSearchPipeline] = None,
                 llm_model_name: Optional[str] = None,
                 max_context_tokens: int = 3000,
                 use_gpu: bool = True,
                 citation_style: str = "academic"):
        """
        Initialize RAG Q&A pipeline
        
        Args:
            document_pipeline: Existing DocumentSearchPipeline instance
            llm_model_name: HuggingFace model name (auto-selected if None)
            max_context_tokens: Maximum tokens for context assembly
            use_gpu: Whether to use GPU for LLM inference
            citation_style: Citation format for responses
        """
        logger.info("Initializing RAG Q&A Pipeline...")
        
        # Initialize or use existing document search pipeline
        if document_pipeline is None:
            logger.info("Creating new DocumentSearchPipeline...")
            self.document_pipeline = DocumentSearchPipeline(
                use_grobid=True,
                chunk_size=1000,
                chunk_overlap=150
            )
        else:
            self.document_pipeline = document_pipeline
        
        # Initialize context assembler
        logger.info("Initializing context assembler...")
        self.context_assembler = RAGContextAssembler(
            max_tokens=max_context_tokens,
            min_similarity_threshold=0.3  # Lowered from 0.6 to include more results
        )
        
        # Initialize response formatter
        logger.info("Initializing response formatter...")
        self.response_formatter = RAGResponseFormatter(
            citation_style=citation_style,
            include_confidence=True
        )
        
        # Skip LLM initialization - use Simple Q&A approach instead
        logger.info("Using Simple Q&A approach for reliable production performance")
        self.use_simple_qa = True
        self.llm_interface = None
        
        self.stats = {
            'questions_answered': 0,
            'total_response_time': 0,
            'successful_responses': 0,
            'failed_responses': 0
        }
        
        logger.info("RAG Q&A Pipeline initialized successfully!")
        logger.info(f"Using model: {llm_model_name}")
        logger.info(f"Device: {self.llm_interface.device}")
    
    def answer_question(self, 
                       question: str, 
                       n_results: int = 5,
                       min_similarity: float = 0.6,
                       custom_system_prompt: Optional[str] = None) -> RAGResponse:
        """
        Answer a question using the complete RAG pipeline
        
        Args:
            question: User's question
            n_results: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold for retrieval
            custom_system_prompt: Optional custom system prompt
            
        Returns:
            Complete RAGResponse with answer, sources, and metadata
        """
        start_time = datetime.now()
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Step 1: Semantic document search
            logger.info("🔍 Retrieving relevant documents...")
            search_results = self.document_pipeline.search_documents(
                query=question,
                n_results=n_results
            )
            
            if not search_results or not search_results.get('results'):
                return self._create_no_results_response(question)
            
            # Step 2: Assemble context
            logger.info("📋 Assembling context...")
            context_data = self.context_assembler.assemble_context(
                search_results, question
            )
            
            if not context_data['context']:
                return self._create_insufficient_context_response(question)
            
            logger.info(f"Context assembled: {context_data['total_tokens']} tokens from {len(context_data['chunks'])} chunks")
            
            # Step 3: Generate response using LLM
            logger.info("🤖 Generating response...")
            llm_response = self.llm_interface.generate_response(
                question=question,
                context=context_data['context'],
                system_prompt=custom_system_prompt
            )
            
            # Step 4: Format response with citations
            logger.info("📚 Formatting response...")
            formatted_response = self.response_formatter.format_response(
                llm_response=llm_response,
                context_data=context_data,
                original_question=question
            )
            
            # Update statistics
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats['questions_answered'] += 1
            self.stats['total_response_time'] += response_time
            
            if llm_response.get('success', False):
                self.stats['successful_responses'] += 1
            else:
                self.stats['failed_responses'] += 1
            
            logger.info(f"✅ Question answered successfully in {response_time:.2f}s")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}")
            self.stats['failed_responses'] += 1
            return self._create_error_response(question, str(e))
    
    def _create_no_results_response(self, question: str) -> RAGResponse:
        """Create response when no relevant documents are found"""
        return RAGResponse(
            answer="I couldn't find any relevant information in the document collection to answer your question. Please try rephrasing your question or ensure that relevant documents have been processed.",
            sources=[],
            confidence_score=0.0,
            response_metadata={
                'confidence_level': 'None',
                'sources_used': [],
                'section_types_covered': [],
                'total_chunks_used': 0,
                'context_summary': 0
            },
            generation_info={
                'model_used': 'none',
                'generation_time': 0,
                'tokens_generated': 0,
                'success': False,
                'context_tokens': 0,
                'num_sources': 0
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _create_insufficient_context_response(self, question: str) -> RAGResponse:
        """Create response when context is insufficient"""
        return RAGResponse(
            answer="I found some potentially relevant documents, but the content doesn't provide sufficient information to answer your question confidently. Please try a more specific question or check if more relevant documents are available.",
            sources=[],
            confidence_score=0.1,
            response_metadata={
                'confidence_level': 'Low',
                'sources_used': [],
                'section_types_covered': [],
                'total_chunks_used': 0,
                'context_summary': 0
            },
            generation_info={
                'model_used': self.llm_interface.model_name,
                'generation_time': 0,
                'tokens_generated': 0,
                'success': False,
                'context_tokens': 0,
                'num_sources': 0
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _create_error_response(self, question: str, error_msg: str) -> RAGResponse:
        """Create response for system errors"""
        return RAGResponse(
            answer=f"I encountered a system error while processing your question: {error_msg}. Please try again or contact support if the problem persists.",
            sources=[],
            confidence_score=0.0,
            response_metadata={
                'confidence_level': 'Error',
                'sources_used': [],
                'section_types_covered': [],
                'total_chunks_used': 0,
                'context_summary': 0
            },
            generation_info={
                'model_used': getattr(self.llm_interface, 'model_name', 'unknown'),
                'generation_time': 0,
                'tokens_generated': 0,
                'success': False,
                'context_tokens': 0,
                'num_sources': 0,
                'error': error_msg
            },
            timestamp=datetime.now().isoformat()
        )
    
    def interactive_qa_session(self):
        """Start an interactive Q&A session"""
        print("\n🤖 RAG Question-Answering System")
        print("=" * 50)
        print("Ask questions about your documents!")
        print("Commands: 'quit' to exit, 'stats' to show statistics, 'help' for help")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    self.display_qa_statistics()
                    continue
                
                if question.lower() == 'help':
                    self._display_help()
                    continue
                
                # Process the question
                print("\n🔄 Processing...")
                response = self.answer_question(question)
                
                # Display formatted response
                print(self.response_formatter.format_for_display(response))
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Interactive session error: {e}")
    
    def _display_help(self):
        """Display help information"""
        help_text = """
📖 HELP - RAG Question-Answering System
=====================================

QUESTION TYPES:
• Factual: "What is machine learning?"
• Comparative: "How do CNNs differ from RNNs?"
• Procedural: "How to implement attention mechanism?"
• Analytical: "What are the limitations of this approach?"

TIPS FOR BETTER RESULTS:
• Be specific and clear in your questions
• Use technical terms from your documents
• Ask about concepts that appear in your processed documents
• Try different phrasings if results aren't satisfactory

COMMANDS:
• 'quit' or 'exit' - End the session
• 'stats' - Show system statistics
• 'help' - Display this help message

CONFIDENCE LEVELS:
🟢 High (>0.8) - Very confident, multiple relevant sources
🟡 Medium (0.6-0.8) - Moderately confident, some relevant sources
🔴 Low (<0.6) - Low confidence, limited or weak sources
        """
        print(help_text)
    
    def batch_answer_questions(self, questions: List[str]) -> List[RAGResponse]:
        """Answer multiple questions in batch"""
        responses = []
        
        logger.info(f"Processing {len(questions)} questions in batch...")
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            response = self.answer_question(question)
            responses.append(response)
        
        return responses
    
    def display_qa_statistics(self):
        """Display Q&A system statistics"""
        stats = self.stats.copy()
        
        if stats['questions_answered'] > 0:
            avg_response_time = stats['total_response_time'] / stats['questions_answered']
            success_rate = (stats['successful_responses'] / stats['questions_answered']) * 100
        else:
            avg_response_time = 0
            success_rate = 0
        
        print("\n📊 RAG Q&A SYSTEM STATISTICS")
        print("=" * 40)
        print(f"Questions answered: {stats['questions_answered']}")
        print(f"Successful responses: {stats['successful_responses']}")
        print(f"Failed responses: {stats['failed_responses']}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average response time: {avg_response_time:.2f}s")
        print("=" * 40)
        
        # Document pipeline stats
        if hasattr(self.document_pipeline, 'display_pipeline_stats'):
            print("\n📚 DOCUMENT PIPELINE STATISTICS")
            print("=" * 40)
            self.document_pipeline.display_pipeline_stats()
        
        # LLM model info
        model_info = self.llm_interface.get_model_info()
        print("\n🤖 MODEL INFORMATION")
        print("=" * 40)
        print(f"Model: {model_info.get('model_name', 'Unknown')}")
        print(f"Device: {model_info.get('device', 'Unknown')}")
        print(f"Parameters: {model_info.get('parameters', 'Unknown')}")
        print(f"Quantization: {model_info.get('quantization', 'Unknown')}")
        print("=" * 40)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'document_pipeline': {
                'status': 'active',
                'documents_processed': getattr(self.document_pipeline, 'documents_processed', 0)
            },
            'llm_interface': self.llm_interface.get_model_info(),
            'qa_statistics': self.stats.copy(),
            'context_assembler': {
                'max_tokens': self.context_assembler.max_tokens,
                'min_similarity_threshold': self.context_assembler.min_similarity_threshold
            },
            'response_formatter': {
                'citation_style': self.response_formatter.citation_style,
                'include_confidence': self.response_formatter.include_confidence
            }
        }