"""
Local LLM Interface for RAG Q&A System
Uses free Hugging Face models optimized for academic/scientific content
Supports models like Mistral-7B, Llama2, and other open-source LLMs
"""

import os
import torch
from typing import Dict, Any, Optional, List
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, BitsAndBytesConfig
)
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMInterface:
    """
    Interface for local LLM inference using free Hugging Face models
    
    Optimized for:
    - Academic/scientific content understanding
    - Memory-efficient inference 
    - GPU acceleration when available
    - Multiple model backends
    """
    
    # Recommended free models for scientific content
    RECOMMENDED_MODELS = {
        'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
        'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
        'zephyr-7b': 'HuggingFaceH4/zephyr-7b-beta',
        'openchat': 'openchat/openchat-3.5-0106',  # Fast and good for QA
        'phi3-mini': 'microsoft/Phi-3-mini-4k-instruct',  # Very efficient
    }
    
    def __init__(self, 
                 model_name: str = 'microsoft/Phi-3-mini-4k-instruct',
                 use_quantization: bool = True,
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 device: Optional[str] = None):
        """
        Initialize local LLM interface
        
        Args:
            model_name: HuggingFace model identifier
            use_quantization: Use 4-bit quantization for memory efficiency
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Device selection with better CUDA detection
        if device is None:
            # Check if CUDA is actually available and working
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    # Test if CUDA actually works
                    test_tensor = torch.tensor([1.0]).cuda()
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception as e:
                logger.warning(f"CUDA test failed: {e}. Using CPU.")
                self.device = "cpu"
        else:
            self.device = device
            
        # Disable quantization for CPU
        if self.device == "cpu":
            self.use_quantization = False
            
        logger.info(f"Initializing LLM interface with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Quantization: {use_quantization}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for memory efficiency
            model_kwargs = {"trust_remote_code": True}
            
            if self.use_quantization and self.device == "cuda":
                logger.info("Setting up 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            
            logger.info("Loading model... (this may take a few minutes)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.device == "cuda" else None,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.info("Falling back to CPU-optimized settings...")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize with CPU-optimized settings if GPU loading fails"""
        try:
            self.device = "cpu"
            self.use_quantization = False
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info("Fallback model loaded successfully on CPU!")
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback model: {e}")
            raise RuntimeError("Could not initialize any model configuration")
    
    def generate_response(self, question: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response using the local LLM
        
        Args:
            question: User's question
            context: Assembled context from retrieved chunks
            system_prompt: Optional system prompt override
            
        Returns:
            Dictionary containing generated response and metadata
        """
        if self.pipeline is None:
            raise RuntimeError("Model not initialized")
        
        # Create system prompt for academic Q&A
        if system_prompt is None:
            system_prompt = self._create_academic_system_prompt()
        
        # Format the prompt
        formatted_prompt = self._format_prompt(system_prompt, context, question)
        
        # Track generation start time
        start_time = datetime.now()
        
        try:
            logger.info("Generating response...")
            
            # Production timeout to prevent hanging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM generation timed out")
            
            # Set 30-second timeout for production
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            # Generate response with fixed parameters for better compatibility
            with torch.no_grad():
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=3072)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(self.max_new_tokens, 128),  # Limit tokens for speed
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Disable cache to avoid DynamicCache error
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
                
                # Extract only the generated part
                generated_ids = outputs[0][len(inputs['input_ids'][0]):]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
            # Cancel timeout
            signal.alarm(0)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Post-process the response
            cleaned_response = self._clean_response(generated_text)
            
            logger.info(f"Response generated in {generation_time:.2f}s")
            
            return {
                'response': cleaned_response,
                'generation_time': generation_time,
                'model_used': self.model_name,
                'tokens_generated': len(self.tokenizer.encode(generated_text)),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': "I apologize, but I encountered an error while generating the response. Please try rephrasing your question or check the system configuration.",
                'generation_time': 0,
                'model_used': self.model_name,
                'tokens_generated': 0,
                'success': False,
                'error': str(e)
            }
    
    def _create_academic_system_prompt(self) -> str:
        """Create system prompt optimized for academic/scientific Q&A"""
        return """You are a knowledgeable research assistant specialized in academic and scientific literature. Your role is to provide accurate, well-reasoned answers based on the provided research context.

Instructions:
1. Answer questions using ONLY the information provided in the context
2. Be precise and cite specific findings from the sources
3. If the context doesn't contain enough information, clearly state this limitation
4. Use clear, academic language appropriate for researchers
5. Highlight key concepts, methodologies, and findings
6. When discussing results, mention any limitations or caveats noted in the sources
7. Structure your response logically with clear reasoning

Always base your responses on evidence from the provided context and maintain scientific accuracy."""
    
    def _format_prompt(self, system_prompt: str, context: str, question: str) -> str:
        """Format the complete prompt for the model"""
        # Different models may require different prompt formats
        if "mistral" in self.model_name.lower():
            return f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
        elif "llama" in self.model_name.lower():
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
        elif "phi" in self.model_name.lower():
            return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\nContext:\n{context}\n\nQuestion: {question}<|end|>\n<|assistant|>"
        else:
            # Generic format for other models
            return f"System: {system_prompt}\n\nContext:\n{context}\n\nHuman: {question}\n\nAssistant:"
    
    def _clean_response(self, response: str) -> str:
        """Clean and post-process the generated response"""
        # Remove any residual prompt tokens or formatting
        response = response.strip()
        
        # Remove common artifacts
        artifacts = ["<|end|>", "</s>", "[/INST]", "<|assistant|>", "Human:", "Assistant:"]
        for artifact in artifacts:
            response = response.replace(artifact, "").strip()
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        # Calculate model parameters (approximate)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": self.use_quantization,
            "parameters": f"{total_params / 1e6:.1f}M",
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "status": "loaded"
        }
    
    @classmethod
    def get_recommended_model_for_hardware(cls) -> str:
        """Get recommended model based on available hardware"""
        if torch.cuda.is_available():
            # Check VRAM (approximate)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory >= 16:
                return cls.RECOMMENDED_MODELS['mistral-7b']
            elif gpu_memory >= 8:
                return cls.RECOMMENDED_MODELS['phi3-mini']
            else:
                return cls.RECOMMENDED_MODELS['phi3-mini']
        else:
            # CPU-only: use most efficient model
            return cls.RECOMMENDED_MODELS['phi3-mini']
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()