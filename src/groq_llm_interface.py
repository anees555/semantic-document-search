#!/usr/bin/env python3
"""
Groq LLM Interface for RAG Q&A System
Uses Groq's lightning-fast inference API with free tier access
Supports Llama-3, Mixtral, and other excellent open-source models
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqLLMInterface:
    """
    Interface for Groq LLM API - extremely fast and free!
    
    Features:
    - Lightning-fast inference (up to 500+ tokens/sec)
    - Free tier: 14,400 requests/day
    - Multiple excellent models available
    - No local GPU/CPU requirements
    """
    
    def __init__(self, 
                 model_name: str = None,
                 api_key: Optional[str] = None,
                 max_tokens: int = None,
                 temperature: float = None):
        """
        Initialize Groq LLM interface
        
        Args:
            model_name: Groq model to use (defaults to .env DEFAULT_MODEL or llama-3.1-8b-instant)
            api_key: Groq API key (defaults to .env GROQ_API_KEY)
            max_tokens: Maximum tokens to generate (defaults to .env MAX_TOKENS or 1024)
            temperature: Sampling temperature (defaults to .env TEMPERATURE or 0.1)
        """
        # Load configuration from .env file with fallbacks
        self.model_name = model_name or os.getenv('DEFAULT_MODEL', 'llama-3.1-8b-instant')
        self.max_tokens = max_tokens or int(os.getenv('MAX_TOKENS', '1024'))
        self.temperature = temperature or float(os.getenv('TEMPERATURE', '0.1'))
        
        # Get API key from parameter, .env file, or environment variable
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Groq API key required! Either:\n"
                "1. Add GROQ_API_KEY to .env file, or\n"
                "2. Set GROQ_API_KEY environment variable, or\n"
                "3. Pass api_key parameter\n"
                "Get your free API key at: https://console.groq.com/"
            )
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Available models on Groq
        self.available_models = {
            "llama-3.1-70b-versatile": "Meta Llama 3.1 70B (Recommended for complex tasks)",
            "llama-3.1-8b-instant": "Meta Llama 3.1 8B (Fastest, good for simple tasks)",
            "mixtral-8x7b-32768": "Mistral Mixtral 8x7B (Excellent for reasoning)",
            "gemma2-9b-it": "Google Gemma 2 9B (Good balance)",
            "llama3-70b-8192": "Meta Llama 3 70B (High quality)",
            "llama3-8b-8192": "Meta Llama 3 8B (Fast and efficient)"
        }
        
        logger.info(f"Initialized Groq LLM: {model_name}")
        logger.info(f"Model description: {self.available_models.get(model_name, 'Custom model')}")
    
    def generate_response(self, 
                         question: str, 
                         context: str, 
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response using Groq API
        
        Args:
            question: User's question
            context: Retrieved document context
            system_prompt: Optional system prompt override
            
        Returns:
            Dictionary with generated response and metadata
        """
        try:
            start_time = time.time()
            
            # Default system prompt for academic/scientific content
            if not system_prompt:
                system_prompt = """You are an expert research assistant specializing in academic and scientific literature. 
Provide clear, accurate, and well-structured answers based on the given context. 
Use formal academic tone while remaining accessible. 
If the context doesn't contain enough information, say so clearly.
Always cite relevant information from the provided context."""

            # Construct the prompt
            user_prompt = f"""Context from research documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Structure your response clearly and cite specific information from the context when relevant."""

            # Prepare API request
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            logger.info(f"Sending request to Groq API with model: {self.model_name}")
            
            # Make API request
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response
            generated_text = result["choices"][0]["message"]["content"]
            
            # Calculate metrics
            generation_time = time.time() - start_time
            usage = result.get("usage", {})
            
            logger.info(f"Generated response in {generation_time:.2f}s")
            logger.info(f"Tokens - Prompt: {usage.get('prompt_tokens', 0)}, "
                       f"Completion: {usage.get('completion_tokens', 0)}, "
                       f"Total: {usage.get('total_tokens', 0)}")
            
            return {
                "success": True,
                "response": generated_text,
                "model": self.model_name,
                "generation_time": generation_time,
                "usage": usage,
                "error": None
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {
                "success": False,
                "response": "I apologize, but I encountered an error while generating the response. Please try again.",
                "model": self.model_name,
                "generation_time": 0,
                "usage": {},
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "response": "I apologize, but I encountered an unexpected error. Please try again.",
                "model": self.model_name,
                "generation_time": 0,
                "usage": {},
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Test API connection and key validity"""
        try:
            test_payload = {
                "model": "llama-3.1-8b-instant",  # Use fastest model for test
                "messages": [{"role": "user", "content": "Hello, this is a test."}],
                "max_tokens": 10
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Groq API connection successful!")
                return True
            else:
                logger.error(f"❌ API test failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
            return False
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models"""
        return self.available_models
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        """
        Estimate API cost (Groq has generous free tier)
        
        Free tier limits (as of 2026):
        - 14,400 requests per day
        - Rate limits vary by model
        """
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "model": self.model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "note": "Groq offers generous free tier - 14,400 requests/day",
            "cost_usd": 0.0  # Free tier
        }