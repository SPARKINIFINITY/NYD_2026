"""
Answer Generator

Main LLM-based answer generation component that uses API-based models
to generate grounded answers with proper citations and structured output.
"""

import json
import time
import logging
import os
import requests
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Main LLM-based answer generator using API-based models"""
    
    def __init__(self, 
                 model_name: str = "openai/gpt-3.5-turbo",
                 device: str = "auto",
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9):
        
        # Map common model names to OpenRouter IDs
        self.model_name = self._map_model_name(model_name)
        self.device = device  # Keep for compatibility but not used in API mode
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # API configuration
        self.api_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = self._get_api_key(self.model_name)
        
        # Generation statistics
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0.0,
            "total_tokens_generated": 0
        }
        
        # Generation configuration as a property
        self.generation_config = {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        # Validate API configuration
        self._validate_api_config()
        
        logger.info(f"Initialized AnswerGenerator with API-based model: {model_name}")
    
    def _map_model_name(self, model_name: str) -> str:
        """Map common model names to OpenRouter model IDs (OPEN SOURCE ONLY)"""
        
        # Model name mapping - ALL OPEN SOURCE MODELS
        model_mapping = {
            # Map to Llama 3.3 (Open Source - Meta Llama 3.3 License) - FREE!
            "microsoft/DialoGPT-medium": "meta-llama/llama-3.3-70b-instruct:free",
            "microsoft/DialoGPT-small": "meta-llama/llama-3.3-70b-instruct:free",
            "microsoft/DialoGPT-large": "meta-llama/llama-3.3-70b-instruct:free",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "meta-llama/llama-3.3-70b-instruct:free",
            
            # Direct open source models (no mapping needed)
            "meta-llama/llama-3.3-70b-instruct:free": "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
            "mistralai/mistral-7b-instruct": "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct": "mistralai/mixtral-8x7b-instruct",
        }
        
        # Return mapped name or original if not in mapping
        mapped_name = model_mapping.get(model_name, model_name)
        
        if mapped_name != model_name:
            logger.info(f"Mapped model '{model_name}' to '{mapped_name}' (Open Source)")
        
        return mapped_name
    
    def _get_api_key(self, model_name: str) -> str:
        """Get the appropriate API key based on model name"""
        
        # Check if it's a Llama model
        if "llama" in model_name.lower() or "405b" in model_name.lower():
            api_key = os.getenv("OPENROUTER_API_KEY_LLAMA")
            if not api_key:
                logger.warning("OPENROUTER_API_KEY_LLAMA not found, trying default")
                api_key = os.getenv("OPENROUTER_API_KEY_DIALOGPT")
        else:
            # Default to DialoGPT key (now used for GPT models)
            api_key = os.getenv("OPENROUTER_API_KEY_DIALOGPT")
            if not api_key:
                logger.warning("OPENROUTER_API_KEY_DIALOGPT not found, trying Llama key")
                api_key = os.getenv("OPENROUTER_API_KEY_LLAMA")
        
        return api_key
    
    def _validate_api_config(self):
        """Validate API configuration"""
        
        if not self.api_key:
            raise ValueError(
                "No API key found. Please set OPENROUTER_API_KEY_DIALOGPT or "
                "OPENROUTER_API_KEY_LLAMA in your .env file"
            )
        
        if not self.api_base_url:
            raise ValueError("API base URL not configured")
        
        logger.info("API configuration validated successfully")
    
    def generate_answer(self, 
                       prompt: Dict[str, str], 
                       evidence_documents: List[Dict[str, Any]],
                       max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate grounded answer using the language model
        
        Args:
            prompt: Dictionary with system_prompt and user_prompt
            evidence_documents: List of evidence documents
            max_retries: Maximum number of generation retries
        
        Returns:
            Generated answer with metadata
        """
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Prepare input text
                input_text = self._prepare_input_text(prompt)
                
                # Generate response
                generated_text = self._generate_text(input_text)
                
                # Post-process and validate
                processed_answer = self._post_process_answer(generated_text, evidence_documents)
                
                # Add metadata
                generation_time = time.time() - start_time
                processed_answer["metadata"] = {
                    "model_used": self.model_name,
                    "generation_time": generation_time,
                    "evidence_count": len(evidence_documents),
                    "attempt": attempt + 1,
                    "device": self.device
                }
                
                # Update statistics
                self._update_generation_stats(True, generation_time, len(generated_text))
                
                return processed_answer
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Final attempt failed, return error response
                    self._update_generation_stats(False, time.time() - start_time, 0)
                    return self._create_error_response(str(e), evidence_documents)
        
        # Should not reach here
        return self._create_error_response("Max retries exceeded", evidence_documents)
    
    def _prepare_input_text(self, prompt: Dict[str, str]) -> List[Dict[str, str]]:
        """Prepare input messages for the API"""
        
        system_prompt = prompt.get("system_prompt", "")
        user_prompt = prompt.get("user_prompt", "")
        
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        return messages
    
    def _generate_text(self, messages: List[Dict[str, str]]) -> str:
        """Generate text using the API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/NYD-2026",
            "X-Title": "NYD RAG System"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text from API response
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]
                
                # Track token usage if available
                if "usage" in result:
                    self.generation_stats["total_tokens_generated"] += result["usage"].get("total_tokens", 0)
                
                return generated_text.strip()
            else:
                raise ValueError("No response generated from API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            raise
    
    def _post_process_answer(self, 
                           generated_text: str, 
                           evidence_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Post-process generated answer to ensure proper format"""
        
        # Try to extract JSON from generated text
        json_match = self._extract_json_from_text(generated_text)
        
        if json_match:
            try:
                parsed_json = json.loads(json_match)
                
                # Validate required fields
                required_fields = ["summary", "detailed_answer", "references", "confidence"]
                if all(field in parsed_json for field in required_fields):
                    return parsed_json
                
            except json.JSONDecodeError:
                pass
        
        # If JSON extraction failed, create structured response from text
        return self._create_structured_response_from_text(generated_text, evidence_documents)
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from generated text"""
        
        # Look for JSON object patterns
        import re
        
        # Pattern 1: Complete JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # Test if it's valid JSON
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        # Pattern 2: Look for JSON-like structure with quotes
        json_like_pattern = r'\{.*?"summary".*?"detailed_answer".*?"references".*?"confidence".*?\}'
        match = re.search(json_like_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(0)
        
        return None
    
    def _create_structured_response_from_text(self, 
                                            text: str, 
                                            evidence_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create structured response when JSON extraction fails"""
        
        # Extract summary (first sentence or paragraph)
        sentences = text.split('.')
        summary = sentences[0].strip() if sentences else text[:100]
        
        # Use full text as detailed answer
        detailed_answer = text.strip()
        
        # Create basic references from evidence documents
        references = []
        for i, doc in enumerate(evidence_documents[:3]):  # Top 3 documents
            references.append({
                "doc_id": doc.get('id', f'doc_{i+1}'),
                "span": doc.get('content', '')[:150] + "..." if len(doc.get('content', '')) > 150 else doc.get('content', ''),
                "relevance_score": doc.get('similarity_score', 0.5)
            })
        
        # Estimate confidence based on text length and evidence
        confidence = min(0.8, len(text) / 200.0) if text else 0.1
        
        return {
            "summary": summary,
            "detailed_answer": detailed_answer,
            "references": references,
            "confidence": confidence
        }
    
    def _create_error_response(self, error_message: str, evidence_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create error response when generation fails"""
        
        return {
            "summary": "Generation failed",
            "detailed_answer": f"I apologize, but I encountered an error while generating the answer: {error_message}",
            "references": [],
            "confidence": 0.0,
            "error": error_message,
            "metadata": {
                "model_used": self.model_name,
                "generation_time": 0.0,
                "evidence_count": len(evidence_documents),
                "attempt": 1,
                "device": self.device,
                "error": error_message
            }
        }
    
    def _update_generation_stats(self, success: bool, generation_time: float, tokens_generated: int):
        """Update generation statistics"""
        
        self.generation_stats["total_generations"] += 1
        
        if success:
            self.generation_stats["successful_generations"] += 1
        else:
            self.generation_stats["failed_generations"] += 1
        
        # Update average generation time
        total_time = (self.generation_stats["avg_generation_time"] * 
                     (self.generation_stats["total_generations"] - 1) + generation_time)
        self.generation_stats["avg_generation_time"] = total_time / self.generation_stats["total_generations"]
        
        self.generation_stats["total_tokens_generated"] += tokens_generated
    
    def batch_generate(self, 
                      prompts: List[Dict[str, str]], 
                      evidence_batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Generate answers for multiple prompts in batch"""
        
        if len(prompts) != len(evidence_batches):
            raise ValueError("Number of prompts must match number of evidence batches")
        
        results = []
        
        for prompt, evidence_docs in zip(prompts, evidence_batches):
            result = self.generate_answer(prompt, evidence_docs)
            results.append(result)
        
        return results
    
    def update_generation_config(self, **kwargs):
        """Update generation configuration"""
        
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "top_p" in kwargs:
            self.top_p = kwargs["top_p"]
        
                # Update the generation_config dictionary
        if "max_length" in kwargs:
            self.generation_config["max_length"] = self.max_length
        if "temperature" in kwargs:
            self.generation_config["temperature"] = self.temperature
        if "top_p" in kwargs:
            self.generation_config["top_p"] = self.top_p
        
        logger.info(f"Updated generation config: {kwargs}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        
        info = {
            "model_name": self.model_name,
            "api_mode": "OpenRouter API",
            "api_base_url": self.api_base_url,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        return info
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        
        stats = self.generation_stats.copy()
        
        # Calculate derived metrics
        if stats["total_generations"] > 0:
            stats["success_rate"] = stats["successful_generations"] / stats["total_generations"]
            stats["failure_rate"] = stats["failed_generations"] / stats["total_generations"]
            stats["avg_tokens_per_generation"] = stats["total_tokens_generated"] / stats["total_generations"]
        
        return stats
    
    def clear_stats(self):
        """Clear generation statistics"""
        
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0.0,
            "total_tokens_generated": 0
        }
        
        # Generation configuration as a property
        self.generation_config = {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        logger.info("Cleared generation statistics")