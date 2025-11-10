"""
Generator Orchestrator

Main orchestrator that integrates all generator components and coordinates
with retrieval and reranking systems for end-to-end answer generation.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .answer_generator import AnswerGenerator
from .prompt_template_manager import PromptTemplateManager
from .citation_manager import CitationManager
from .confidence_estimator import ConfidenceEstimator
from .output_formatter import OutputFormatter

logger = logging.getLogger(__name__)

class GeneratorOrchestrator:
    """Main orchestrator for the generator system"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "auto",
                 max_workers: int = 4,
                 enable_async: bool = True):
        
        self.model_name = model_name
        self.device = device
        self.max_workers = max_workers
        self.enable_async = enable_async
        
        # Initialize components
        self.answer_generator = AnswerGenerator(model_name=model_name, device=device)
        self.prompt_manager = PromptTemplateManager()
        self.citation_manager = CitationManager()
        self.confidence_estimator = ConfidenceEstimator()
        self.output_formatter = OutputFormatter()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if enable_async else None
        
        # Orchestrator statistics
        self.orchestrator_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Simple response cache
        self.response_cache = {}
        self.cache_enabled = True
        self.max_cache_size = 1000
        
        logger.info(f"Initialized GeneratorOrchestrator with model: {model_name}")
    
    def generate_grounded_answer(self, 
                               query: str, 
                               intent: str, 
                               evidence_documents: List[Dict[str, Any]],
                               previous_context: str = None,
                               use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate grounded answer with full pipeline
        
        Args:
            query: User query
            intent: Query intent (fact, explain, compare, etc.)
            evidence_documents: Retrieved and reranked evidence documents
            previous_context: Previous conversation context
            use_cache: Whether to use response caching
        
        Returns:
            Complete grounded answer with metadata
        """
        
        start_time = time.time()
        self.orchestrator_stats["total_requests"] += 1
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, intent, evidence_documents)
            
            if use_cache and self.cache_enabled and cache_key in self.response_cache:
                self.orchestrator_stats["cache_hits"] += 1
                cached_response = self.response_cache[cache_key].copy()
                cached_response["metadata"]["from_cache"] = True
                cached_response["metadata"]["cache_timestamp"] = time.time()
                return cached_response
            
            self.orchestrator_stats["cache_misses"] += 1
            
            # Step 1: Generate prompt using template manager
            prompt = self.prompt_manager.get_prompt(
                query=query,
                intent=intent,
                evidence_documents=evidence_documents,
                previous_context=previous_context
            )
            
            # Adapt prompt for specific model
            adapted_prompt = self.prompt_manager.get_model_specific_prompt(prompt, self.model_name)
            
            # Step 2: Generate answer using LLM
            logger.debug("Generating answer with LLM...")
            raw_answer = self.answer_generator.generate_answer(adapted_prompt, evidence_documents)
            logger.debug(f"Raw answer keys: {list(raw_answer.keys())}")
            
            # Step 3: Extract and validate citations
            logger.debug("Extracting citations...")
            references = self.citation_manager.extract_evidence_spans(
                answer_text=raw_answer.get("detailed_answer", ""),
                evidence_documents=evidence_documents,
                query=query
            )
            
            # Update answer with extracted references
            raw_answer["references"] = self.citation_manager.format_references_for_output(references)
            # Ensure metadata is always a dictionary
            if "metadata" not in raw_answer or not isinstance(raw_answer.get("metadata"), dict):
                raw_answer["metadata"] = {}
            
            # Step 4: Estimate confidence
            logger.debug("Estimating confidence...")
            confidence_score = self.confidence_estimator.estimate_confidence(
                answer=raw_answer,
                evidence_documents=evidence_documents,
                query=query,
                model_metadata=raw_answer.get("metadata", {})
            )
            logger.debug(f"Confidence score: {confidence_score}")
            
            raw_answer["confidence"] = confidence_score
            
            # Step 5: Format output
            logger.debug("Formatting output...")
            formatted_answer = self.output_formatter.format_output(
                raw_answer=raw_answer,
                evidence_documents=evidence_documents,
                metadata=raw_answer.get("metadata", {})
            )
            logger.debug(f"Formatted answer keys: {list(formatted_answer.keys())}")
            
            # Step 6: Final validation and quality checks
            final_answer = self._perform_quality_checks(formatted_answer, query, evidence_documents)
            
            # Add orchestrator metadata
            processing_time = time.time() - start_time
            if "metadata" not in final_answer:
                final_answer["metadata"] = {}
            final_answer["metadata"]["orchestrator_processing_time"] = processing_time
            final_answer["metadata"]["pipeline_version"] = "1.0.0"
            
            # Cache the response
            if use_cache and self.cache_enabled:
                self._cache_response(cache_key, final_answer)
            
            # Update statistics
            self._update_orchestrator_stats(True, processing_time)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            processing_time = time.time() - start_time
            self._update_orchestrator_stats(False, processing_time)
            
            # Return error response
            return self._create_error_response(query, str(e), evidence_documents)
    
    def generate_batch_answers(self, 
                             queries: List[str], 
                             intents: List[str], 
                             evidence_batches: List[List[Dict[str, Any]]],
                             previous_contexts: List[str] = None,
                             use_parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries in batch
        
        Args:
            queries: List of user queries
            intents: List of query intents
            evidence_batches: List of evidence document lists
            previous_contexts: List of previous contexts (optional)
            use_parallel: Whether to use parallel processing
        
        Returns:
            List of grounded answers
        """
        
        if len(queries) != len(intents) or len(queries) != len(evidence_batches):
            raise ValueError("Queries, intents, and evidence batches must have same length")
        
        if not previous_contexts:
            previous_contexts = [None] * len(queries)
        
        if use_parallel and self.enable_async and len(queries) > 1:
            return self._generate_batch_parallel(queries, intents, evidence_batches, previous_contexts)
        else:
            return self._generate_batch_sequential(queries, intents, evidence_batches, previous_contexts)
    
    def _generate_batch_sequential(self, 
                                 queries: List[str], 
                                 intents: List[str], 
                                 evidence_batches: List[List[Dict[str, Any]]], 
                                 previous_contexts: List[str]) -> List[Dict[str, Any]]:
        """Generate batch answers sequentially"""
        
        results = []
        
        for query, intent, evidence_docs, context in zip(queries, intents, evidence_batches, previous_contexts):
            result = self.generate_grounded_answer(query, intent, evidence_docs, context)
            results.append(result)
        
        return results
    
    def _generate_batch_parallel(self, 
                               queries: List[str], 
                               intents: List[str], 
                               evidence_batches: List[List[Dict[str, Any]]], 
                               previous_contexts: List[str]) -> List[Dict[str, Any]]:
        """Generate batch answers in parallel"""
        
        futures = []
        
        for query, intent, evidence_docs, context in zip(queries, intents, evidence_batches, previous_contexts):
            future = self.executor.submit(
                self.generate_grounded_answer, 
                query, intent, evidence_docs, context
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # 60 second timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel generation failed: {str(e)}")
                error_result = self._create_error_response("", str(e), [])
                results.append(error_result)
        
        return results
    
    def _perform_quality_checks(self, 
                              answer: Dict[str, Any], 
                              query: str, 
                              evidence_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform final quality checks on the answer"""
        
        quality_checks = {
            "citation_validation": False,
            "format_validation": False,
            "content_quality": False,
            "confidence_reasonableness": False
        }
        
        # Citation validation
        citation_validation = self.citation_manager.validate_citations(
            answer_text=answer.get("detailed_answer", ""),
            references=answer.get("references", []),
            evidence_documents=evidence_documents
        )
        
        quality_checks["citation_validation"] = citation_validation["validation_score"] > 0.7
        
        # Format validation
        format_validation = self.output_formatter._validate_output(answer)
        quality_checks["format_validation"] = format_validation["is_valid"]
        
        # Content quality (basic checks)
        detailed_answer = answer.get("detailed_answer", "")
        summary = answer.get("summary", "")
        
        quality_checks["content_quality"] = (
            len(detailed_answer) > 50 and 
            len(summary) > 10 and 
            detailed_answer != summary
        )
        
        # Confidence reasonableness
        confidence = answer.get("confidence", 0.0)
        quality_checks["confidence_reasonableness"] = 0.1 <= confidence <= 0.95
        
        # Add quality metadata
        if "metadata" not in answer:
            answer["metadata"] = {}
        answer["metadata"]["quality_checks"] = quality_checks
        answer["metadata"]["overall_quality_score"] = sum(quality_checks.values()) / len(quality_checks)
        
        # Apply quality-based adjustments
        if answer["metadata"]["overall_quality_score"] < 0.5:
            # Lower confidence for poor quality answers
            answer["confidence"] = min(answer["confidence"], 0.6)
            
            # Add quality warning
            if "warnings" not in answer["metadata"]:
                answer["metadata"]["warnings"] = []
            answer["metadata"]["warnings"].append("Answer quality below threshold")
        
        return answer
    
    def _generate_cache_key(self, 
                          query: str, 
                          intent: str, 
                          evidence_documents: List[Dict[str, Any]]) -> str:
        """Generate cache key for response caching"""
        
        # Create hash from query, intent, and evidence document IDs
        import hashlib
        
        evidence_ids = [doc.get('id', '') for doc in evidence_documents]
        cache_input = f"{query}|{intent}|{','.join(sorted(evidence_ids))}"
        
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response with size management"""
        
        if len(self.response_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.response_cache.keys())[:self.max_cache_size // 4]
            for key in oldest_keys:
                del self.response_cache[key]
        
        # Store response with timestamp
        cached_response = response.copy()
        cached_response["metadata"]["cached_at"] = time.time()
        
        self.response_cache[cache_key] = cached_response
    
    def _create_error_response(self, 
                             query: str, 
                             error_message: str, 
                             evidence_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create error response when generation fails"""
        
        return {
            "summary": "Answer generation failed",
            "detailed_answer": f"I apologize, but I encountered an error while generating the answer: {error_message}",
            "references": [],
            "confidence": 0.0,
            "metadata": {
                "error": error_message,
                "model_used": self.model_name,
                "evidence_count": len(evidence_documents),
                "generation_time": 0.0,
                "pipeline_version": "1.0.0"
            }
        }
    
    def _update_orchestrator_stats(self, success: bool, processing_time: float):
        """Update orchestrator statistics"""
        
        if success:
            self.orchestrator_stats["successful_generations"] += 1
        else:
            self.orchestrator_stats["failed_generations"] += 1
        
        # Update average processing time
        total_time = (self.orchestrator_stats["avg_processing_time"] * 
                     (self.orchestrator_stats["total_requests"] - 1) + processing_time)
        self.orchestrator_stats["avg_processing_time"] = total_time / self.orchestrator_stats["total_requests"]
    
    def update_model(self, model_name: str, device: str = None):
        """Update the language model"""
        
        if device is None:
            device = self.device
        
        self.model_name = model_name
        self.device = device
        
        # Reinitialize answer generator with new model
        self.answer_generator = AnswerGenerator(model_name=model_name, device=device)
        
        # Clear cache since model changed
        self.clear_cache()
        
        logger.info(f"Updated model to: {model_name}")
    
    def update_generation_config(self, **kwargs):
        """Update generation configuration"""
        
        self.answer_generator.update_generation_config(**kwargs)
        
        # Clear cache since generation config changed
        self.clear_cache()
    
    def update_confidence_weights(self, **kwargs):
        """Update confidence estimation weights"""
        
        self.confidence_estimator.update_weights(**kwargs)
    
    def update_output_constraints(self, **kwargs):
        """Update output formatting constraints"""
        
        self.output_formatter.update_schema_constraints(**kwargs)
    
    def add_custom_template(self, intent: str, system_prompt: str, user_template: str):
        """Add custom prompt template"""
        
        self.prompt_manager.add_custom_template(intent, system_prompt, user_template)
    
    def clear_cache(self):
        """Clear response cache"""
        
        self.response_cache.clear()
        logger.info("Cleared response cache")
    
    def enable_cache(self, enabled: bool = True):
        """Enable or disable response caching"""
        
        self.cache_enabled = enabled
        logger.info(f"Response caching {'enabled' if enabled else 'disabled'}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        return {
            "orchestrator": {
                "model_name": self.model_name,
                "device": self.device,
                "max_workers": self.max_workers,
                "async_enabled": self.enable_async,
                "cache_enabled": self.cache_enabled,
                "cache_size": len(self.response_cache)
            },
            "components": {
                "answer_generator": self.answer_generator.get_model_info(),
                "prompt_manager": self.prompt_manager.get_template_stats(),
                "citation_manager": self.citation_manager.get_stats(),
                "confidence_estimator": self.confidence_estimator.get_confidence_stats(),
                "output_formatter": self.output_formatter.get_formatting_stats()
            }
        }
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        
        stats = self.orchestrator_stats.copy()
        
        # Calculate derived metrics
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_generations"] / stats["total_requests"]
            stats["failure_rate"] = stats["failed_generations"] / stats["total_requests"]
            
            total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
            if total_cache_requests > 0:
                stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        
        return stats
    
    def clear_all_stats(self):
        """Clear all component statistics"""
        
        self.orchestrator_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.answer_generator.clear_stats()
        self.confidence_estimator.clear_stats()
        self.output_formatter.clear_stats()
        
        logger.info("Cleared all statistics")
    
    def shutdown(self):
        """Shutdown the orchestrator and cleanup resources"""
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.clear_cache()
        
        logger.info("GeneratorOrchestrator shutdown complete")