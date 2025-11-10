"""
Generator Services - Unified Interface

This module provides a single, easy-to-use interface for all generator functionality:
- Answer generation using multiple strategies
- LLM judge validation and quality assurance
- Enhanced response formatting and citation management
- Performance monitoring and optimization

Usage:
    from generator.generator_services import GeneratorServices
    
    # Initialize services
    generator = GeneratorServices()
    
    # Generate answer
    result = generator.generate_answer("What is karma?", evidence_documents)
    
    # Generate with specific strategy
    result = generator.generate_with_strategy("Compare Rama and Krishna", evidence_documents, "grounded")
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import existing generator components
from .generator_orchestrator import GeneratorOrchestrator
from .enhanced_generator_orchestrator import EnhancedGeneratorOrchestrator
from .answer_generator import AnswerGenerator
from .llm_judge_validator import LLMJudgeValidator
from .enhanced_response_formatter import EnhancedResponseFormatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratorServices:
    """
    Unified interface for all generator functionality.
    Provides easy access to answer generation, validation, and formatting.
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/llama-3.3-70b-instruct:free",
                 enable_enhanced_orchestrator: bool = True,
                 enable_llm_judge: bool = True,
                 enable_enhanced_formatting: bool = True,
                 enable_caching: bool = True,
                 device: str = "auto"):
        """
        Initialize GeneratorServices
        
        Args:
            model_name: LLM model to use for generation
            enable_enhanced_orchestrator: Use enhanced orchestrator with advanced features
            enable_llm_judge: Enable LLM judge validation
            enable_enhanced_formatting: Enable enhanced response formatting
            enable_caching: Enable result caching
            device: Device to run models on ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.enable_enhanced_orchestrator = enable_enhanced_orchestrator
        self.enable_llm_judge = enable_llm_judge
        self.enable_enhanced_formatting = enable_enhanced_formatting
        self.enable_caching = enable_caching
        self.device = device
        
        # Initialize core components using existing orchestrators
        try:
            if enable_enhanced_orchestrator:
                self.generator_orchestrator = EnhancedGeneratorOrchestrator(
                    model_name=model_name,
                    device=device
                )
            else:
                self.generator_orchestrator = GeneratorOrchestrator(
                    model_name=model_name,
                    device=device
                )
            
            self.answer_generator = AnswerGenerator(
                model_name=model_name,
                device=device
            )
            
            if enable_llm_judge:
                self.llm_judge_validator = LLMJudgeValidator()
            else:
                self.llm_judge_validator = None
            
            if enable_enhanced_formatting:
                self.response_formatter = EnhancedResponseFormatter()
            else:
                self.response_formatter = None
            
            logger.info("✅ GeneratorServices initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize GeneratorServices: {str(e)}")
            # Initialize with minimal functionality
            self.generator_orchestrator = None
            self.answer_generator = None
            self.llm_judge_validator = None
            self.response_formatter = None
        
        # Service statistics
        self.stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0.0,
            "avg_confidence_score": 0.0,
            "avg_quality_score": 0.0,
            "strategy_usage": {},
            "validation_usage": 0,
            "enhanced_formatting_usage": 0
        }
        
        # Result cache
        self.cache = {} if enable_caching else None
        self.cache_ttl = 1800  # 30 minutes    

    def generate_answer(self, 
                       query: str,
                       evidence_documents: List[Dict[str, Any]],
                       intent: str = "fact",
                       strategy: str = "grounded",
                       enable_validation: bool = None,
                       enable_enhanced_formatting: bool = None) -> Dict[str, Any]:
        """
        Generate answer using specified strategy
        
        Args:
            query: User query
            evidence_documents: List of evidence documents
            intent: Query intent ("fact", "explain", "compare", etc.)
            strategy: Generation strategy ("grounded", "direct", "enhanced")
            enable_validation: Enable LLM judge validation (overrides default)
            enable_enhanced_formatting: Enable enhanced formatting (overrides default)
            
        Returns:
            Dictionary containing generated answer and metadata
        """
        start_time = time.time()
        
        # Use defaults if not specified
        if enable_validation is None:
            enable_validation = self.enable_llm_judge
        if enable_enhanced_formatting is None:
            enable_enhanced_formatting = self.enable_enhanced_formatting
        
        try:
            # Validate inputs
            if not query.strip():
                return self._create_error_result("Empty query provided")
            
            # Check cache first
            if self.enable_caching:
                cached_result = self._check_cache(query, evidence_documents, intent, strategy)
                if cached_result:
                    logger.info(f"📋 Cache hit for generation query: {query[:50]}...")
                    return cached_result
            
            # Generate answer using orchestrator
            if self.generator_orchestrator:
                generated_result = self.generator_orchestrator.generate_grounded_answer(
                    query=query,
                    intent=intent,
                    evidence_documents=evidence_documents
                )
            else:
                # Fallback to direct answer generator
                generated_result = self._fallback_generation(query, evidence_documents, intent, strategy)
            
            # Apply LLM judge validation if enabled
            validation_result = None
            if enable_validation and self.llm_judge_validator:
                try:
                    validation_result = self.llm_judge_validator.validate_answer(
                        generated_answer=generated_result,
                        evidence_documents=evidence_documents,
                        query=query,
                        intent=intent
                    )
                    self.stats["validation_usage"] += 1
                    logger.info(f"🧠 LLM judge validation applied")
                except Exception as e:
                    logger.warning(f"LLM judge validation failed: {str(e)}")
                    validation_result = None
            
            # Apply enhanced formatting if enabled
            formatted_result = generated_result
            if enable_enhanced_formatting and self.response_formatter:
                try:
                    formatted_result = self.response_formatter.format_enhanced_response(
                        raw_answer=generated_result,
                        evidence_documents=evidence_documents
                    )
                    self.stats["enhanced_formatting_usage"] += 1
                    logger.info(f"📝 Enhanced formatting applied")
                except Exception as e:
                    logger.warning(f"Enhanced formatting failed: {str(e)}")
                    formatted_result = generated_result
            
            # Add generation metadata
            generation_time = time.time() - start_time
            final_result = {
                'generated_answer': formatted_result,
                'validation_result': validation_result,
                'generation_metadata': {
                    'generation_time': generation_time,
                    'strategy_used': strategy,
                    'model_used': self.model_name,
                    'intent': intent,
                    'validation_applied': enable_validation and validation_result is not None,
                    'enhanced_formatting_applied': enable_enhanced_formatting and formatted_result != generated_result,
                    'evidence_count': len(evidence_documents),
                    'timestamp': datetime.now().isoformat(),
                    'service_version': '1.0.0'
                },
                'success': True
            }
            
            # Cache result
            if self.enable_caching:
                self._cache_result(query, evidence_documents, intent, strategy, final_result)
            
            # Update statistics
            confidence_score = formatted_result.get('confidence', 0.0)
            quality_score = self._calculate_quality_score(formatted_result, validation_result)
            self._update_stats(True, generation_time, confidence_score, quality_score, strategy)
            
            logger.info(f"✅ Answer generation completed in {generation_time:.3f}s")
            return final_result
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_result = {
                'generated_answer': {
                    'summary': 'Generation failed',
                    'detailed_answer': f'Unable to generate answer: {str(e)}',
                    'references': [],
                    'confidence': 0.0
                },
                'validation_result': None,
                'generation_metadata': {
                    'generation_time': generation_time,
                    'strategy_used': strategy,
                    'model_used': self.model_name,
                    'intent': intent,
                    'validation_applied': False,
                    'enhanced_formatting_applied': False,
                    'evidence_count': len(evidence_documents) if evidence_documents else 0,
                    'timestamp': datetime.now().isoformat(),
                    'error_occurred': True,
                    'error_message': str(e)
                },
                'success': False,
                'error': str(e)
            }
            
            self._update_stats(False, generation_time, 0.0, 0.0, strategy)
            logger.error(f"❌ Answer generation failed for query '{query[:50]}...': {str(e)}")
            return error_result
    
    def generate_with_strategy(self, 
                             query: str, 
                             evidence_documents: List[Dict[str, Any]], 
                             strategy: str,
                             intent: str = "fact",
                             **kwargs) -> Dict[str, Any]:
        """
        Generate answer with a specific strategy
        
        Args:
            query: User query
            evidence_documents: List of evidence documents
            strategy: Specific strategy ("grounded", "direct", "enhanced")
            intent: Query intent
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Dictionary containing generated answer
        """
        return self.generate_answer(
            query=query,
            evidence_documents=evidence_documents,
            intent=intent,
            strategy=strategy,
            **kwargs
        )
    
    def generate_batch(self, 
                      queries: List[str],
                      evidence_batches: List[List[Dict[str, Any]]],
                      intents: List[str] = None,
                      strategy: str = "grounded") -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries in batch
        
        Args:
            queries: List of user queries
            evidence_batches: List of evidence document lists (one per query)
            intents: List of query intents (one per query)
            strategy: Generation strategy to use for all queries
            
        Returns:
            List of generation results for each query
        """
        if len(queries) != len(evidence_batches):
            raise ValueError("Number of queries must match number of evidence batches")
        
        if intents is None:
            intents = ["fact"] * len(queries)
        elif len(intents) != len(queries):
            raise ValueError("Number of intents must match number of queries")
        
        results = []
        
        for i, (query, evidence_docs, intent) in enumerate(zip(queries, evidence_batches, intents)):
            logger.info(f"🔄 Processing batch generation {i+1}/{len(queries)}: {query[:50]}...")
            
            result = self.generate_answer(
                query=query,
                evidence_documents=evidence_docs,
                intent=intent,
                strategy=strategy
            )
            
            result['batch_index'] = i
            results.append(result)
        
        logger.info(f"✅ Batch generation completed: {len(queries)} queries processed")
        return results
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available generation strategies
        
        Returns:
            List of strategy names
        """
        strategies = ["direct"]  # Always available through answer generator
        
        if self.generator_orchestrator:
            strategies.append("grounded")
            
            if self.enable_enhanced_orchestrator:
                strategies.extend(["enhanced", "adaptive"])
        
        return strategies
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive generation statistics
        
        Returns:
            Dictionary containing performance metrics and usage statistics
        """
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats["total_generations"] > 0:
            stats["success_rate"] = stats["successful_generations"] / stats["total_generations"]
            stats["failure_rate"] = stats["failed_generations"] / stats["total_generations"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        # Add component usage rates
        if stats["total_generations"] > 0:
            stats["validation_usage_rate"] = stats["validation_usage"] / stats["total_generations"]
            stats["enhanced_formatting_usage_rate"] = stats["enhanced_formatting_usage"] / stats["total_generations"]
        else:
            stats["validation_usage_rate"] = 0.0
            stats["enhanced_formatting_usage_rate"] = 0.0
        
        # Add cache statistics
        if self.cache:
            stats["cache_size"] = len(self.cache)
            stats["cache_enabled"] = True
        else:
            stats["cache_size"] = 0
            stats["cache_enabled"] = False
        
        return stats
    
    def optimize_performance(self) -> Dict[str, Any]:
        """
        Analyze performance and provide optimization recommendations
        
        Returns:
            Dictionary containing performance analysis and recommendations
        """
        stats = self.get_generation_stats()
        recommendations = []
        
        # Analyze performance metrics
        if stats["success_rate"] < 0.9:
            recommendations.append("Consider enabling fallback strategies for better reliability")
        
        if stats["avg_generation_time"] > 10.0:
            recommendations.append("Generation time is high - consider using a smaller/faster model")
        
        if stats["avg_confidence_score"] < 0.5:
            recommendations.append("Low confidence scores - consider improving evidence quality or using validation")
        
        if stats["avg_quality_score"] < 0.6:
            recommendations.append("Low quality scores - consider enabling LLM judge validation")
        
        if not self.enable_llm_judge and stats["total_generations"] > 10:
            recommendations.append("Consider enabling LLM judge validation for better quality assurance")
        
        if not self.enable_enhanced_formatting and stats["total_generations"] > 10:
            recommendations.append("Consider enabling enhanced formatting for better response quality")
        
        # Strategy-specific recommendations
        strategy_usage = stats.get("strategy_usage", {})
        if strategy_usage:
            most_used = max(strategy_usage.items(), key=lambda x: x[1])
            recommendations.append(f"Most used strategy: {most_used[0]} ({most_used[1]} times)")
        
        return {
            "performance_analysis": stats,
            "recommendations": recommendations,
            "optimization_score": min(stats["success_rate"] * stats["avg_quality_score"] * 100, 100),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self) -> bool:
        """
        Clear the result cache
        
        Returns:
            True if cache was cleared, False if caching is disabled
        """
        if self.cache:
            cache_size = len(self.cache)
            self.cache.clear()
            logger.info(f"🗑️ Cache cleared: {cache_size} entries removed")
            return True
        else:
            logger.info("📋 No cache to clear (caching disabled)")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on generator services
        
        Returns:
            Dictionary containing health status and component availability
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "issues": []
        }
        
        # Check generator orchestrator
        if self.generator_orchestrator:
            try:
                # Test basic functionality
                system_info = self.generator_orchestrator.get_system_info()
                health_status["components"]["generator_orchestrator"] = "available"
                if self.enable_enhanced_orchestrator:
                    health_status["components"]["orchestrator_type"] = "enhanced"
                else:
                    health_status["components"]["orchestrator_type"] = "standard"
            except Exception as e:
                health_status["components"]["generator_orchestrator"] = f"error: {str(e)}"
                health_status["issues"].append("Generator orchestrator not functioning properly")
                health_status["status"] = "degraded"
        else:
            health_status["components"]["generator_orchestrator"] = "not_available"
            health_status["issues"].append("Generator orchestrator not initialized")
            health_status["status"] = "degraded"
        
        # Check answer generator
        if self.answer_generator:
            try:
                model_info = self.answer_generator.get_model_info()
                health_status["components"]["answer_generator"] = "available"
                health_status["components"]["model_name"] = self.model_name
            except Exception as e:
                health_status["components"]["answer_generator"] = f"error: {str(e)}"
                health_status["issues"].append("Answer generator not functioning properly")
                health_status["status"] = "degraded"
        else:
            health_status["components"]["answer_generator"] = "not_available"
            health_status["issues"].append("Answer generator not initialized")
            health_status["status"] = "unhealthy"
        
        # Check LLM judge validator
        if self.llm_judge_validator:
            health_status["components"]["llm_judge_validator"] = "available"
        else:
            health_status["components"]["llm_judge_validator"] = "not_available" if self.enable_llm_judge else "disabled"
        
        # Check response formatter
        if self.response_formatter:
            health_status["components"]["response_formatter"] = "available"
        else:
            health_status["components"]["response_formatter"] = "not_available" if self.enable_enhanced_formatting else "disabled"
        
        # Check cache
        if self.enable_caching:
            health_status["components"]["cache"] = f"enabled ({len(self.cache) if self.cache else 0} entries)"
        else:
            health_status["components"]["cache"] = "disabled"
        
        # Overall status
        if len(health_status["issues"]) > 2:
            health_status["status"] = "unhealthy"
        elif len(health_status["issues"]) > 0:
            health_status["status"] = "degraded"
        
        return health_status
    
    def _fallback_generation(self, 
                           query: str, 
                           evidence_documents: List[Dict[str, Any]], 
                           intent: str, 
                           strategy: str) -> Dict[str, Any]:
        """
        Fallback generation when orchestrator is not available
        """
        if self.answer_generator:
            try:
                # Create simple prompt
                prompt = {
                    "system_prompt": f"You are a helpful assistant. Answer the question based on the provided evidence. Intent: {intent}",
                    "user_prompt": f"Question: {query}\n\nAnswer based on the evidence provided."
                }
                
                result = self.answer_generator.generate_answer(prompt, evidence_documents)
                return result
            except Exception as e:
                logger.error(f"Fallback generation failed: {str(e)}")
        
        # Ultimate fallback - return basic response
        return {
            'summary': 'Unable to generate answer',
            'detailed_answer': f'I apologize, but I cannot generate an answer for: {query}',
            'references': [],
            'confidence': 0.0,
            'metadata': {
                'fallback_used': True,
                'error': 'No generation methods available'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error generation result"""
        return {
            'generated_answer': {
                'summary': 'Generation failed',
                'detailed_answer': f'Unable to generate answer: {error_message}',
                'references': [],
                'confidence': 0.0
            },
            'validation_result': None,
            'generation_metadata': {
                'generation_time': 0.0,
                'strategy_used': 'none',
                'model_used': self.model_name,
                'intent': 'unknown',
                'validation_applied': False,
                'enhanced_formatting_applied': False,
                'evidence_count': 0,
                'timestamp': datetime.now().isoformat(),
                'error_occurred': True,
                'error_message': error_message
            },
            'success': False,
            'error': error_message
        }
    
    def _check_cache(self, query: str, evidence_documents: List[Dict[str, Any]], intent: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Check if generation result is cached"""
        if not self.cache:
            return None
        
        # Create cache key based on query, evidence IDs, intent, and strategy
        evidence_hash = hash(str(sorted([doc.get('id', str(hash(doc.get('content', '')))) for doc in evidence_documents])))
        cache_key = f"{query}_{intent}_{strategy}_{evidence_hash}"
        
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            
            # Check TTL
            if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                cached_entry['result']['from_cache'] = True
                return cached_entry['result']
            else:
                del self.cache[cache_key]
        
        return None
    
    def _cache_result(self, query: str, evidence_documents: List[Dict[str, Any]], intent: str, strategy: str, result: Dict[str, Any]):
        """Cache generation result"""
        if not self.cache:
            return
        
        evidence_hash = hash(str(sorted([doc.get('id', str(hash(doc.get('content', '')))) for doc in evidence_documents])))
        cache_key = f"{query}_{intent}_{strategy}_{evidence_hash}"
        
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Limit cache size
        if len(self.cache) > 200:
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]['timestamp']
            )[:50]
            
            for key in oldest_keys:
                del self.cache[key]
    
    def _calculate_quality_score(self, generated_answer: Dict[str, Any], validation_result: Any) -> float:
        """Calculate quality score for generated answer"""
        quality_factors = []
        
        # Content quality
        detailed_answer = generated_answer.get('detailed_answer', '')
        content_score = min(len(detailed_answer) / 200, 1.0)
        quality_factors.append(content_score)
        
        # Reference quality
        references = generated_answer.get('references', [])
        reference_score = min(len(references) / 3, 1.0)
        quality_factors.append(reference_score)
        
        # Confidence score
        confidence = generated_answer.get('confidence', 0.5)
        quality_factors.append(confidence)
        
        # Validation score (if available)
        if validation_result and hasattr(validation_result, 'overall_confidence'):
            validation_score = validation_result.overall_confidence
            quality_factors.append(validation_score)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _update_stats(self, success: bool, generation_time: float, confidence_score: float, quality_score: float, strategy: str):
        """Update service statistics"""
        self.stats["total_generations"] += 1
        
        if success:
            self.stats["successful_generations"] += 1
        else:
            self.stats["failed_generations"] += 1
        
        # Update averages
        total_generations = self.stats["total_generations"]
        
        # Average generation time
        total_time = (self.stats["avg_generation_time"] * (total_generations - 1) + generation_time)
        self.stats["avg_generation_time"] = total_time / total_generations
        
        # Average scores (only for successful generations)
        if success:
            successful_generations = self.stats["successful_generations"]
            
            # Average confidence score
            total_confidence = (self.stats["avg_confidence_score"] * (successful_generations - 1) + confidence_score)
            self.stats["avg_confidence_score"] = total_confidence / successful_generations
            
            # Average quality score
            total_quality = (self.stats["avg_quality_score"] * (successful_generations - 1) + quality_score)
            self.stats["avg_quality_score"] = total_quality / successful_generations
        
        # Strategy usage
        if strategy not in self.stats["strategy_usage"]:
            self.stats["strategy_usage"][strategy] = 0
        self.stats["strategy_usage"][strategy] += 1


# Convenience functions for easy usage
def generate_query(query: str, evidence_documents: List[Dict[str, Any]], intent: str = "fact", strategy: str = "grounded") -> Dict[str, Any]:
    """
    Convenience function to generate answer for a single query
    
    Args:
        query: User query
        evidence_documents: List of evidence documents
        intent: Query intent
        strategy: Generation strategy to use
        
    Returns:
        Dictionary containing generated answer
    """
    generator = GeneratorServices()
    return generator.generate_answer(query, evidence_documents, intent, strategy)


def get_generator_health() -> Dict[str, Any]:
    """
    Convenience function to check generator health
    
    Returns:
        Dictionary containing health status
    """
    generator = GeneratorServices()
    return generator.health_check()


# Example usage and testing
if __name__ == "__main__":
    # Initialize services
    print("🚀 Initializing GeneratorServices...")
    generator_services = GeneratorServices(
        model_name="meta-llama/llama-3.3-70b-instruct:free",
        enable_enhanced_orchestrator=True,
        enable_llm_judge=True,
        enable_enhanced_formatting=True,
        enable_caching=True
    )
    
    # Test health check
    print("\n🏥 Health Check:")
    health = generator_services.health_check()
    print(f"Status: {health['status']}")
    print(f"Components: {health['components']}")
    
    # Test generation
    print("\n🤖 Testing Answer Generation:")
    test_query = "What is artificial intelligence?"
    
    # Sample evidence documents
    test_evidence = [
        {
            'id': 'doc_1',
            'content': 'Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans.',
            'similarity_score': 0.9
        },
        {
            'id': 'doc_2',
            'content': 'Machine learning is a method of data analysis that automates analytical model building using algorithms.',
            'similarity_score': 0.8
        }
    ]
    
    result = generator_services.generate_answer(
        query=test_query,
        evidence_documents=test_evidence,
        intent="fact",
        strategy="grounded"
    )
    
    print(f"Query: {test_query}")
    print(f"Evidence documents: {len(test_evidence)}")
    print(f"Success: {result['success']}")
    print(f"Generation time: {result['generation_metadata']['generation_time']:.3f}s")
    
    if result['success']:
        answer = result['generated_answer']
        print(f"Summary: {answer.get('summary', '')}")
        print(f"Answer: {answer.get('detailed_answer', '')[:100]}...")
        print(f"Confidence: {answer.get('confidence', 0):.3f}")
    
    # Get statistics
    print("\n📊 Service Statistics:")
    stats = generator_services.get_generation_stats()
    print(f"Total generations: {stats['total_generations']}")
    print(f"Success rate: {stats['success_rate']:.3f}")
    print(f"Average generation time: {stats['avg_generation_time']:.3f}s")
    print(f"Average confidence: {stats['avg_confidence_score']:.3f}")
    print(f"Average quality: {stats['avg_quality_score']:.3f}")
    
    # Get optimization recommendations
    print("\n🎯 Performance Optimization:")
    optimization = generator_services.optimize_performance()
    print(f"Optimization score: {optimization['optimization_score']:.1f}/100")
    print("Recommendations:")
    for rec in optimization['recommendations']:
        print(f"  • {rec}")
    
    print("\n✅ GeneratorServices testing completed!")