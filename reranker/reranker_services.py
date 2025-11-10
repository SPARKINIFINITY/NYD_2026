"""
Reranker Services - Unified Interface

This module provides a single, easy-to-use interface for all reranker functionality:
- Document reranking using multiple strategies
- Cross-encoder and cascade reranking
- Signal processing and advanced reranking
- Performance monitoring and optimization

Usage:
    from reranker.reranker_services import RerankerServices
    
    # Initialize services
    reranker = RerankerServices()
    
    # Rerank documents
    results = reranker.rerank_documents("What is karma?", documents)
    
    # Rerank with specific strategy
    results = reranker.rerank_with_strategy("Compare Rama and Krishna", documents, "cascade")
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import existing reranker components
from .reranker_orchestrator import RerankerOrchestrator
from .cascade_reranker import CascadeReranker
from .cross_encoder_reranker import CrossEncoderReranker
from .advanced_reranker import AdvancedReranker
from .signal_processor import SignalProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankerServices:
    """
    Unified interface for all reranker functionality.
    Provides easy access to document reranking, optimization, and performance monitoring.
    """
    
    def __init__(self, 
                 enable_cascade: bool = True,
                 enable_cross_encoder: bool = True,
                 enable_signal_processing: bool = True,
                 enable_caching: bool = True,
                 default_top_k: int = 10):
        """
        Initialize RerankerServices
        
        Args:
            enable_cascade: Enable cascade reranking
            enable_cross_encoder: Enable cross-encoder reranking
            enable_signal_processing: Enable signal processing
            enable_caching: Enable result caching
            default_top_k: Default number of results to return
        """
        self.enable_cascade = enable_cascade
        self.enable_cross_encoder = enable_cross_encoder
        self.enable_signal_processing = enable_signal_processing
        self.enable_caching = enable_caching
        self.default_top_k = default_top_k
        
        # Initialize core components using existing orchestrators
        try:
            self.reranker_orchestrator = RerankerOrchestrator()
            
            if enable_cascade:
                self.cascade_reranker = CascadeReranker()
            else:
                self.cascade_reranker = None
            
            if enable_cross_encoder:
                self.cross_encoder_reranker = CrossEncoderReranker()
            else:
                self.cross_encoder_reranker = None
            
            if enable_signal_processing:
                self.signal_processor = SignalProcessor()
            else:
                self.signal_processor = None
            
            self.advanced_reranker = AdvancedReranker()
            
            logger.info("✅ RerankerServices initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RerankerServices: {str(e)}")
            # Initialize with minimal functionality
            self.reranker_orchestrator = None
            self.cascade_reranker = None
            self.cross_encoder_reranker = None
            self.signal_processor = None
            self.advanced_reranker = None
        
        # Service statistics
        self.stats = {
            "total_rerankings": 0,
            "successful_rerankings": 0,
            "failed_rerankings": 0,
            "avg_reranking_time": 0.0,
            "avg_quality_improvement": 0.0,
            "strategy_usage": {},
            "cascade_usage": 0,
            "cross_encoder_usage": 0
        }
        
        # Result cache
        self.cache = {} if enable_caching else None
        self.cache_ttl = 1800  # 30 minutes
    
    def rerank_documents(self, 
                        query: str,
                        documents: List[Dict[str, Any]],
                        strategy: str = "cascade",
                        top_k: int = None,
                        enable_signal_processing: bool = None) -> Dict[str, Any]:
        """
        Rerank documents using specified strategy
        
        Args:
            query: Search query
            documents: List of documents to rerank
            strategy: Reranking strategy ("cascade", "cross_encoder", "advanced", "hybrid")
            top_k: Number of top results to return (overrides default)
            enable_signal_processing: Enable signal processing (overrides default)
            
        Returns:
            Dictionary containing reranked documents and metadata
        """
        start_time = time.time()
        
        # Use defaults if not specified
        if top_k is None:
            top_k = self.default_top_k
        if enable_signal_processing is None:
            enable_signal_processing = self.enable_signal_processing
        
        try:
            # Validate inputs
            if not documents:
                return self._create_empty_result("No documents to rerank")
            
            # Check cache first
            if self.enable_caching:
                cached_result = self._check_cache(query, documents, strategy)
                if cached_result:
                    logger.info(f"📋 Cache hit for reranking query: {query[:50]}...")
                    return cached_result
            
            # Apply signal processing if enabled
            processed_documents = documents
            if enable_signal_processing and self.signal_processor:
                try:
                    processed_documents = self.signal_processor.process_signals(
                        query=query,
                        candidates=documents
                    )
                    logger.info(f"🔄 Signal processing applied to {len(documents)} documents")
                except Exception as e:
                    logger.warning(f"Signal processing failed: {str(e)}")
                    processed_documents = documents
            
            # Execute reranking using specific strategy (bypass orchestrator for now)
            # The orchestrator doesn't have rerank_documents method, so use fallback
            reranked_results = self._fallback_reranking(query, processed_documents, strategy, top_k)
            
            # Add reranking metadata
            reranking_time = time.time() - start_time
            reranked_results['reranking_metadata'] = {
                'reranking_time': reranking_time,
                'strategy_used': strategy,
                'signal_processing_applied': enable_signal_processing and self.signal_processor is not None,
                'original_count': len(documents),
                'final_count': len(reranked_results.get('reranked_results', [])),
                'timestamp': datetime.now().isoformat(),
                'service_version': '1.0.0'
            }
            
            # Cache result
            if self.enable_caching:
                self._cache_result(query, documents, strategy, reranked_results)
            
            # Update statistics
            quality_improvement = self._calculate_quality_improvement(documents, reranked_results.get('reranked_results', []))
            self._update_stats(True, reranking_time, quality_improvement, strategy)
            
            logger.info(f"✅ Reranking completed: {len(reranked_results.get('reranked_results', []))} results in {reranking_time:.3f}s")
            return reranked_results
            
        except Exception as e:
            reranking_time = time.time() - start_time
            error_result = {
                'reranked_results': documents[:top_k],  # Return original top-k as fallback
                'total_results': len(documents[:top_k]),
                'success': False,
                'error': str(e),
                'reranking_metadata': {
                    'reranking_time': reranking_time,
                    'strategy_used': strategy,
                    'signal_processing_applied': False,
                    'original_count': len(documents),
                    'final_count': len(documents[:top_k]),
                    'timestamp': datetime.now().isoformat(),
                    'error_occurred': True
                }
            }
            
            self._update_stats(False, reranking_time, 0.0, strategy)
            logger.error(f"❌ Reranking failed for query '{query[:50]}...': {str(e)}")
            return error_result
    
    def rerank_with_strategy(self, 
                           query: str, 
                           documents: List[Dict[str, Any]], 
                           strategy: str,
                           **kwargs) -> Dict[str, Any]:
        """
        Rerank documents with a specific strategy
        
        Args:
            query: Search query
            documents: List of documents to rerank
            strategy: Specific strategy ("cascade", "cross_encoder", "advanced", "hybrid")
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Dictionary containing reranked documents
        """
        return self.rerank_documents(
            query=query,
            documents=documents,
            strategy=strategy,
            **kwargs
        )
    
    def rerank_batch(self, 
                    queries: List[str],
                    document_batches: List[List[Dict[str, Any]]],
                    strategy: str = "cascade",
                    top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents for multiple queries in batch
        
        Args:
            queries: List of search queries
            document_batches: List of document lists (one per query)
            strategy: Reranking strategy to use for all queries
            top_k: Number of top results per query
            
        Returns:
            List of reranking results for each query
        """
        if len(queries) != len(document_batches):
            raise ValueError("Number of queries must match number of document batches")
        
        results = []
        
        for i, (query, documents) in enumerate(zip(queries, document_batches)):
            logger.info(f"🔄 Processing batch reranking {i+1}/{len(queries)}: {query[:50]}...")
            
            result = self.rerank_documents(
                query=query,
                documents=documents,
                strategy=strategy,
                top_k=top_k
            )
            
            result['batch_index'] = i
            results.append(result)
        
        logger.info(f"✅ Batch reranking completed: {len(queries)} queries processed")
        return results
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available reranking strategies
        
        Returns:
            List of strategy names
        """
        strategies = ["advanced"]  # Always available
        
        if self.cascade_reranker:
            strategies.append("cascade")
        
        if self.cross_encoder_reranker:
            strategies.append("cross_encoder")
        
        if self.cascade_reranker and self.cross_encoder_reranker:
            strategies.append("hybrid")
        
        # Check if orchestrator has additional strategies
        if self.reranker_orchestrator:
            try:
                additional_strategies = self.reranker_orchestrator.get_available_strategies()
                strategies.extend(additional_strategies)
            except:
                pass
        
        return list(set(strategies))  # Remove duplicates
    
    def get_reranking_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive reranking statistics
        
        Returns:
            Dictionary containing performance metrics and usage statistics
        """
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats["total_rerankings"] > 0:
            stats["success_rate"] = stats["successful_rerankings"] / stats["total_rerankings"]
            stats["failure_rate"] = stats["failed_rerankings"] / stats["total_rerankings"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        # Add component usage rates
        if stats["total_rerankings"] > 0:
            stats["cascade_usage_rate"] = stats["cascade_usage"] / stats["total_rerankings"]
            stats["cross_encoder_usage_rate"] = stats["cross_encoder_usage"] / stats["total_rerankings"]
        else:
            stats["cascade_usage_rate"] = 0.0
            stats["cross_encoder_usage_rate"] = 0.0
        
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
        stats = self.get_reranking_stats()
        recommendations = []
        
        # Analyze performance metrics
        if stats["success_rate"] < 0.9:
            recommendations.append("Consider enabling fallback strategies for better reliability")
        
        if stats["avg_reranking_time"] > 2.0:
            recommendations.append("Reranking time is high - consider optimizing or using faster strategies")
        
        if stats["avg_quality_improvement"] < 0.1:
            recommendations.append("Low quality improvement - consider using more advanced reranking strategies")
        
        if not self.enable_cascade and stats["total_rerankings"] > 10:
            recommendations.append("Consider enabling cascade reranking for better quality")
        
        if not self.enable_cross_encoder and stats["total_rerankings"] > 10:
            recommendations.append("Consider enabling cross-encoder reranking for better accuracy")
        
        # Strategy-specific recommendations
        strategy_usage = stats.get("strategy_usage", {})
        if strategy_usage:
            most_used = max(strategy_usage.items(), key=lambda x: x[1])
            recommendations.append(f"Most used strategy: {most_used[0]} ({most_used[1]} times)")
        
        return {
            "performance_analysis": stats,
            "recommendations": recommendations,
            "optimization_score": min(stats["success_rate"] * stats["avg_quality_improvement"] * 100, 100),
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
        Perform health check on reranker services
        
        Returns:
            Dictionary containing health status and component availability
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "issues": []
        }
        
        # Check reranker orchestrator
        if self.reranker_orchestrator:
            try:
                # Test basic functionality
                test_result = self.reranker_orchestrator.get_available_strategies()
                health_status["components"]["reranker_orchestrator"] = "available"
            except Exception as e:
                health_status["components"]["reranker_orchestrator"] = f"error: {str(e)}"
                health_status["issues"].append("Reranker orchestrator not functioning properly")
                health_status["status"] = "degraded"
        else:
            health_status["components"]["reranker_orchestrator"] = "not_available"
            health_status["issues"].append("Reranker orchestrator not initialized")
            health_status["status"] = "degraded"
        
        # Check cascade reranker
        if self.cascade_reranker:
            health_status["components"]["cascade_reranker"] = "available"
        else:
            health_status["components"]["cascade_reranker"] = "not_available" if self.enable_cascade else "disabled"
        
        # Check cross-encoder reranker
        if self.cross_encoder_reranker:
            health_status["components"]["cross_encoder_reranker"] = "available"
        else:
            health_status["components"]["cross_encoder_reranker"] = "not_available" if self.enable_cross_encoder else "disabled"
        
        # Check signal processor
        if self.signal_processor:
            health_status["components"]["signal_processor"] = "available"
        else:
            health_status["components"]["signal_processor"] = "not_available" if self.enable_signal_processing else "disabled"
        
        # Check advanced reranker
        if self.advanced_reranker:
            health_status["components"]["advanced_reranker"] = "available"
        else:
            health_status["components"]["advanced_reranker"] = "not_available"
            health_status["issues"].append("Advanced reranker not available")
        
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
    
    def _fallback_reranking(self, 
                          query: str, 
                          documents: List[Dict[str, Any]], 
                          strategy: str, 
                          top_k: int) -> Dict[str, Any]:
        """
        Fallback reranking when orchestrator is not available
        """
        if strategy == "cascade" and self.cascade_reranker:
            try:
                results = self.cascade_reranker.rerank(query, documents, top_k)
                self.stats["cascade_usage"] += 1
                return {
                    'reranked_results': results,
                    'total_results': len(results),
                    'success': True,
                    'fallback_used': True
                }
            except Exception as e:
                logger.error(f"Cascade reranking failed: {str(e)}")
        
        elif strategy == "cross_encoder" and self.cross_encoder_reranker:
            try:
                results = self.cross_encoder_reranker.rerank(query, documents, top_k)
                self.stats["cross_encoder_usage"] += 1
                return {
                    'reranked_results': results,
                    'total_results': len(results),
                    'success': True,
                    'fallback_used': True
                }
            except Exception as e:
                logger.error(f"Cross-encoder reranking failed: {str(e)}")
        
        elif self.advanced_reranker:
            try:
                results = self.advanced_reranker.rerank(query, documents, top_k)
                return {
                    'reranked_results': results,
                    'total_results': len(results),
                    'success': True,
                    'fallback_used': True
                }
            except Exception as e:
                logger.error(f"Advanced reranking failed: {str(e)}")
        
        # Ultimate fallback - return original documents
        return {
            'reranked_results': documents[:top_k],
            'total_results': len(documents[:top_k]),
            'success': False,
            'error': 'No reranking methods available'
        }
    
    def _create_empty_result(self, message: str) -> Dict[str, Any]:
        """Create empty reranking result"""
        return {
            'reranked_results': [],
            'total_results': 0,
            'success': False,
            'error': message,
            'reranking_metadata': {
                'reranking_time': 0.0,
                'strategy_used': 'none',
                'signal_processing_applied': False,
                'original_count': 0,
                'final_count': 0,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _check_cache(self, query: str, documents: List[Dict[str, Any]], strategy: str) -> Optional[Dict[str, Any]]:
        """Check if reranking result is cached"""
        if not self.cache:
            return None
        
        # Create cache key based on query, document IDs, and strategy
        doc_ids = [doc.get('id', str(hash(doc.get('content', '')))) for doc in documents]
        cache_key = f"{query}_{strategy}_{hash(str(sorted(doc_ids)))}"
        
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            
            # Check TTL
            if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                cached_entry['result']['from_cache'] = True
                return cached_entry['result']
            else:
                del self.cache[cache_key]
        
        return None
    
    def _cache_result(self, query: str, documents: List[Dict[str, Any]], strategy: str, result: Dict[str, Any]):
        """Cache reranking result"""
        if not self.cache:
            return
        
        doc_ids = [doc.get('id', str(hash(doc.get('content', '')))) for doc in documents]
        cache_key = f"{query}_{strategy}_{hash(str(sorted(doc_ids)))}"
        
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Limit cache size
        if len(self.cache) > 500:
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]['timestamp']
            )[:100]
            
            for key in oldest_keys:
                del self.cache[key]
    
    def _calculate_quality_improvement(self, original_docs: List[Dict[str, Any]], reranked_docs: List[Dict[str, Any]]) -> float:
        """Calculate quality improvement from reranking"""
        if not original_docs or not reranked_docs:
            return 0.0
        
        # Simple quality metric based on average similarity scores
        original_avg = sum(doc.get('similarity_score', 0.5) for doc in original_docs[:len(reranked_docs)]) / len(reranked_docs)
        reranked_avg = sum(doc.get('similarity_score', 0.5) for doc in reranked_docs) / len(reranked_docs)
        
        return max(0.0, reranked_avg - original_avg)
    
    def _update_stats(self, success: bool, reranking_time: float, quality_improvement: float, strategy: str):
        """Update service statistics"""
        self.stats["total_rerankings"] += 1
        
        if success:
            self.stats["successful_rerankings"] += 1
        else:
            self.stats["failed_rerankings"] += 1
        
        # Update averages
        total_rerankings = self.stats["total_rerankings"]
        
        # Average reranking time
        total_time = (self.stats["avg_reranking_time"] * (total_rerankings - 1) + reranking_time)
        self.stats["avg_reranking_time"] = total_time / total_rerankings
        
        # Average quality improvement (only for successful rerankings)
        if success:
            successful_rerankings = self.stats["successful_rerankings"]
            total_improvement = (self.stats["avg_quality_improvement"] * (successful_rerankings - 1) + quality_improvement)
            self.stats["avg_quality_improvement"] = total_improvement / successful_rerankings
        
        # Strategy usage
        if strategy not in self.stats["strategy_usage"]:
            self.stats["strategy_usage"][strategy] = 0
        self.stats["strategy_usage"][strategy] += 1


# Convenience functions for easy usage
def rerank_query(query: str, documents: List[Dict[str, Any]], strategy: str = "cascade", top_k: int = 10) -> Dict[str, Any]:
    """
    Convenience function to rerank documents for a single query
    
    Args:
        query: Search query
        documents: List of documents to rerank
        strategy: Reranking strategy to use
        top_k: Number of top results to return
        
    Returns:
        Dictionary containing reranked documents
    """
    reranker = RerankerServices()
    return reranker.rerank_documents(query, documents, strategy, top_k)


def get_reranker_health() -> Dict[str, Any]:
    """
    Convenience function to check reranker health
    
    Returns:
        Dictionary containing health status
    """
    reranker = RerankerServices()
    return reranker.health_check()


# Example usage and testing
if __name__ == "__main__":
    # Initialize services
    print("🚀 Initializing RerankerServices...")
    reranker_services = RerankerServices(
        enable_cascade=True,
        enable_cross_encoder=True,
        enable_signal_processing=True,
        enable_caching=True,
        default_top_k=5
    )
    
    # Test health check
    print("\n🏥 Health Check:")
    health = reranker_services.health_check()
    print(f"Status: {health['status']}")
    print(f"Components: {health['components']}")
    
    # Test reranking
    print("\n🔄 Testing Reranking:")
    test_query = "What is artificial intelligence?"
    
    # Sample documents
    test_documents = [
        {
            'id': 'doc_1',
            'content': 'Artificial intelligence is a branch of computer science.',
            'similarity_score': 0.7
        },
        {
            'id': 'doc_2',
            'content': 'Machine learning is a subset of AI that enables computers to learn.',
            'similarity_score': 0.8
        },
        {
            'id': 'doc_3',
            'content': 'Deep learning uses neural networks with multiple layers.',
            'similarity_score': 0.6
        }
    ]
    
    result = reranker_services.rerank_documents(
        query=test_query,
        documents=test_documents,
        strategy="cascade",
        top_k=3
    )
    
    print(f"Query: {test_query}")
    print(f"Original documents: {len(test_documents)}")
    print(f"Reranked results: {result['total_results']}")
    print(f"Success: {result.get('success', False)}")
    print(f"Reranking time: {result['reranking_metadata']['reranking_time']:.3f}s")
    
    # Get statistics
    print("\n📊 Service Statistics:")
    stats = reranker_services.get_reranking_stats()
    print(f"Total rerankings: {stats['total_rerankings']}")
    print(f"Success rate: {stats['success_rate']:.3f}")
    print(f"Average reranking time: {stats['avg_reranking_time']:.3f}s")
    print(f"Average quality improvement: {stats['avg_quality_improvement']:.3f}")
    
    # Get optimization recommendations
    print("\n🎯 Performance Optimization:")
    optimization = reranker_services.optimize_performance()
    print(f"Optimization score: {optimization['optimization_score']:.1f}/100")
    print("Recommendations:")
    for rec in optimization['recommendations']:
        print(f"  • {rec}")
    
    print("\n✅ RerankerServices testing completed!")