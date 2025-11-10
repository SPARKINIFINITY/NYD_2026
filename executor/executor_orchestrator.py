"""
Executor Orchestrator - Complete System Integration

Orchestrates the entire retrieval system by integrating all components:
- Planner integration for intelligent execution plans
- Multiple retriever coordination (Dense, Sparse, Graph, MemoRAG)
- Fusion engine for result combination
- Performance monitoring and optimization
- Health monitoring and system status
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

try:
    # Try relative imports first (when used as module)
    from .retrieval_executor import RetrievalExecutor
    from .dense_retriever import DenseRetriever
    from .sparse_retriever import SparseRetriever
    from .graph_retriever import GraphRetriever
    from .memorag_retriever import MemoRAGRetriever
    from .fusion_engine import FusionEngine
except ImportError:
    # Fall back to absolute imports (when run directly)
    from retrieval_executor import RetrievalExecutor
    from dense_retriever import DenseRetriever
    from sparse_retriever import SparseRetriever
    from graph_retriever import GraphRetriever
    from memorag_retriever import MemoRAGRetriever
    from fusion_engine import FusionEngine

logger = logging.getLogger(__name__)

class ExecutorOrchestrator:
    """
    Complete system orchestrator that integrates all retrieval components
    with planner intelligence and performance optimization
    """
    
    def __init__(self, 
                 enable_planner: bool = True,
                 max_workers: int = 4,
                 timeout_seconds: float = 30.0,
                 enable_caching: bool = True,
                 enable_monitoring: bool = True):
        
        self.enable_planner = enable_planner
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.enable_caching = enable_caching
        self.enable_monitoring = enable_monitoring
        
        # System monitoring (initialize first)
        self.system_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "planner_usage": 0,
            "retriever_performance": {},
            "fusion_method_usage": {},
            "session_count": 0,
            "cache_performance": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0
            }
        }
        
        # Performance optimization
        self.performance_history = []
        self.optimization_recommendations = []
        
        # Health monitoring
        self.health_status = {
            "status": "initializing",
            "last_check": datetime.now(),
            "component_health": {},
            "issues": []
        }
        
        # Initialize core components
        self.retrieval_executor = RetrievalExecutor(
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
            enable_parallel=True
        )
        
        self.fusion_engine = FusionEngine(
            default_method="rrf",
            rrf_k=60,
            score_normalization="min_max"
        )
        
        # Initialize retrievers
        self.retrievers = {}
        self._initialize_retrievers()
        
        logger.info("ExecutorOrchestrator initialized successfully")
    
    def _initialize_retrievers(self):
        """Initialize all retriever components"""
        try:
            # Dense retriever
            self.retrievers['dense'] = DenseRetriever(
                model_name="all-MiniLM-L6-v2",  # Faster model for better performance
                use_quantized=False
            )
            
            # Sparse retrievers
            self.retrievers['tfidf'] = SparseRetriever(
                method="tfidf",
                max_features=10000,
                ngram_range=(1, 2)
            )
            
            self.retrievers['bm25'] = SparseRetriever(
                method="bm25",
                max_features=10000,
                ngram_range=(1, 2)
            )
            
            # Graph retriever
            self.retrievers['graph'] = GraphRetriever(
                max_hops=3,
                min_edge_weight=0.1
            )
            
            # MemoRAG retriever
            self.retrievers['memorag'] = MemoRAGRetriever(
                similarity_threshold=0.8,
                cache_ttl_hours=24,
                max_cache_size=1000
            )
            
            # Register retrievers with executor
            for name, retriever in self.retrievers.items():
                self.retrieval_executor.register_retriever(name, retriever)
                self.system_stats["retriever_performance"][name] = {
                    "queries": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "success_rate": 0.0
                }
            
            logger.info(f"Initialized {len(self.retrievers)} retrievers")
            
        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
            raise
    
    def setup_data_sources(self, 
                          documents: List[Dict[str, Any]] = None,
                          knowledge_graph: Dict[str, Any] = None,
                          index_paths: Dict[str, str] = None):
        """
        Setup data sources for all retrievers
        
        Args:
            documents: List of documents for indexing
            knowledge_graph: Knowledge graph data for graph retriever
            index_paths: Pre-built index paths for retrievers
        """
        try:
            logger.info("Setting up data sources...")
            
            if documents:
                # Build indexes for dense and sparse retrievers
                if 'dense' in self.retrievers:
                    self.retrievers['dense'].build_index(documents)
                
                if 'tfidf' in self.retrievers:
                    self.retrievers['tfidf'].build_index(documents)
                
                if 'bm25' in self.retrievers:
                    self.retrievers['bm25'].build_index(documents)
                
                logger.info(f"Built indexes for {len(documents)} documents")
            
            if knowledge_graph and 'graph' in self.retrievers:
                self.retrievers['graph'].build_graph(knowledge_graph, documents)
                logger.info("Built knowledge graph")
            
            if index_paths:
                # Load pre-built indexes
                for retriever_name, path in index_paths.items():
                    if retriever_name in self.retrievers:
                        if hasattr(self.retrievers[retriever_name], 'load_index'):
                            self.retrievers[retriever_name].load_index(path)
                            logger.info(f"Loaded index for {retriever_name}")
            
            # Update health status
            self.health_status["status"] = "ready"
            self._update_health_check()
            
            logger.info("Data sources setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup data sources: {e}")
            self.health_status["status"] = "error"
            raise
    
    def process_query(self, 
                     query: str,
                     intent: str = None,
                     entities: List[str] = None,
                     session_id: str = None,
                     planner_plan: Dict[str, Any] = None,
                     optimization_goal: str = "balanced") -> Dict[str, Any]:
        """
        Process query end-to-end with optional planner integration
        
        Args:
            query: Search query
            intent: Query intent (fact, explain, compare, etc.)
            entities: Extracted entities from query
            session_id: Session identifier for MemoRAG
            planner_plan: Pre-generated execution plan from planner
            optimization_goal: balanced, speed, accuracy
        
        Returns:
            Complete execution result with metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: '{query[:50]}...'")
            
            # Generate execution plan
            if planner_plan:
                execution_plan = planner_plan
                self.system_stats["planner_usage"] += 1
                logger.debug("Using provided planner plan")
            else:
                execution_plan = self._generate_execution_plan(
                    query, intent, entities, optimization_goal
                )
                logger.debug("Generated internal execution plan")
            
            # Execute retrieval plan
            retrieval_result = self.retrieval_executor.execute_plan(
                execution_plan, query, session_id
            )
            
            # Post-process results
            processed_result = self._post_process_results(
                retrieval_result, query, execution_plan
            )
            
            # Update monitoring
            execution_time = time.time() - start_time
            self._update_system_stats(processed_result, execution_time, execution_plan)
            
            # Add orchestrator metadata
            processed_result.update({
                'orchestrator_metadata': {
                    'total_execution_time': execution_time,
                    'planner_used': planner_plan is not None,
                    'optimization_goal': optimization_goal,
                    'session_id': session_id,
                    'intent': intent,
                    'entities': entities,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            logger.info(f"Query processed successfully in {execution_time:.3f}s")
            
            return processed_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query processing failed after {execution_time:.3f}s: {e}")
            
            self.system_stats["failed_queries"] += 1
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'query': query,
                'orchestrator_metadata': {
                    'total_execution_time': execution_time,
                    'error_occurred': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _generate_execution_plan(self, 
                               query: str, 
                               intent: str = None, 
                               entities: List[str] = None,
                               optimization_goal: str = "balanced") -> Dict[str, Any]:
        """Generate execution plan when planner is not available"""
        
        # Simple rule-based plan generation
        plan = {
            'strategy': 'hybrid',
            'retrievers': [],
            'multihop': 0,
            'rerank': True,
            'fusion_method': 'rrf',
            'confidence': 0.8,
            'complexity': 0.5
        }
        
        # Determine retrievers based on query characteristics
        query_lower = query.lower()
        
        # Always include dense retriever for semantic search
        plan['retrievers'].append('dense')
        plan['dense_k'] = 50
        
        # Add sparse retriever for keyword matching
        if any(word in query_lower for word in ['what', 'who', 'where', 'when', 'how']):
            plan['retrievers'].append('bm25')
            plan['bm25_k'] = 50
        else:
            plan['retrievers'].append('tfidf')
            plan['tfidf_k'] = 50
        
        # Add graph retriever if entities are present
        if entities and len(entities) > 0:
            plan['retrievers'].append('graph')
            plan['graph_k'] = 30
        
        # Add MemoRAG if session-based
        if intent in ['explain', 'compare'] or len(query.split()) > 8:
            plan['retrievers'].append('memorag')
            plan['memorag_k'] = 20
        
        # Adjust based on optimization goal
        if optimization_goal == "speed":
            # Reduce k values and retrievers for speed
            for retriever in plan['retrievers']:
                k_key = f'{retriever}_k'
                if k_key in plan:
                    plan[k_key] = min(plan[k_key], 30)
            plan['rerank'] = False
            
        elif optimization_goal == "accuracy":
            # Increase k values and enable multi-hop for accuracy
            for retriever in plan['retrievers']:
                k_key = f'{retriever}_k'
                if k_key in plan:
                    plan[k_key] = min(plan[k_key] * 2, 100)
            
            if entities and len(entities) > 1:
                plan['multihop'] = 1
        
        return plan
    
    def _post_process_results(self, 
                            retrieval_result: Dict[str, Any], 
                            query: str,
                            execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process retrieval results"""
        
        if not retrieval_result.get('success', False):
            return retrieval_result
        
        results = retrieval_result.get('results', [])
        
        # Apply additional filtering and ranking
        filtered_results = self._apply_quality_filters(results, query)
        
        # Add result explanations
        explained_results = self._add_result_explanations(filtered_results, query)
        
        # Update result with post-processed data
        retrieval_result['results'] = explained_results
        retrieval_result['post_processing'] = {
            'quality_filtered': len(results) - len(filtered_results),
            'explanations_added': len(explained_results),
            'final_count': len(explained_results)
        }
        
        return retrieval_result
    
    def _apply_quality_filters(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply quality filters to results"""
        
        filtered_results = []
        
        for result in results:
            # Basic quality checks
            score = result.get('similarity_score', 0)
            content = result.get('content', '')
            
            # Skip very low scoring results (lowered threshold for testing)
            if score < 0.01:
                continue
            
            # Skip very short content
            if len(content.strip()) < 20:
                continue
            
            # Skip duplicate content (simple check)
            is_duplicate = False
            for existing in filtered_results:
                if self._calculate_content_similarity(content, existing.get('content', '')) > 0.9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Simple content similarity calculation"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _add_result_explanations(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Add explanations to results"""
        
        explained_results = []
        
        for result in results:
            explained_result = result.copy()
            
            # Add retrieval explanation
            explanation = {
                'retriever_used': result.get('retriever_type', 'unknown'),
                'score_explanation': self._explain_score(result),
                'relevance_factors': self._identify_relevance_factors(result, query)
            }
            
            explained_result['explanation'] = explanation
            explained_results.append(explained_result)
        
        return explained_results
    
    def _explain_score(self, result: Dict[str, Any]) -> str:
        """Generate score explanation"""
        score = result.get('similarity_score', 0)
        retriever_type = result.get('retriever_type', 'unknown')
        
        if score > 0.8:
            return f"High relevance ({score:.3f}) from {retriever_type} retriever"
        elif score > 0.5:
            return f"Moderate relevance ({score:.3f}) from {retriever_type} retriever"
        else:
            return f"Low relevance ({score:.3f}) from {retriever_type} retriever"
    
    def _identify_relevance_factors(self, result: Dict[str, Any], query: str) -> List[str]:
        """Identify factors contributing to relevance"""
        factors = []
        
        # Check for entity matches
        if 'matched_entities' in result:
            entities = result['matched_entities']
            if entities:
                factors.append(f"Entity matches: {', '.join(entities[:3])}")
        
        # Check for keyword matches
        content = result.get('content', '').lower()
        query_words = query.lower().split()
        
        matches = [word for word in query_words if word in content]
        if matches:
            factors.append(f"Keyword matches: {', '.join(matches[:3])}")
        
        # Check retriever-specific factors
        retriever_type = result.get('retriever_type', '')
        
        if retriever_type == 'dense':
            factors.append("Semantic similarity match")
        elif retriever_type == 'sparse':
            factors.append("Keyword frequency match")
        elif retriever_type == 'graph':
            factors.append("Knowledge graph connection")
        elif retriever_type == 'memorag':
            if result.get('memo_info', {}).get('cache_hit'):
                factors.append("Previous context match")
        
        return factors
    
    def _update_system_stats(self, result: Dict[str, Any], execution_time: float, plan: Dict[str, Any]):
        """Update system statistics"""
        
        self.system_stats["total_queries"] += 1
        
        if result.get('success', False):
            self.system_stats["successful_queries"] += 1
        else:
            self.system_stats["failed_queries"] += 1
        
        self.system_stats["total_execution_time"] += execution_time
        self.system_stats["average_execution_time"] = (
            self.system_stats["total_execution_time"] / self.system_stats["total_queries"]
        )
        
        # Update retriever performance
        retrievers_used = result.get('execution_metadata', {}).get('retrievers_used', [])
        for retriever_name in retrievers_used:
            if retriever_name in self.system_stats["retriever_performance"]:
                perf = self.system_stats["retriever_performance"][retriever_name]
                perf["queries"] += 1
                perf["total_time"] += execution_time
                perf["avg_time"] = perf["total_time"] / perf["queries"]
        
        # Update fusion method usage
        fusion_method = plan.get('fusion_method', 'unknown')
        if fusion_method not in self.system_stats["fusion_method_usage"]:
            self.system_stats["fusion_method_usage"][fusion_method] = 0
        self.system_stats["fusion_method_usage"][fusion_method] += 1
        
        # Store performance history for optimization
        self.performance_history.append({
            'timestamp': datetime.now(),
            'execution_time': execution_time,
            'success': result.get('success', False),
            'result_count': len(result.get('results', [])),
            'retrievers_used': retrievers_used,
            'fusion_method': fusion_method
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def batch_process(self, 
                     queries: List[str], 
                     session_id: str = None,
                     optimization_goal: str = "balanced") -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        
        logger.info(f"Processing batch of {len(queries)} queries")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(
                    self.process_query, 
                    query, 
                    session_id=session_id,
                    optimization_goal=optimization_goal
                ): query 
                for query in queries
            }
            
            # Collect results
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    result['batch_query'] = query
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch query failed: {query} - {e}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'batch_query': query
                    })
        
        logger.info(f"Batch processing completed: {len(results)} results")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        self._update_health_check()
        
        status = {
            'health': self.health_status,
            'statistics': self.system_stats,
            'performance': {
                'recent_avg_time': self._calculate_recent_avg_time(),
                'success_rate': (
                    self.system_stats["successful_queries"] / 
                    max(self.system_stats["total_queries"], 1)
                ),
                'queries_per_minute': self._calculate_queries_per_minute()
            },
            'retrievers': {
                name: retriever.get_stats() if hasattr(retriever, 'get_stats') else {}
                for name, retriever in self.retrievers.items()
            },
            'optimization': {
                'recommendations': self.optimization_recommendations,
                'performance_trend': self._analyze_performance_trend()
            }
        }
        
        return status
    
    def _update_health_check(self):
        """Update health check status"""
        
        self.health_status["last_check"] = datetime.now()
        self.health_status["issues"] = []
        
        # Check each retriever
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'health_check'):
                health = retriever.health_check()
                self.health_status["component_health"][name] = health
                
                if health.get('status') != 'healthy':
                    self.health_status["issues"].extend([
                        f"{name}: {issue}" for issue in health.get('issues', [])
                    ])
        
        # Check executor health
        executor_health = self.retrieval_executor.health_check()
        self.health_status["component_health"]["executor"] = executor_health
        
        if executor_health.get('status') != 'healthy':
            self.health_status["issues"].extend([
                f"executor: {issue}" for issue in executor_health.get('issues', [])
            ])
        
        # Overall health assessment
        if not self.health_status["issues"]:
            self.health_status["status"] = "healthy"
        elif len(self.health_status["issues"]) < 3:
            self.health_status["status"] = "warning"
        else:
            self.health_status["status"] = "unhealthy"
    
    def _calculate_recent_avg_time(self) -> float:
        """Calculate average execution time for recent queries"""
        if not self.performance_history:
            return 0.0
        
        recent_history = self.performance_history[-50:]  # Last 50 queries
        times = [h['execution_time'] for h in recent_history]
        
        return sum(times) / len(times) if times else 0.0
    
    def _calculate_queries_per_minute(self) -> float:
        """Calculate queries per minute rate"""
        if len(self.performance_history) < 2:
            return 0.0
        
        recent_history = self.performance_history[-100:]  # Last 100 queries
        
        if len(recent_history) < 2:
            return 0.0
        
        time_span = (recent_history[-1]['timestamp'] - recent_history[0]['timestamp']).total_seconds()
        
        if time_span > 0:
            return (len(recent_history) / time_span) * 60
        
        return 0.0
    
    def _analyze_performance_trend(self) -> str:
        """Analyze performance trend"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent_times = [h['execution_time'] for h in self.performance_history[-20:]]
        older_times = [h['execution_time'] for h in self.performance_history[-40:-20]]
        
        if not older_times:
            return "insufficient_data"
        
        recent_avg = sum(recent_times) / len(recent_times)
        older_avg = sum(older_times) / len(older_times)
        
        if recent_avg < older_avg * 0.9:
            return "improving"
        elif recent_avg > older_avg * 1.1:
            return "degrading"
        else:
            return "stable"
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        optimization_score = 100
        
        # Analyze execution times
        if self.system_stats["average_execution_time"] > 5.0:
            recommendations.append("Consider reducing k values or number of retrievers")
            optimization_score -= 20
        
        # Analyze success rate
        success_rate = (
            self.system_stats["successful_queries"] / 
            max(self.system_stats["total_queries"], 1)
        )
        
        if success_rate < 0.9:
            recommendations.append("Investigate query failures and improve error handling")
            optimization_score -= 15
        
        # Analyze retriever performance
        for name, perf in self.system_stats["retriever_performance"].items():
            if perf["avg_time"] > 2.0:
                recommendations.append(f"Optimize {name} retriever performance")
                optimization_score -= 10
        
        # Analyze cache performance
        if 'memorag' in self.retrievers:
            memorag_stats = self.retrievers['memorag'].get_stats()
            cache_hit_rate = memorag_stats.get('cache_hit_rate', 0)
            
            if cache_hit_rate < 0.3:
                recommendations.append("Improve MemoRAG cache hit rate with better similarity thresholds")
                optimization_score -= 10
        
        # Performance trend analysis
        trend = self._analyze_performance_trend()
        if trend == "degrading":
            recommendations.append("Performance is degrading - investigate system resources")
            optimization_score -= 15
        
        if not recommendations:
            recommendations.append("System is performing optimally")
        
        self.optimization_recommendations = recommendations
        
        return {
            'optimization_score': max(optimization_score, 0),
            'recommendations': recommendations,
            'performance_trend': trend,
            'key_metrics': {
                'avg_execution_time': self.system_stats["average_execution_time"],
                'success_rate': success_rate,
                'queries_per_minute': self._calculate_queries_per_minute()
            }
        }
    
    def clear_caches(self):
        """Clear all caches in the system"""
        
        cleared_count = 0
        
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'clear_cache'):
                retriever.clear_cache()
                cleared_count += 1
        
        # Clear fusion engine cache if it has one
        if hasattr(self.fusion_engine, 'clear_cache'):
            self.fusion_engine.clear_cache()
        
        logger.info(f"Cleared caches for {cleared_count} components")
    
    def export_system_config(self) -> Dict[str, Any]:
        """Export system configuration for backup/restore"""
        
        config = {
            'orchestrator_config': {
                'enable_planner': self.enable_planner,
                'max_workers': self.max_workers,
                'timeout_seconds': self.timeout_seconds,
                'enable_caching': self.enable_caching,
                'enable_monitoring': self.enable_monitoring
            },
            'retriever_configs': {},
            'fusion_config': self.fusion_engine.get_stats() if hasattr(self.fusion_engine, 'get_stats') else {},
            'system_stats': self.system_stats,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Export retriever configurations
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'get_stats'):
                config['retriever_configs'][name] = retriever.get_stats()
        
        return config
    
    def shutdown(self):
        """Graceful shutdown of the orchestrator"""
        
        logger.info("Shutting down ExecutorOrchestrator...")
        
        # Clear caches
        self.clear_caches()
        
        # Export final stats
        final_config = self.export_system_config()
        
        logger.info("ExecutorOrchestrator shutdown completed")
        
        return final_config


# Convenience functions for easy integration

def create_orchestrator(documents: List[Dict[str, Any]] = None,
                       knowledge_graph: Dict[str, Any] = None,
                       enable_planner: bool = True) -> ExecutorOrchestrator:
    """
    Convenience function to create and setup orchestrator
    
    Args:
        documents: Documents to index
        knowledge_graph: Knowledge graph data
        enable_planner: Whether to enable planner integration
    
    Returns:
        Configured ExecutorOrchestrator
    """
    
    orchestrator = ExecutorOrchestrator(enable_planner=enable_planner)
    
    if documents or knowledge_graph:
        orchestrator.setup_data_sources(documents, knowledge_graph)
    
    return orchestrator


def quick_query(query: str, 
               documents: List[Dict[str, Any]] = None,
               optimization_goal: str = "balanced") -> Dict[str, Any]:
    """
    Convenience function for quick query processing
    
    Args:
        query: Search query
        documents: Optional documents to search
        optimization_goal: Speed, accuracy, or balanced
    
    Returns:
        Query results
    """
    
    orchestrator = create_orchestrator(documents)
    
    return orchestrator.process_query(
        query=query,
        optimization_goal=optimization_goal
    )