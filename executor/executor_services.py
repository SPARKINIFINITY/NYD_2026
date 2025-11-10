"""
Executor Services - High-Level Service Interface

Provides a high-level service interface for the executor system,
integrating with planner and providing convenient methods for
different types of retrieval operations.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

try:
    from .executor_orchestrator import ExecutorOrchestrator
except ImportError:
    from executor_orchestrator import ExecutorOrchestrator

logger = logging.getLogger(__name__)

class ExecutorServices:
    """
    High-level service interface for the executor system
    """
    
    def __init__(self, 
                 enable_fusion: bool = True,
                 enable_caching: bool = True,
                 max_results: int = 10,
                 max_workers: int = 4,
                 documents: List[Dict[str, Any]] = None,
                 knowledge_graph: Dict[str, Any] = None):
        
        self.enable_fusion = enable_fusion
        self.enable_caching = enable_caching
        self.max_results = max_results
        
        # Initialize orchestrator
        self.orchestrator = ExecutorOrchestrator(
            enable_planner=True,
            max_workers=max_workers,
            enable_caching=enable_caching,
            enable_monitoring=True
        )
        
        # Setup data sources if provided
        if documents or knowledge_graph:
            self.orchestrator.setup_data_sources(documents, knowledge_graph)
        
        # Flag to track if data sources are set up
        self.data_sources_ready = bool(documents or knowledge_graph)
        
        # Service statistics
        self.service_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "avg_results_count": 0.0,
            "success_rate": 0.0
        }
        
        logger.info(f"ExecutorServices initialized (data sources ready: {self.data_sources_ready})")
    
    def setup_data_sources(self, 
                          documents: List[Dict[str, Any]] = None,
                          knowledge_graph: Dict[str, Any] = None):
        """
        Setup data sources for the orchestrator
        
        Args:
            documents: List of documents for indexing
            knowledge_graph: Knowledge graph data
        """
        try:
            self.orchestrator.setup_data_sources(documents, knowledge_graph)
            self.data_sources_ready = True
            logger.info("Data sources setup completed for ExecutorServices")
        except Exception as e:
            logger.error(f"Failed to setup data sources: {e}")
            raise
    
    def execute_retrieval(self, 
                         query: str,
                         retrieval_strategy: str = "hybrid",
                         max_results: int = None,
                         session_id: str = None) -> Dict[str, Any]:
        """
        Execute retrieval with specified strategy
        
        Args:
            query: Search query
            retrieval_strategy: hybrid, semantic, keyword, graph
            max_results: Maximum results to return
            session_id: Session identifier
        
        Returns:
            Retrieval results with metadata
        """
        start_time = time.time()
        max_results = max_results or self.max_results
        
        # Warn if data sources are not ready
        if not self.data_sources_ready:
            logger.warning("Data sources not set up - retrievers may not return results. Use setup_data_sources() first.")
        
        try:
            # Generate execution plan based on strategy
            execution_plan = self._strategy_to_plan(retrieval_strategy, max_results)
            
            # Process query
            result = self.orchestrator.process_query(
                query=query,
                session_id=session_id,
                planner_plan=execution_plan,
                optimization_goal="balanced"
            )
            
            # Format response
            formatted_result = self._format_service_response(result, query, retrieval_strategy)
            
            # Update stats
            execution_time = time.time() - start_time
            self._update_service_stats(formatted_result, execution_time)
            
            return formatted_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Retrieval execution failed: {e}")
            
            error_result = {
                'success': False,
                'error': str(e),
                'query': query,
                'strategy': retrieval_strategy,
                'execution_time': execution_time,
                'total_results': 0,
                'documents': [],
                'execution_metadata': {
                    'execution_time': execution_time,
                    'error_occurred': True
                }
            }
            
            self._update_service_stats(error_result, execution_time)
            return error_result 
   
    def execute_with_planner_output(self, 
                                   query: str,
                                   planner_plan: Dict[str, Any],
                                   session_id: str = None) -> Dict[str, Any]:
        """
        Execute retrieval using planner output
        
        Args:
            query: Search query
            planner_plan: Execution plan from planner
            session_id: Session identifier
        
        Returns:
            Retrieval results
        """
        start_time = time.time()
        
        try:
            # Process query with planner plan
            result = self.orchestrator.process_query(
                query=query,
                session_id=session_id,
                planner_plan=planner_plan,
                optimization_goal="balanced"
            )
            
            # Format response
            formatted_result = self._format_service_response(
                result, query, planner_plan.get('strategy', 'planner')
            )
            
            # Update stats
            execution_time = time.time() - start_time
            self._update_service_stats(formatted_result, execution_time)
            
            return formatted_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Planner execution failed: {e}")
            
            error_result = {
                'success': False,
                'error': str(e),
                'query': query,
                'planner_plan': planner_plan,
                'execution_time': execution_time,
                'total_results': 0,
                'documents': [],
                'execution_metadata': {
                    'execution_time': execution_time,
                    'error_occurred': True
                }
            }
            
            self._update_service_stats(error_result, execution_time)
            return error_result
    
    def execute_batch(self, 
                     queries: List[str],
                     retrieval_strategy: str = "hybrid",
                     max_results: int = None,
                     session_id: str = None) -> List[Dict[str, Any]]:
        """
        Execute batch retrieval
        
        Args:
            queries: List of search queries
            retrieval_strategy: Retrieval strategy to use
            max_results: Maximum results per query
            session_id: Session identifier
        
        Returns:
            List of retrieval results
        """
        logger.info(f"Executing batch retrieval for {len(queries)} queries")
        
        results = []
        
        for query in queries:
            result = self.execute_retrieval(
                query=query,
                retrieval_strategy=retrieval_strategy,
                max_results=max_results,
                session_id=session_id
            )
            results.append(result)
        
        return results
    
    def _strategy_to_plan(self, strategy: str, max_results: int) -> Dict[str, Any]:
        """Convert strategy to execution plan"""
        
        base_k = min(max_results * 2, 50)  # Get more results for fusion
        
        if strategy == "semantic":
            return {
                'strategy': 'semantic',
                'retrievers': ['dense'],
                'dense_k': base_k,
                'multihop': 0,
                'rerank': True,
                'fusion_method': 'rrf',
                'confidence': 0.8
            }
        
        elif strategy == "keyword":
            return {
                'strategy': 'keyword',
                'retrievers': ['bm25', 'tfidf'],
                'bm25_k': base_k // 2,
                'tfidf_k': base_k // 2,
                'multihop': 0,
                'rerank': True,
                'fusion_method': 'weighted_sum',
                'confidence': 0.7
            }
        
        elif strategy == "graph":
            return {
                'strategy': 'graph',
                'retrievers': ['graph', 'dense'],
                'graph_k': base_k // 2,
                'dense_k': base_k // 2,
                'multihop': 1,
                'rerank': True,
                'fusion_method': 'rrf',
                'confidence': 0.75
            }
        
        else:  # hybrid (default)
            return {
                'strategy': 'hybrid',
                'retrievers': ['dense', 'bm25', 'memorag'],
                'dense_k': base_k // 3,
                'bm25_k': base_k // 3,
                'memorag_k': base_k // 3,
                'multihop': 0,
                'rerank': True,
                'fusion_method': 'rrf',
                'confidence': 0.8
            }
    
    def _format_service_response(self, 
                               result: Dict[str, Any], 
                               query: str,
                               strategy: str) -> Dict[str, Any]:
        """Format orchestrator response for service interface"""
        
        if not result.get('success', False):
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'query': query,
                'strategy': strategy,
                'total_results': 0,
                'documents': [],
                'execution_metadata': result.get('execution_metadata', {}),
                'fusion_applied': False
            }
        
        # Extract documents
        documents = []
        results = result.get('results', [])
        
        for i, res in enumerate(results[:self.max_results]):
            doc = {
                'id': res.get('id', f'doc_{i}'),
                'title': res.get('title', res.get('content', '')[:100] + '...'),
                'content': res.get('content', ''),
                'score': res.get('similarity_score', 0.0),
                'retriever': res.get('retriever_type', 'unknown'),
                'rank': i + 1,
                'metadata': {
                    'entities': res.get('entities', []),
                    'topic': res.get('topic', ''),
                    'explanation': res.get('explanation', {})
                }
            }
            documents.append(doc)
        
        # Check if fusion was applied
        fusion_applied = len(result.get('retriever_results', {})) > 1
        
        return {
            'success': True,
            'query': query,
            'strategy': strategy,
            'total_results': len(documents),
            'documents': documents,
            'execution_metadata': result.get('execution_metadata', {}),
            'orchestrator_metadata': result.get('orchestrator_metadata', {}),
            'fusion_applied': fusion_applied,
            'performance_metrics': result.get('performance_metrics', {})
        }
    
    def _update_service_stats(self, result: Dict[str, Any], execution_time: float):
        """Update service statistics"""
        
        self.service_stats["total_queries"] += 1
        
        if result.get('success', False):
            self.service_stats["successful_queries"] += 1
        else:
            self.service_stats["failed_queries"] += 1
        
        self.service_stats["total_execution_time"] += execution_time
        self.service_stats["avg_execution_time"] = (
            self.service_stats["total_execution_time"] / 
            self.service_stats["total_queries"]
        )
        
        # Update average results count
        result_count = result.get('total_results', 0)
        current_avg = self.service_stats["avg_results_count"]
        total_queries = self.service_stats["total_queries"]
        
        self.service_stats["avg_results_count"] = (
            (current_avg * (total_queries - 1) + result_count) / total_queries
        )
        
        # Update success rate
        self.service_stats["success_rate"] = (
            self.service_stats["successful_queries"] / 
            self.service_stats["total_queries"]
        )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return self.service_stats.copy()
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Get performance optimization recommendations"""
        return self.orchestrator.optimize_performance()
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        
        orchestrator_status = self.orchestrator.get_system_status()
        
        return {
            'status': orchestrator_status['health']['status'],
            'components': {
                'orchestrator': orchestrator_status['health']['status'],
                'retrievers': len(orchestrator_status['retrievers']),
                'fusion_engine': 'active' if self.enable_fusion else 'disabled',
                'caching': 'active' if self.enable_caching else 'disabled'
            },
            'service_stats': self.service_stats,
            'system_status': orchestrator_status
        }
    
    def clear_caches(self):
        """Clear all system caches"""
        self.orchestrator.clear_caches()
    
    def shutdown(self):
        """Shutdown services"""
        return self.orchestrator.shutdown()


# Convenience function for direct planner integration
def execute_with_planner(query: str, planner_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to execute query with planner output
    
    Args:
        query: Search query
        planner_plan: Execution plan from planner
    
    Returns:
        Retrieval results
    """
    
    services = ExecutorServices()
    
    return services.execute_with_planner_output(query, planner_plan)