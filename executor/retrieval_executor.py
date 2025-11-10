"""
Retrieval Executor - Parallel Retrieval Orchestrator

Main executor that runs multiple retrievers in parallel and coordinates
the retrieval process based on execution plans from the planner.
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from datetime import datetime

try:
    # Try relative imports first (when used as module)
    from .dense_retriever import DenseRetriever
    from .sparse_retriever import SparseRetriever
    from .graph_retriever import GraphRetriever
    from .memorag_retriever import MemoRAGRetriever
    from .fusion_engine import FusionEngine
except ImportError:
    # Fall back to absolute imports (when run directly)
    from dense_retriever import DenseRetriever
    from sparse_retriever import SparseRetriever
    from graph_retriever import GraphRetriever
    from memorag_retriever import MemoRAGRetriever
    from fusion_engine import FusionEngine

logger = logging.getLogger(__name__)

class RetrievalExecutor:
    """Orchestrates parallel retrieval execution"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 timeout_seconds: float = 30.0,
                 enable_parallel: bool = True):
        
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.enable_parallel = enable_parallel
        
        # Initialize retrievers
        self.retrievers = {}
        self.fusion_engine = FusionEngine()
        
        # Execution statistics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0,
            "retriever_usage": {},
            "parallel_executions": 0,
            "sequential_executions": 0
        }
        
        logger.info(f"Initialized retrieval executor with max_workers: {max_workers}")
    
    def register_retriever(self, name: str, retriever: Any) -> bool:
        """Register a retriever with the executor"""
        try:
            # Validate retriever has required methods
            if not hasattr(retriever, 'retrieve'):
                logger.error(f"Retriever {name} missing 'retrieve' method")
                return False
            
            self.retrievers[name] = retriever
            self.execution_stats["retriever_usage"][name] = 0
            
            logger.info(f"Registered retriever: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register retriever {name}: {e}")
            return False
    
    def execute_plan(self, 
                    execution_plan: Dict[str, Any], 
                    query: str,
                    session_id: str = None) -> Dict[str, Any]:
        """
        Execute retrieval plan
        
        Args:
            execution_plan: Plan from planner with retriever configs
            query: Search query
            session_id: Optional session ID for MemoRAG
        
        Returns:
            Execution results with fused results and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Executing retrieval plan for query: '{query[:50]}...'")
            
            # Extract plan parameters
            retrievers_to_use = execution_plan.get('retrievers', [])
            multihop_steps = execution_plan.get('multihop', 0)
            rerank = execution_plan.get('rerank', False)
            fusion_method = execution_plan.get('fusion_method', 'rrf')
            
            if not retrievers_to_use:
                logger.warning("No retrievers specified in execution plan")
                return self._create_empty_result("No retrievers specified")
            
            # Execute retrievers
            retriever_results = {}
            
            if multihop_steps > 0:
                # Multi-hop execution
                retriever_results = self._execute_multihop(
                    query, retrievers_to_use, execution_plan, multihop_steps, session_id
                )
            else:
                # Single-step execution
                retriever_results = self._execute_single_step(
                    query, retrievers_to_use, execution_plan, session_id
                )
            
            # Fuse results
            fused_results = self._fuse_results(
                retriever_results, execution_plan, fusion_method
            )
            
            # Apply reranking if specified
            if rerank and fused_results:
                fused_results = self._apply_reranking(fused_results, query)
            
            # Create execution result
            execution_time = time.time() - start_time
            
            result = {
                'success': True,
                'results': fused_results,
                'execution_metadata': {
                    'query': query,
                    'session_id': session_id,
                    'execution_time': execution_time,
                    'retrievers_used': list(retriever_results.keys()),
                    'total_results_before_fusion': sum(len(results) for results in retriever_results.values()),
                    'final_result_count': len(fused_results),
                    'multihop_steps': multihop_steps,
                    'reranking_applied': rerank,
                    'fusion_method': fusion_method,
                    'execution_plan': execution_plan,
                    'timestamp': datetime.now().isoformat()
                },
                'retriever_results': retriever_results,
                'performance_metrics': self._calculate_performance_metrics(
                    retriever_results, execution_time
                )
            }
            
            # Update statistics
            self._update_execution_stats(result, execution_time)
            
            logger.info(f"Execution completed in {execution_time:.3f}s, returned {len(fused_results)} results")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Execution failed after {execution_time:.3f}s: {e}")
            
            self.execution_stats["failed_executions"] += 1
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'query': query,
                'execution_plan': execution_plan
            }
    
    def _execute_single_step(self, 
                           query: str, 
                           retrievers_to_use: List[str], 
                           execution_plan: Dict[str, Any],
                           session_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Execute retrievers in single step (parallel or sequential)"""
        
        retriever_tasks = []
        
        # Prepare retriever tasks
        for retriever_name in retrievers_to_use:
            if retriever_name not in self.retrievers:
                logger.warning(f"Retriever {retriever_name} not registered, skipping")
                continue
            
            k_value = execution_plan.get(f'{retriever_name}_k', 10)
            
            task = {
                'name': retriever_name,
                'retriever': self.retrievers[retriever_name],
                'query': query,
                'k': k_value,
                'session_id': session_id
            }
            retriever_tasks.append(task)
        
        if not retriever_tasks:
            logger.warning("No valid retrievers found for execution")
            return {}
        
        # Execute tasks
        if self.enable_parallel and len(retriever_tasks) > 1:
            return self._execute_parallel(retriever_tasks)
        else:
            return self._execute_sequential(retriever_tasks)
    
    def _execute_parallel(self, retriever_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Execute retrievers in parallel"""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_name = {}
            
            for task in retriever_tasks:
                future = executor.submit(self._execute_single_retriever, task)
                future_to_name[future] = task['name']
            
            # Collect results with timeout
            try:
                for future in concurrent.futures.as_completed(future_to_name, timeout=self.timeout_seconds):
                    retriever_name = future_to_name[future]
                    try:
                        result = future.result()
                        results[retriever_name] = result
                    except Exception as e:
                        logger.error(f"Retriever {retriever_name} failed: {e}")
                        results[retriever_name] = []
                
                self.execution_stats["parallel_executions"] += 1
                
            except concurrent.futures.TimeoutError:
                logger.error(f"Parallel execution timed out after {self.timeout_seconds}s")
                # Cancel remaining futures
                for future in future_to_name:
                    future.cancel()
        
        return results
    
    def _execute_sequential(self, retriever_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Execute retrievers sequentially"""
        results = {}
        
        for task in retriever_tasks:
            try:
                result = self._execute_single_retriever(task)
                results[task['name']] = result
            except Exception as e:
                logger.error(f"Retriever {task['name']} failed: {e}")
                results[task['name']] = []
        
        self.execution_stats["sequential_executions"] += 1
        
        return results
    
    def _execute_single_retriever(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a single retriever task"""
        retriever = task['retriever']
        retriever_name = task['name']
        
        start_time = time.time()
        
        try:
            # Handle different retriever types
            if retriever_name == 'memorag' and isinstance(retriever, MemoRAGRetriever):
                # MemoRAG needs special handling
                base_retrievers = [r for name, r in self.retrievers.items() 
                                 if name != 'memorag' and hasattr(r, 'retrieve')]
                results, memo_info = retriever.retrieve(
                    task['query'], 
                    task['session_id'] or 'default',
                    base_retrievers,
                    task['k']
                )
                
                # Add MemoRAG metadata to results
                for result in results:
                    result['memo_info'] = memo_info
                
            elif retriever_name == 'graph' and isinstance(retriever, GraphRetriever):
                # Graph retriever needs entities
                # For now, extract simple entities from query (could be enhanced)
                query_entities = self._extract_simple_entities(task['query'])
                results = retriever.retrieve(query_entities, task['k'])
                
            else:
                # Standard retriever
                results = retriever.retrieve(task['query'], task['k'])
            
            execution_time = time.time() - start_time
            
            # Add execution metadata to results
            for result in results:
                result['execution_time'] = execution_time
                result['retriever_name'] = retriever_name
            
            logger.debug(f"Retriever {retriever_name} completed in {execution_time:.3f}s, returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing retriever {retriever_name}: {e}")
            return []
    
    def _execute_multihop(self, 
                         query: str, 
                         retrievers_to_use: List[str], 
                         execution_plan: Dict[str, Any],
                         multihop_steps: int,
                         session_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Execute multi-hop retrieval"""
        
        logger.info(f"Executing {multihop_steps}-hop retrieval")
        
        all_results = {}
        current_query = query
        
        for hop in range(multihop_steps):
            logger.debug(f"Executing hop {hop + 1}/{multihop_steps}")
            
            # Execute current hop
            hop_results = self._execute_single_step(
                current_query, retrievers_to_use, execution_plan, session_id
            )
            
            # Merge results
            for retriever_name, results in hop_results.items():
                if retriever_name not in all_results:
                    all_results[retriever_name] = []
                
                # Add hop information to results
                for result in results:
                    result['hop_number'] = hop + 1
                    result['multihop_query'] = current_query
                
                all_results[retriever_name].extend(results)
            
            # Generate next query from current results (simple approach)
            if hop < multihop_steps - 1:
                current_query = self._generate_next_hop_query(hop_results, query)
        
        # Remove duplicates and limit results
        for retriever_name in all_results:
            all_results[retriever_name] = self._deduplicate_results(
                all_results[retriever_name]
            )[:execution_plan.get(f'{retriever_name}_k', 10)]
        
        return all_results
    
    def _generate_next_hop_query(self, current_results: Dict[str, List[Dict[str, Any]]], 
                                original_query: str) -> str:
        """Generate query for next hop based on current results"""
        
        # Simple approach: extract entities from top results
        entities = set()
        
        for retriever_results in current_results.values():
            for result in retriever_results[:3]:  # Top 3 results
                # Extract entities if available
                if 'matched_entities' in result:
                    entities.update(result['matched_entities'])
                
                # Extract key terms from content
                content = result.get('content', '')
                if content:
                    # Simple keyword extraction
                    words = content.lower().split()
                    # Add capitalized words as potential entities
                    for word in words:
                        if word.istitle() and len(word) > 3:
                            entities.add(word)
        
        # Create expanded query
        if entities:
            entity_list = list(entities)[:5]  # Limit to 5 entities
            expanded_query = f"{original_query} {' '.join(entity_list)}"
        else:
            expanded_query = original_query
        
        logger.debug(f"Generated next hop query: '{expanded_query}'")
        return expanded_query
    
    def _fuse_results(self, 
                     retriever_results: Dict[str, List[Dict[str, Any]]], 
                     execution_plan: Dict[str, Any],
                     fusion_method: str) -> List[Dict[str, Any]]:
        """Fuse results from multiple retrievers"""
        
        if not retriever_results:
            return []
        
        # Extract weights from execution plan
        weights = {}
        for retriever_name in retriever_results.keys():
            # Use k value as weight (higher k = higher weight)
            k_value = execution_plan.get(f'{retriever_name}_k', 10)
            weights[retriever_name] = k_value / 100.0  # Normalize
        
        # Get total k for final results
        total_k = sum(execution_plan.get(f'{name}_k', 10) for name in retriever_results.keys())
        final_k = min(total_k // len(retriever_results), 50)  # Reasonable limit
        
        # Fuse results
        fused_results = self.fusion_engine.fuse_results(
            retriever_results, 
            weights, 
            fusion_method, 
            final_k
        )
        
        return fused_results
    
    def _apply_reranking(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply reranking to results (simple implementation)"""
        
        # Simple reranking based on query term overlap
        query_terms = set(query.lower().split())
        
        for result in results:
            content = result.get('content', '').lower()
            content_terms = set(content.split())
            
            # Calculate term overlap
            overlap = len(query_terms.intersection(content_terms))
            overlap_score = overlap / len(query_terms) if query_terms else 0
            
            # Boost score based on overlap
            original_score = result.get('similarity_score', 0)
            reranked_score = original_score + (overlap_score * 0.1)
            
            result['similarity_score'] = reranked_score
            result['rerank_boost'] = overlap_score * 0.1
            result['reranked'] = True
        
        # Re-sort by new scores
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return results
    
    def _extract_simple_entities(self, query: str) -> List[str]:
        """Extract simple entities from query (improved version)"""
        words = query.split()
        entities = []
        
        # Common stop words to exclude
        stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'between', 'difference', 'compare', 'versus'}
        
        for i, word in enumerate(words):
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            clean_lower = clean_word.lower()
            
            # Skip stop words
            if clean_lower in stop_words:
                continue
            
            # Consider capitalized words as entities
            if clean_word and clean_word[0].isupper():
                entities.append(clean_word)
            # Also consider longer words (potential concepts)
            elif len(clean_word) > 4:
                entities.append(clean_word)
            
            # Check for compound terms (e.g., "karma yoga")
            if i < len(words) - 1:
                next_word = ''.join(c for c in words[i+1] if c.isalnum())
                if next_word and len(next_word) > 3:
                    compound = f"{clean_word} {next_word}"
                    if clean_lower not in stop_words:
                        entities.append(compound)
        
        return entities
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results"""
        seen = set()
        unique_results = []
        
        for result in results:
            # Create identifier
            identifier = result.get('id') or result.get('content', '')[:100]
            
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)
        
        return unique_results
    
    def _calculate_performance_metrics(self, 
                                     retriever_results: Dict[str, List[Dict[str, Any]]], 
                                     total_time: float) -> Dict[str, Any]:
        """Calculate performance metrics for execution"""
        
        metrics = {
            'total_execution_time': total_time,
            'retriever_performance': {},
            'result_distribution': {},
            'efficiency_metrics': {}
        }
        
        # Per-retriever metrics
        for retriever_name, results in retriever_results.items():
            if results:
                avg_score = sum(r.get('similarity_score', 0) for r in results) / len(results)
                exec_times = [r.get('execution_time', 0) for r in results if 'execution_time' in r]
                avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
                
                metrics['retriever_performance'][retriever_name] = {
                    'result_count': len(results),
                    'average_score': avg_score,
                    'average_execution_time': avg_exec_time,
                    'results_per_second': len(results) / avg_exec_time if avg_exec_time > 0 else 0
                }
        
        # Result distribution
        total_results = sum(len(results) for results in retriever_results.values())
        for retriever_name, results in retriever_results.items():
            metrics['result_distribution'][retriever_name] = {
                'count': len(results),
                'percentage': len(results) / total_results if total_results > 0 else 0
            }
        
        # Efficiency metrics
        metrics['efficiency_metrics'] = {
            'results_per_second': total_results / total_time if total_time > 0 else 0,
            'average_results_per_retriever': total_results / len(retriever_results) if retriever_results else 0,
            'execution_efficiency': 1.0 / total_time if total_time > 0 else 0
        }
        
        return metrics
    
    def _update_execution_stats(self, result: Dict[str, Any], execution_time: float):
        """Update execution statistics"""
        self.execution_stats["total_executions"] += 1
        
        if result.get('success', False):
            self.execution_stats["successful_executions"] += 1
        
        self.execution_stats["total_execution_time"] += execution_time
        self.execution_stats["average_execution_time"] = (
            self.execution_stats["total_execution_time"] / 
            self.execution_stats["total_executions"]
        )
        
        # Update retriever usage
        for retriever_name in result.get('execution_metadata', {}).get('retrievers_used', []):
            if retriever_name in self.execution_stats["retriever_usage"]:
                self.execution_stats["retriever_usage"][retriever_name] += 1
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with reason"""
        return {
            'success': False,
            'results': [],
            'error': reason,
            'execution_metadata': {
                'execution_time': 0.0,
                'retrievers_used': [],
                'total_results_before_fusion': 0,
                'final_result_count': 0
            }
        }
    
    def get_registered_retrievers(self) -> List[str]:
        """Get list of registered retrievers"""
        return list(self.retrievers.keys())
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self.execution_stats.copy()
        
        # Add success rate
        if stats["total_executions"] > 0:
            stats["success_rate"] = stats["successful_executions"] / stats["total_executions"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check executor health"""
        health = {
            'status': 'healthy',
            'issues': [],
            'registered_retrievers': len(self.retrievers),
            'retriever_health': {}
        }
        
        # Check each retriever
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'health_check'):
                retriever_health = retriever.health_check()
                health['retriever_health'][name] = retriever_health
                
                if retriever_health.get('status') != 'healthy':
                    health['issues'].append(f"Retriever {name} is {retriever_health.get('status')}")
        
        # Overall health assessment
        if health['issues']:
            health['status'] = 'warning' if len(health['issues']) < len(self.retrievers) else 'unhealthy'
        
        return health