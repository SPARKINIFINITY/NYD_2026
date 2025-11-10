"""
Reranker Orchestrator - Integration with Retrieval System

Orchestrates the complete reranking flow by integrating with the retrieval
system to provide end-to-end query processing with maximum precision.
"""

import sys
import os
from typing import Dict, List, Any, Optional
import logging
import time
from datetime import datetime

# Add paths for integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'executor'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'planner'))

from .cascade_reranker import CascadeReranker

try:
    from executor import ExecutorOrchestrator
except ImportError:
    ExecutorOrchestrator = None
    logging.warning("Executor not available - running in standalone mode")

try:
    from planner import AgenticPlanner
except ImportError:
    AgenticPlanner = None
    logging.warning("Planner not available - running in standalone mode")

logger = logging.getLogger(__name__)

class RerankerOrchestrator:
    """Orchestrates complete query processing with reranking integration"""
    
    def __init__(self, 
                 executor_orchestrator: ExecutorOrchestrator = None,
                 planner: AgenticPlanner = None,
                 stage1_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 stage2_model: str = "cross-encoder/ms-marco-electra-base",
                 use_llm_scoring: bool = False,
                 rerank_threshold: int = 50,  # Minimum candidates to trigger reranking
                 final_results_count: int = 7):
        
        self.executor_orchestrator = executor_orchestrator
        self.planner = planner
        self.rerank_threshold = rerank_threshold
        self.final_results_count = final_results_count
        
        # Initialize cascade reranker
        self.cascade_reranker = CascadeReranker(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            use_llm_scoring=use_llm_scoring,
            final_top_k=final_results_count
        )
        
        # Orchestration statistics
        self.orchestration_stats = {
            "total_queries": 0,
            "reranked_queries": 0,
            "skipped_reranking": 0,
            "total_processing_time": 0.0,
            "total_reranking_time": 0.0,
            "average_processing_time": 0.0,
            "average_reranking_time": 0.0,
            "precision_improvements": []
        }
        
        logger.info(f"Initialized reranker orchestrator with threshold: {rerank_threshold}")
    
    def process_query_with_reranking(self, 
                                   query: str,
                                   intent: str = None,
                                   entities: List[str] = None,
                                   session_id: str = None,
                                   optimization_goal: str = "balanced",
                                   force_reranking: bool = False,
                                   custom_final_k: int = None) -> Dict[str, Any]:
        """
        Process query with complete retrieval and reranking pipeline
        
        Args:
            query: User query
            intent: Detected intent
            entities: Extracted entities
            session_id: Session identifier
            optimization_goal: Optimization objective
            force_reranking: Force reranking even if below threshold
            custom_final_k: Override final result count
        
        Returns:
            Complete processing results with reranked results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query with reranking: '{query[:50]}...'")
            
            # Step 1: Execute retrieval pipeline
            if self.executor_orchestrator:
                retrieval_result = self.executor_orchestrator.process_query(
                    query=query,
                    intent=intent,
                    entities=entities,
                    session_id=session_id,
                    optimization_goal=optimization_goal
                )
            else:
                # Fallback: create mock retrieval result
                retrieval_result = self._create_mock_retrieval_result(query)
            
            if not retrieval_result.get("success", False):
                return self._create_error_result(
                    query, "Retrieval pipeline failed", retrieval_result.get("error", "Unknown error")
                )
            
            retrieval_results = retrieval_result.get("results", [])
            retrieval_time = retrieval_result.get("metadata", {}).get("response_time", 0)
            
            logger.debug(f"Retrieval completed: {len(retrieval_results)} results in {retrieval_time:.3f}s")
            
            # Step 2: Determine if reranking is needed
            should_rerank = (
                force_reranking or 
                len(retrieval_results) >= self.rerank_threshold
            )
            
            if not should_rerank:
                logger.debug(f"Skipping reranking: {len(retrieval_results)} < {self.rerank_threshold} threshold")
                
                final_results = retrieval_results[:custom_final_k or self.final_results_count]
                
                # Add reranking metadata
                for i, result in enumerate(final_results):
                    result['final_rank'] = i + 1
                    result['reranking_applied'] = False
                
                processing_time = time.time() - start_time
                
                self.orchestration_stats["total_queries"] += 1
                self.orchestration_stats["skipped_reranking"] += 1
                self.orchestration_stats["total_processing_time"] += processing_time
                
                return {
                    "success": True,
                    "query": query,
                    "results": final_results,
                    "reranking_applied": False,
                    "metadata": {
                        "total_processing_time": processing_time,
                        "retrieval_time": retrieval_time,
                        "reranking_time": 0.0,
                        "original_result_count": len(retrieval_results),
                        "final_result_count": len(final_results),
                        "reranking_skipped_reason": f"Below threshold ({len(retrieval_results)} < {self.rerank_threshold})"
                    }
                }
            
            # Step 3: Execute cascade reranking
            logger.debug(f"Executing cascade reranking on {len(retrieval_results)} candidates")
            
            # Get session context for signal processing
            session_context = self._get_session_context(session_id)
            
            reranking_start = time.time()
            cascade_result = self.cascade_reranker.rerank(
                query=query,
                candidates=retrieval_results,
                session_context=session_context,
                custom_final_k=custom_final_k or self.final_results_count
            )
            reranking_time = time.time() - reranking_start
            
            if not cascade_result.get("success", False):
                logger.error(f"Cascade reranking failed: {cascade_result.get('error')}")
                # Fallback to original results
                final_results = retrieval_results[:custom_final_k or self.final_results_count]
                reranking_applied = False
            else:
                final_results = cascade_result.get("final_results", [])
                reranking_applied = True
                
                logger.debug(f"Reranking completed: {len(retrieval_results)} → {len(final_results)} in {reranking_time:.3f}s")
            
            # Step 4: Analyze reranking impact
            reranking_analysis = self._analyze_reranking_impact(
                retrieval_results, final_results, reranking_applied
            )
            
            # Step 5: Learn from results (if planner available)
            if self.planner and reranking_applied:
                self._provide_reranking_feedback(
                    query, retrieval_result, cascade_result, reranking_analysis
                )
            
            # Step 6: Create final response
            processing_time = time.time() - start_time
            
            final_response = {
                "success": True,
                "query": query,
                "results": final_results,
                "reranking_applied": reranking_applied,
                "metadata": {
                    "total_processing_time": processing_time,
                    "retrieval_time": retrieval_time,
                    "reranking_time": reranking_time,
                    "original_result_count": len(retrieval_results),
                    "final_result_count": len(final_results),
                    "reranking_analysis": reranking_analysis,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Include cascade metadata if reranking was applied
            if reranking_applied and cascade_result.get("cascade_metadata"):
                final_response["metadata"]["cascade_metadata"] = cascade_result["cascade_metadata"]
            
            # Update statistics
            self._update_orchestration_stats(final_response, processing_time, reranking_time)
            
            logger.info(f"Query processing completed in {processing_time:.3f}s (reranking: {reranking_applied})")
            
            return final_response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query processing failed after {processing_time:.3f}s: {e}")
            
            return self._create_error_result(query, "Processing failed", str(e))
    
    def batch_process_with_reranking(self, 
                                   queries: List[str],
                                   intents: List[str] = None,
                                   entities_list: List[List[str]] = None,
                                   session_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Batch process multiple queries with reranking"""
        
        # Validate inputs
        num_queries = len(queries)
        intents = intents or [None] * num_queries
        entities_list = entities_list or [None] * num_queries
        session_ids = session_ids or [None] * num_queries
        
        if len(intents) != num_queries:
            raise ValueError("Number of intents must match number of queries")
        if len(entities_list) != num_queries:
            raise ValueError("Number of entity lists must match number of queries")
        if len(session_ids) != num_queries:
            raise ValueError("Number of session IDs must match number of queries")
        
        batch_results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing batch query {i+1}/{num_queries}")
            
            result = self.process_query_with_reranking(
                query=query,
                intent=intents[i],
                entities=entities_list[i],
                session_id=session_ids[i]
            )
            
            batch_results.append(result)
        
        return batch_results
    
    def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context for signal processing"""
        
        if not session_id or not self.planner:
            return {}
        
        try:
            # Get session insights from planner
            session_insights = self.planner.get_session_insights()
            
            return {
                "recent_entities": session_insights.get("current_context", {}).get("recent_entities", []),
                "recent_queries": session_insights.get("current_context", {}).get("recent_queries", []),
                "current_topic": session_insights.get("current_context", {}).get("current_topic"),
                "conversation_length": session_insights.get("session_overview", {}).get("total_queries", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            return {}
    
    def _analyze_reranking_impact(self, 
                                original_results: List[Dict[str, Any]], 
                                final_results: List[Dict[str, Any]], 
                                reranking_applied: bool) -> Dict[str, Any]:
        """Analyze the impact of reranking on results"""
        
        analysis = {
            "reranking_applied": reranking_applied,
            "original_count": len(original_results),
            "final_count": len(final_results),
            "compression_ratio": len(final_results) / len(original_results) if original_results else 0
        }
        
        if not reranking_applied or not original_results or not final_results:
            return analysis
        
        # Analyze score improvements
        original_top_scores = [r.get('similarity_score', 0.0) for r in original_results[:len(final_results)]]
        final_scores = [r.get('integrated_score', r.get('similarity_score', 0.0)) for r in final_results]
        
        if original_top_scores and final_scores:
            analysis["score_analysis"] = {
                "avg_original_score": sum(original_top_scores) / len(original_top_scores),
                "avg_final_score": sum(final_scores) / len(final_scores),
                "score_improvement": (sum(final_scores) / len(final_scores)) - (sum(original_top_scores) / len(original_top_scores)),
                "max_final_score": max(final_scores),
                "min_final_score": min(final_scores)
            }
        
        # Analyze ranking changes
        original_top_ids = [r.get('id', f'doc_{i}') for i, r in enumerate(original_results[:10])]
        final_ids = [r.get('id', f'doc_{i}') for i, r in enumerate(final_results)]
        
        # Check how many top original results survived
        survived_top_results = len(set(original_top_ids[:len(final_results)]).intersection(set(final_ids)))
        
        analysis["ranking_analysis"] = {
            "top_results_survived": survived_top_results,
            "survival_rate": survived_top_results / min(len(original_top_ids), len(final_results)) if original_top_ids and final_results else 0,
            "ranking_significantly_changed": survived_top_results < len(final_results) * 0.7
        }
        
        return analysis
    
    def _provide_reranking_feedback(self, 
                                  query: str,
                                  retrieval_result: Dict[str, Any], 
                                  cascade_result: Dict[str, Any], 
                                  reranking_analysis: Dict[str, Any]):
        """Provide feedback to planner about reranking effectiveness"""
        
        if not self.planner:
            return
        
        try:
            # Calculate effectiveness metrics
            score_improvement = reranking_analysis.get("score_analysis", {}).get("score_improvement", 0)
            final_count = reranking_analysis.get("final_count", 0)
            
            # Estimate user satisfaction based on reranking quality
            cascade_metadata = cascade_result.get("cascade_metadata", {})
            timing = cascade_metadata.get("timing", {})
            
            # Simple satisfaction estimate
            satisfaction_factors = []
            
            if score_improvement > 0.1:
                satisfaction_factors.append(0.3)  # Good score improvement
            
            if final_count >= 5:
                satisfaction_factors.append(0.2)  # Sufficient results
            
            if timing.get("total_time", 10) < 2.0:
                satisfaction_factors.append(0.2)  # Fast processing
            
            estimated_satisfaction = min(sum(satisfaction_factors) + 0.5, 1.0)  # Base 0.5 + bonuses
            
            # Create feedback for planner
            execution_results = {
                "success": True,
                "execution_time": timing.get("total_time", 0),
                "results_count": final_count,
                "quality_score": reranking_analysis.get("score_analysis", {}).get("avg_final_score", 0.5),
                "reranking_applied": True,
                "reranking_improvement": score_improvement
            }
            
            # Get original execution plan from retrieval result
            original_plan = retrieval_result.get("metadata", {}).get("execution_plan", {})
            
            if original_plan:
                self.planner.learn_from_execution_feedback(
                    original_plan,
                    execution_results,
                    estimated_satisfaction
                )
                
                logger.debug(f"Provided reranking feedback: satisfaction={estimated_satisfaction:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to provide reranking feedback: {e}")
    
    def _update_orchestration_stats(self, result: Dict[str, Any], processing_time: float, reranking_time: float):
        """Update orchestration statistics"""
        
        self.orchestration_stats["total_queries"] += 1
        self.orchestration_stats["total_processing_time"] += processing_time
        
        if result.get("reranking_applied", False):
            self.orchestration_stats["reranked_queries"] += 1
            self.orchestration_stats["total_reranking_time"] += reranking_time
            
            # Track precision improvements
            reranking_analysis = result.get("metadata", {}).get("reranking_analysis", {})
            score_improvement = reranking_analysis.get("score_analysis", {}).get("score_improvement", 0)
            
            if score_improvement > 0:
                self.orchestration_stats["precision_improvements"].append(score_improvement)
        
        # Calculate averages
        total_queries = self.orchestration_stats["total_queries"]
        self.orchestration_stats["average_processing_time"] = (
            self.orchestration_stats["total_processing_time"] / total_queries
        )
        
        reranked_queries = self.orchestration_stats["reranked_queries"]
        if reranked_queries > 0:
            self.orchestration_stats["average_reranking_time"] = (
                self.orchestration_stats["total_reranking_time"] / reranked_queries
            )
    
    def _create_mock_retrieval_result(self, query: str) -> Dict[str, Any]:
        """Create mock retrieval result for testing"""
        
        mock_results = [
            {
                "id": f"doc_{i}",
                "content": f"Mock document {i} content related to: {query}",
                "similarity_score": 0.9 - (i * 0.1),
                "retriever_type": "mock"
            }
            for i in range(10)
        ]
        
        return {
            "success": True,
            "results": mock_results,
            "metadata": {
                "response_time": 0.1,
                "execution_plan": {"retrievers": ["mock"], "mock_k": 10}
            }
        }
    
    def _create_error_result(self, query: str, error_type: str, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        
        return {
            "success": False,
            "query": query,
            "error": error_message,
            "error_type": error_type,
            "results": [],
            "reranking_applied": False,
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        
        report = {
            "orchestration_stats": self.orchestration_stats.copy(),
            "cascade_performance": self.cascade_reranker.get_performance_analysis(),
            "system_efficiency": {}
        }
        
        # Calculate system efficiency metrics
        stats = self.orchestration_stats
        
        if stats["total_queries"] > 0:
            reranking_rate = stats["reranked_queries"] / stats["total_queries"]
            avg_processing_time = stats["average_processing_time"]
            
            report["system_efficiency"] = {
                "reranking_rate": reranking_rate,
                "average_processing_time": avg_processing_time,
                "queries_per_minute": 60 / avg_processing_time if avg_processing_time > 0 else 0,
                "reranking_overhead": stats.get("average_reranking_time", 0) / avg_processing_time if avg_processing_time > 0 else 0
            }
            
            # Precision improvement analysis
            if stats["precision_improvements"]:
                import numpy as np
                improvements = stats["precision_improvements"]
                
                report["precision_analysis"] = {
                    "queries_with_improvement": len(improvements),
                    "average_improvement": np.mean(improvements),
                    "max_improvement": max(improvements),
                    "improvement_std": np.std(improvements),
                    "significant_improvements": sum(1 for imp in improvements if imp > 0.2)
                }
        
        return report
    
    def benchmark_end_to_end(self, 
                           test_queries: List[str], 
                           num_runs: int = 3) -> Dict[str, Any]:
        """Benchmark end-to-end performance"""
        
        benchmark_results = {
            "test_queries": len(test_queries),
            "num_runs": num_runs,
            "runs": [],
            "summary": {}
        }
        
        for run in range(num_runs):
            logger.info(f"Benchmark run {run + 1}/{num_runs}")
            
            run_results = []
            run_start = time.time()
            
            for query in test_queries:
                result = self.process_query_with_reranking(query)
                
                run_results.append({
                    "query": query,
                    "success": result.get("success", False),
                    "processing_time": result.get("metadata", {}).get("total_processing_time", 0),
                    "reranking_applied": result.get("reranking_applied", False),
                    "final_count": len(result.get("results", []))
                })
            
            run_time = time.time() - run_start
            
            benchmark_results["runs"].append({
                "run": run + 1,
                "total_time": run_time,
                "results": run_results
            })
        
        # Calculate summary statistics
        if benchmark_results["runs"]:
            import numpy as np
            
            all_processing_times = []
            all_reranking_applied = []
            
            for run in benchmark_results["runs"]:
                for result in run["results"]:
                    all_processing_times.append(result["processing_time"])
                    all_reranking_applied.append(result["reranking_applied"])
            
            benchmark_results["summary"] = {
                "avg_processing_time": np.mean(all_processing_times),
                "std_processing_time": np.std(all_processing_times),
                "reranking_rate": sum(all_reranking_applied) / len(all_reranking_applied),
                "total_queries_processed": len(all_processing_times),
                "throughput_queries_per_second": len(all_processing_times) / sum(run["total_time"] for run in benchmark_results["runs"])
            }
        
        return benchmark_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "orchestrator_status": "healthy",
            "components": {
                "cascade_reranker": self.cascade_reranker.health_check(),
                "executor_orchestrator": "available" if self.executor_orchestrator else "not_available",
                "planner": "available" if self.planner else "not_available"
            },
            "configuration": {
                "rerank_threshold": self.rerank_threshold,
                "final_results_count": self.final_results_count,
                "cascade_config": self.cascade_reranker.get_system_info()
            },
            "performance_stats": self.orchestration_stats
        }
        
        # Overall health assessment
        cascade_health = status["components"]["cascade_reranker"]["status"]
        
        if cascade_health != "healthy":
            status["orchestrator_status"] = "warning"
        
        return status