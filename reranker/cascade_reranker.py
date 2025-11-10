"""
Cascade Reranker - Two-Stage Orchestrator

Orchestrates the complete two-stage reranking process:
Stage 1: Fast cross-encoder (200 → 30)
Stage 2: Advanced reranker (30 → 3-7)
"""

import time
from typing import List, Dict, Any, Optional
import logging

from .cross_encoder_reranker import CrossEncoderReranker
from .advanced_reranker import AdvancedReranker
from .signal_processor import SignalProcessor

logger = logging.getLogger(__name__)

class CascadeReranker:
    """Two-stage cascade reranking system"""
    
    def __init__(self, 
                 stage1_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 stage2_model: str = "cross-encoder/ms-marco-electra-base",
                 use_llm_scoring: bool = False,
                 stage1_top_k: int = 30,
                 final_top_k: int = 7,
                 enable_signals: bool = True,
                 signal_weight: float = 0.3):
        
        self.stage1_top_k = stage1_top_k
        self.final_top_k = final_top_k
        self.enable_signals = enable_signals
        self.signal_weight = signal_weight
        
        # Initialize components
        self.stage1_reranker = CrossEncoderReranker(
            model_name=stage1_model,
            batch_size=32  # Larger batch for fast model
        )
        
        self.stage2_reranker = AdvancedReranker(
            model_name=stage2_model,
            use_llm_scoring=use_llm_scoring,
            batch_size=16  # Smaller batch for advanced model
        )
        
        self.signal_processor = SignalProcessor() if enable_signals else None
        
        # Cascade statistics
        self.cascade_stats = {
            "total_cascades": 0,
            "total_input_candidates": 0,
            "total_output_results": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "stage1_time": 0.0,
            "stage2_time": 0.0,
            "signal_processing_time": 0.0,
            "compression_ratios": []
        }
        
        logger.info(f"Initialized cascade reranker: {stage1_model} → {stage2_model}")
    
    def rerank(self, 
               query: str, 
               candidates: List[Dict[str, Any]], 
               session_context: Dict[str, Any] = None,
               custom_stage1_k: int = None,
               custom_final_k: int = None) -> Dict[str, Any]:
        """
        Execute complete two-stage cascade reranking
        
        Args:
            query: Search query
            candidates: Initial candidate list (~200)
            session_context: Optional session context for signals
            custom_stage1_k: Override stage 1 top-k
            custom_final_k: Override final top-k
        
        Returns:
            Complete reranking results with metadata
        """
        start_time = time.time()
        
        # Use custom k values if provided
        stage1_k = custom_stage1_k or self.stage1_top_k
        final_k = custom_final_k or self.final_top_k
        
        try:
            logger.info(f"Starting cascade reranking: {len(candidates)} → {stage1_k} → {final_k}")
            
            # Stage 1: Fast cross-encoder reranking
            stage1_start = time.time()
            stage1_results = self.stage1_reranker.rerank(
                query, candidates, top_k=stage1_k
            )
            stage1_time = time.time() - stage1_start
            
            logger.debug(f"Stage 1 completed: {len(candidates)} → {len(stage1_results)} in {stage1_time:.3f}s")
            
            # Signal processing (if enabled)
            signal_time = 0.0
            if self.enable_signals and self.signal_processor:
                signal_start = time.time()
                stage1_results = self.signal_processor.process_signals(
                    query, stage1_results, session_context
                )
                signal_time = time.time() - signal_start
                logger.debug(f"Signal processing completed in {signal_time:.3f}s")
            
            # Stage 2: Advanced reranking
            stage2_start = time.time()
            final_results = self.stage2_reranker.rerank(
                query, 
                stage1_results, 
                top_k=final_k,
                use_signals=self.enable_signals,
                signal_weight=self.signal_weight
            )
            stage2_time = time.time() - stage2_start
            
            logger.debug(f"Stage 2 completed: {len(stage1_results)} → {len(final_results)} in {stage2_time:.3f}s")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Create comprehensive result
            cascade_result = {
                "success": True,
                "query": query,
                "final_results": final_results,
                "cascade_metadata": {
                    "input_count": len(candidates),
                    "stage1_count": len(stage1_results),
                    "final_count": len(final_results),
                    "stage1_compression": len(stage1_results) / len(candidates) if candidates else 0,
                    "stage2_compression": len(final_results) / len(stage1_results) if stage1_results else 0,
                    "overall_compression": len(final_results) / len(candidates) if candidates else 0,
                    "timing": {
                        "total_time": total_time,
                        "stage1_time": stage1_time,
                        "stage2_time": stage2_time,
                        "signal_processing_time": signal_time,
                        "stage1_percentage": (stage1_time / total_time) * 100,
                        "stage2_percentage": (stage2_time / total_time) * 100
                    },
                    "models_used": {
                        "stage1_model": self.stage1_reranker.model_name,
                        "stage2_model": self.stage2_reranker.model_name,
                        "llm_scoring": self.stage2_reranker.use_llm_scoring,
                        "signals_enabled": self.enable_signals
                    }
                },
                "intermediate_results": {
                    "stage1_results": stage1_results[:10],  # Store top 10 for analysis
                },
                "quality_analysis": self._analyze_cascade_quality(
                    candidates, stage1_results, final_results
                )
            }
            
            # Update statistics
            self._update_cascade_stats(cascade_result)
            
            logger.info(f"Cascade reranking completed in {total_time:.3f}s")
            
            return cascade_result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Cascade reranking failed after {total_time:.3f}s: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "final_results": candidates[:final_k],  # Fallback
                "cascade_metadata": {
                    "input_count": len(candidates),
                    "final_count": min(len(candidates), final_k),
                    "timing": {"total_time": total_time},
                    "error_occurred": True
                }
            }
    
    def batch_rerank(self, 
                    queries: List[str], 
                    candidate_lists: List[List[Dict[str, Any]]], 
                    session_contexts: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Batch cascade reranking for multiple queries"""
        
        if len(queries) != len(candidate_lists):
            raise ValueError("Number of queries must match number of candidate lists")
        
        if session_contexts and len(session_contexts) != len(queries):
            raise ValueError("Number of session contexts must match number of queries")
        
        batch_results = []
        
        for i, (query, candidates) in enumerate(zip(queries, candidate_lists)):
            session_context = session_contexts[i] if session_contexts else None
            
            result = self.rerank(query, candidates, session_context)
            batch_results.append(result)
        
        return batch_results
    
    def _analyze_cascade_quality(self, 
                               original_candidates: List[Dict[str, Any]], 
                               stage1_results: List[Dict[str, Any]], 
                               final_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality improvements through cascade stages"""
        
        analysis = {
            "stage_progression": {
                "original_count": len(original_candidates),
                "stage1_count": len(stage1_results),
                "final_count": len(final_results)
            },
            "score_improvements": {},
            "ranking_stability": {},
            "quality_metrics": {}
        }
        
        # Analyze score improvements
        if final_results:
            original_scores = [c.get('similarity_score', 0.0) for c in original_candidates[:len(final_results)]]
            final_scores = [r.get('integrated_score', r.get('advanced_cross_encoder_score', 0.0)) for r in final_results]
            
            analysis["score_improvements"] = {
                "avg_original_score": sum(original_scores) / len(original_scores) if original_scores else 0,
                "avg_final_score": sum(final_scores) / len(final_scores) if final_scores else 0,
                "score_improvement": (sum(final_scores) / len(final_scores)) - (sum(original_scores) / len(original_scores)) if original_scores and final_scores else 0
            }
        
        # Analyze ranking stability (how much rankings changed)
        if len(final_results) >= 3:
            # Check if top results are from original top results
            original_top_ids = set(c.get('id', f'doc_{i}') for i, c in enumerate(original_candidates[:10]))
            final_top_ids = set(r.get('id', f'doc_{i}') for i, r in enumerate(final_results[:3]))
            
            overlap = len(original_top_ids.intersection(final_top_ids))
            analysis["ranking_stability"] = {
                "top_overlap_count": overlap,
                "top_overlap_percentage": (overlap / min(len(original_top_ids), len(final_top_ids))) * 100 if original_top_ids and final_top_ids else 0,
                "ranking_changed": overlap < len(final_top_ids)
            }
        
        # Quality metrics
        if final_results:
            final_scores = [r.get('integrated_score', 0.0) for r in final_results]
            
            analysis["quality_metrics"] = {
                "high_quality_results": sum(1 for score in final_scores if score > 0.8),
                "medium_quality_results": sum(1 for score in final_scores if 0.5 <= score <= 0.8),
                "low_quality_results": sum(1 for score in final_scores if score < 0.5),
                "quality_distribution": {
                    "excellent": sum(1 for score in final_scores if score > 0.9),
                    "good": sum(1 for score in final_scores if 0.7 <= score <= 0.9),
                    "fair": sum(1 for score in final_scores if 0.5 <= score < 0.7),
                    "poor": sum(1 for score in final_scores if score < 0.5)
                }
            }
        
        return analysis
    
    def _update_cascade_stats(self, cascade_result: Dict[str, Any]):
        """Update cascade statistics"""
        
        metadata = cascade_result.get("cascade_metadata", {})
        timing = metadata.get("timing", {})
        
        self.cascade_stats["total_cascades"] += 1
        self.cascade_stats["total_input_candidates"] += metadata.get("input_count", 0)
        self.cascade_stats["total_output_results"] += metadata.get("final_count", 0)
        
        total_time = timing.get("total_time", 0)
        self.cascade_stats["total_time"] += total_time
        self.cascade_stats["stage1_time"] += timing.get("stage1_time", 0)
        self.cascade_stats["stage2_time"] += timing.get("stage2_time", 0)
        self.cascade_stats["signal_processing_time"] += timing.get("signal_processing_time", 0)
        
        # Calculate averages
        total_cascades = self.cascade_stats["total_cascades"]
        self.cascade_stats["average_time"] = self.cascade_stats["total_time"] / total_cascades
        
        # Track compression ratios
        overall_compression = metadata.get("overall_compression", 0)
        self.cascade_stats["compression_ratios"].append(overall_compression)
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get detailed performance analysis of cascade system"""
        
        stats = self.cascade_stats.copy()
        
        # Calculate additional metrics
        if stats["total_cascades"] > 0:
            stats["average_input_candidates"] = stats["total_input_candidates"] / stats["total_cascades"]
            stats["average_output_results"] = stats["total_output_results"] / stats["total_cascades"]
            stats["average_stage1_time"] = stats["stage1_time"] / stats["total_cascades"]
            stats["average_stage2_time"] = stats["stage2_time"] / stats["total_cascades"]
            
            if stats["compression_ratios"]:
                import numpy as np
                stats["compression_statistics"] = {
                    "mean_compression": np.mean(stats["compression_ratios"]),
                    "std_compression": np.std(stats["compression_ratios"]),
                    "min_compression": min(stats["compression_ratios"]),
                    "max_compression": max(stats["compression_ratios"])
                }
        
        # Component statistics
        stats["component_stats"] = {
            "stage1_reranker": self.stage1_reranker.get_stats(),
            "stage2_reranker": self.stage2_reranker.get_stats()
        }
        
        if self.signal_processor:
            stats["signal_processor"] = self.signal_processor.get_stats()
        
        return stats
    
    def benchmark_cascade(self, 
                         query: str, 
                         candidates: List[Dict[str, Any]], 
                         num_runs: int = 3) -> Dict[str, Any]:
        """Benchmark cascade performance"""
        
        if not candidates:
            return {"error": "No candidates provided"}
        
        benchmark_results = {
            "num_runs": num_runs,
            "input_candidates": len(candidates),
            "runs": [],
            "summary": {}
        }
        
        for run in range(num_runs):
            logger.info(f"Benchmark run {run + 1}/{num_runs}")
            
            result = self.rerank(query, candidates)
            
            if result.get("success"):
                timing = result["cascade_metadata"]["timing"]
                benchmark_results["runs"].append({
                    "run": run + 1,
                    "total_time": timing["total_time"],
                    "stage1_time": timing["stage1_time"],
                    "stage2_time": timing["stage2_time"],
                    "final_count": result["cascade_metadata"]["final_count"]
                })
        
        # Calculate summary statistics
        if benchmark_results["runs"]:
            import numpy as np
            
            total_times = [run["total_time"] for run in benchmark_results["runs"]]
            stage1_times = [run["stage1_time"] for run in benchmark_results["runs"]]
            stage2_times = [run["stage2_time"] for run in benchmark_results["runs"]]
            
            benchmark_results["summary"] = {
                "avg_total_time": np.mean(total_times),
                "std_total_time": np.std(total_times),
                "avg_stage1_time": np.mean(stage1_times),
                "avg_stage2_time": np.mean(stage2_times),
                "throughput_candidates_per_second": len(candidates) / np.mean(total_times),
                "stage1_percentage": (np.mean(stage1_times) / np.mean(total_times)) * 100,
                "stage2_percentage": (np.mean(stage2_times) / np.mean(total_times)) * 100
            }
        
        return benchmark_results
    
    def explain_cascade_decision(self, 
                               cascade_result: Dict[str, Any], 
                               result_index: int = 0) -> Dict[str, Any]:
        """Explain why a specific result was selected through cascade"""
        
        final_results = cascade_result.get("final_results", [])
        
        if result_index >= len(final_results):
            return {"error": "Result index out of range"}
        
        result = final_results[result_index]
        metadata = cascade_result.get("cascade_metadata", {})
        
        explanation = {
            "result_position": result_index + 1,
            "document_id": result.get('id', 'unknown'),
            "cascade_journey": {
                "original_rank": result.get('original_rank', 'unknown'),
                "stage1_rank": result.get('reranked_rank', 'unknown'),
                "final_rank": result.get('advanced_rerank_position', result_index + 1)
            },
            "score_evolution": {
                "original_score": result.get('similarity_score', 0.0),
                "stage1_score": result.get('cross_encoder_score', 0.0),
                "stage2_score": result.get('advanced_cross_encoder_score', 0.0),
                "final_score": result.get('integrated_score', 0.0)
            },
            "selection_factors": [],
            "cascade_efficiency": {
                "survived_stage1": True,
                "survived_stage2": True,
                "compression_survived": f"{metadata.get('input_count', 0)} → {metadata.get('final_count', 0)}"
            }
        }
        
        # Analyze selection factors
        final_score = explanation["score_evolution"]["final_score"]
        
        if final_score > 0.8:
            explanation["selection_factors"].append("High final integrated score")
        
        if result.get('advanced_cross_encoder_score', 0) > 0.7:
            explanation["selection_factors"].append("Strong stage-2 cross-encoder score")
        
        if result.get('combined_signal_score', 0) > 0.6:
            explanation["selection_factors"].append("Strong multi-signal alignment")
        
        # Rank improvement analysis
        original_rank = result.get('original_rank', float('inf'))
        final_rank = result_index + 1
        
        if isinstance(original_rank, int) and original_rank > final_rank:
            improvement = original_rank - final_rank
            explanation["selection_factors"].append(f"Promoted {improvement} positions through cascade")
        
        return explanation
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        return {
            "cascade_configuration": {
                "stage1_top_k": self.stage1_top_k,
                "final_top_k": self.final_top_k,
                "enable_signals": self.enable_signals,
                "signal_weight": self.signal_weight
            },
            "component_info": {
                "stage1_reranker": self.stage1_reranker.get_model_info(),
                "stage2_reranker": self.stage2_reranker.get_model_info()
            },
            "performance_stats": self.cascade_stats
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of entire cascade system"""
        
        health = {
            "status": "healthy",
            "issues": [],
            "component_health": {}
        }
        
        # Check stage 1 reranker
        stage1_health = self.stage1_reranker.health_check()
        health["component_health"]["stage1_reranker"] = stage1_health
        
        if stage1_health["status"] != "healthy":
            health["status"] = "warning"
            health["issues"].extend([f"Stage1: {issue}" for issue in stage1_health["issues"]])
        
        # Check stage 2 reranker
        stage2_health = self.stage2_reranker.health_check()
        health["component_health"]["stage2_reranker"] = stage2_health
        
        if stage2_health["status"] != "healthy":
            health["status"] = "warning"
            health["issues"].extend([f"Stage2: {issue}" for issue in stage2_health["issues"]])
        
        # Check signal processor
        if self.signal_processor:
            # Signal processor doesn't have health check, assume healthy
            health["component_health"]["signal_processor"] = {"status": "healthy"}
        
        return health