"""
Cross-Encoder Reranker - Fast First-Stage Reranking

Implements fast cross-encoder reranking to reduce large result sets
from ~200 to ~30 candidates for further processing.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Fast cross-encoder for first-stage reranking"""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 512,
                 batch_size: int = 32,
                 device: str = None):
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info(f"Loaded cross-encoder model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.model = None
        
        # Performance tracking
        self.reranking_stats = {
            "total_reranks": 0,
            "total_candidates": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "average_candidates_per_rerank": 0.0,
            "throughput_candidates_per_second": 0.0
        }
    
    def rerank(self, 
               query: str, 
               candidates: List[Dict[str, Any]], 
               top_k: int = 30,
               score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder
        
        Args:
            query: Search query
            candidates: List of candidate documents
            top_k: Number of top candidates to return
            score_threshold: Optional minimum score threshold
        
        Returns:
            Reranked list of top candidates
        """
        if not self.model or not candidates:
            logger.warning("Cross-encoder model not available or no candidates provided")
            return candidates[:top_k]
        
        start_time = time.time()
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for candidate in candidates:
                doc_text = self._extract_document_text(candidate)
                query_doc_pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            scores = self.model.predict(query_doc_pairs)
            
            # Combine scores with candidates
            scored_candidates = []
            for i, (candidate, score) in enumerate(zip(candidates, scores)):
                # Preserve original scores
                enhanced_candidate = candidate.copy()
                enhanced_candidate.update({
                    'cross_encoder_score': float(score),
                    'original_rank': i + 1,
                    'reranker_stage': 'cross_encoder_stage1'
                })
                
                # Apply threshold if specified
                if score_threshold is None or score >= score_threshold:
                    scored_candidates.append(enhanced_candidate)
            
            # Sort by cross-encoder score
            scored_candidates.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            # Take top-k
            reranked_candidates = scored_candidates[:top_k]
            
            # Update final ranks
            for i, candidate in enumerate(reranked_candidates):
                candidate['reranked_rank'] = i + 1
                candidate['rank_change'] = candidate['original_rank'] - (i + 1)
            
            # Update statistics
            rerank_time = time.time() - start_time
            self._update_stats(len(candidates), rerank_time)
            
            logger.debug(f"Cross-encoder reranked {len(candidates)} → {len(reranked_candidates)} candidates in {rerank_time:.3f}s")
            
            return reranked_candidates
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return candidates[:top_k]
    
    def batch_rerank(self, 
                    queries: List[str], 
                    candidate_lists: List[List[Dict[str, Any]]], 
                    top_k: int = 30) -> List[List[Dict[str, Any]]]:
        """Batch reranking for multiple queries"""
        
        if len(queries) != len(candidate_lists):
            raise ValueError("Number of queries must match number of candidate lists")
        
        reranked_results = []
        
        for query, candidates in zip(queries, candidate_lists):
            reranked = self.rerank(query, candidates, top_k)
            reranked_results.append(reranked)
        
        return reranked_results
    
    def _extract_document_text(self, candidate: Dict[str, Any]) -> str:
        """Extract text content from candidate document"""
        
        # Try different content fields
        content_fields = ['content', 'text', 'body', 'passage', 'document']
        
        for field in content_fields:
            if field in candidate and candidate[field]:
                content = str(candidate[field])
                # Truncate if too long
                if len(content) > self.max_length * 4:  # Rough character limit
                    content = content[:self.max_length * 4]
                return content
        
        # Fallback: use title or id
        if 'title' in candidate:
            return str(candidate['title'])
        elif 'id' in candidate:
            return f"Document {candidate['id']}"
        else:
            return "No content available"
    
    def _update_stats(self, num_candidates: int, rerank_time: float):
        """Update reranking statistics"""
        
        self.reranking_stats["total_reranks"] += 1
        self.reranking_stats["total_candidates"] += num_candidates
        self.reranking_stats["total_time"] += rerank_time
        
        # Calculate averages
        total_reranks = self.reranking_stats["total_reranks"]
        self.reranking_stats["average_time"] = (
            self.reranking_stats["total_time"] / total_reranks
        )
        self.reranking_stats["average_candidates_per_rerank"] = (
            self.reranking_stats["total_candidates"] / total_reranks
        )
        self.reranking_stats["throughput_candidates_per_second"] = (
            self.reranking_stats["total_candidates"] / self.reranking_stats["total_time"]
        )
    
    def analyze_reranking_impact(self, 
                               original_candidates: List[Dict[str, Any]], 
                               reranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of reranking on result ordering"""
        
        analysis = {
            "original_count": len(original_candidates),
            "reranked_count": len(reranked_candidates),
            "rank_changes": [],
            "score_improvements": [],
            "top_promoted": [],
            "top_demoted": []
        }
        
        # Create mapping of document IDs to original ranks
        original_ranks = {}
        for i, candidate in enumerate(original_candidates):
            doc_id = candidate.get('id', f'doc_{i}')
            original_ranks[doc_id] = i + 1
        
        # Analyze rank changes
        rank_changes = []
        score_improvements = []
        
        for i, candidate in enumerate(reranked_candidates):
            doc_id = candidate.get('id', f'doc_{i}')
            new_rank = i + 1
            original_rank = original_ranks.get(doc_id, len(original_candidates) + 1)
            
            rank_change = original_rank - new_rank  # Positive = promoted
            rank_changes.append(rank_change)
            
            # Score improvement
            original_score = candidate.get('similarity_score', 0.0)
            cross_encoder_score = candidate.get('cross_encoder_score', 0.0)
            score_improvement = cross_encoder_score - original_score
            score_improvements.append(score_improvement)
            
            # Track significant changes
            if rank_change > 10:  # Promoted significantly
                analysis["top_promoted"].append({
                    "doc_id": doc_id,
                    "original_rank": original_rank,
                    "new_rank": new_rank,
                    "rank_change": rank_change,
                    "cross_encoder_score": cross_encoder_score
                })
            elif rank_change < -10:  # Demoted significantly
                analysis["top_demoted"].append({
                    "doc_id": doc_id,
                    "original_rank": original_rank,
                    "new_rank": new_rank,
                    "rank_change": rank_change,
                    "cross_encoder_score": cross_encoder_score
                })
        
        analysis["rank_changes"] = rank_changes
        analysis["score_improvements"] = score_improvements
        
        # Summary statistics
        analysis["summary"] = {
            "avg_rank_change": np.mean(rank_changes) if rank_changes else 0,
            "avg_score_improvement": np.mean(score_improvements) if score_improvements else 0,
            "promoted_count": sum(1 for change in rank_changes if change > 0),
            "demoted_count": sum(1 for change in rank_changes if change < 0),
            "unchanged_count": sum(1 for change in rank_changes if change == 0),
            "significant_promotions": len(analysis["top_promoted"]),
            "significant_demotions": len(analysis["top_demoted"])
        }
        
        return analysis
    
    def explain_reranking(self, 
                         query: str, 
                         candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Explain why a candidate was ranked at its position"""
        
        explanation = {
            "query": query,
            "document_id": candidate.get('id', 'unknown'),
            "cross_encoder_score": candidate.get('cross_encoder_score', 0.0),
            "original_rank": candidate.get('original_rank', 0),
            "reranked_rank": candidate.get('reranked_rank', 0),
            "rank_change": candidate.get('rank_change', 0),
            "explanation": ""
        }
        
        # Generate explanation based on scores and changes
        ce_score = explanation["cross_encoder_score"]
        rank_change = explanation["rank_change"]
        
        if ce_score > 0.8:
            explanation["explanation"] = "High cross-encoder relevance score indicates strong query-document match"
        elif ce_score > 0.5:
            explanation["explanation"] = "Moderate cross-encoder relevance score indicates reasonable query-document match"
        elif ce_score > 0.2:
            explanation["explanation"] = "Low cross-encoder relevance score indicates weak query-document match"
        else:
            explanation["explanation"] = "Very low cross-encoder relevance score indicates poor query-document match"
        
        # Add rank change context
        if rank_change > 5:
            explanation["explanation"] += f" (promoted {rank_change} positions due to better semantic relevance)"
        elif rank_change < -5:
            explanation["explanation"] += f" (demoted {abs(rank_change)} positions due to lower semantic relevance)"
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        
        info = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "model_loaded": self.model is not None
        }
        
        if self.model:
            try:
                # Try to get model parameters if available
                if hasattr(self.model, 'model'):
                    num_params = sum(p.numel() for p in self.model.model.parameters())
                    info["num_parameters"] = num_params
            except:
                pass
        
        return info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranking statistics"""
        return self.reranking_stats.copy()
    
    def benchmark_performance(self, 
                            query: str, 
                            candidates: List[Dict[str, Any]], 
                            num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark reranking performance"""
        
        if not self.model or not candidates:
            return {"error": "Model not available or no candidates"}
        
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            self.rerank(query, candidates, top_k=30)
            end_time = time.time()
            times.append(end_time - start_time)
        
        benchmark = {
            "num_runs": num_runs,
            "num_candidates": len(candidates),
            "times": times,
            "avg_time": np.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": np.std(times),
            "throughput_candidates_per_second": len(candidates) / np.mean(times)
        }
        
        return benchmark
    
    def health_check(self) -> Dict[str, Any]:
        """Check reranker health"""
        
        health = {
            "status": "healthy",
            "issues": []
        }
        
        if not self.model:
            health["status"] = "unhealthy"
            health["issues"].append("Cross-encoder model not loaded")
        
        # Test with dummy data
        try:
            if self.model:
                dummy_pairs = [["test query", "test document"]]
                scores = self.model.predict(dummy_pairs)
                if len(scores) != 1:
                    health["status"] = "warning"
                    health["issues"].append("Model prediction returned unexpected results")
        except Exception as e:
            health["status"] = "unhealthy"
            health["issues"].append(f"Model prediction failed: {str(e)}")
        
        return health
    
    def clear_cache(self):
        """Clear any cached data"""
        # Cross-encoder doesn't typically cache, but we can clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cross-encoder cache cleared")