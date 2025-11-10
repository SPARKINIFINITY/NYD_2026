"""
Advanced Reranker - Second-Stage Strong Reranking

Implements stronger second-stage reranking using advanced cross-encoders
or small LLMs to refine top 30 candidates to final 3-7 results.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class AdvancedReranker:
    """Advanced second-stage reranker for final result refinement"""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-electra-base",
                 use_llm_scoring: bool = False,
                 llm_model: str = "microsoft/DialoGPT-medium",
                 max_length: int = 512,
                 batch_size: int = 16,
                 device: str = None):
        
        self.model_name = model_name
        self.use_llm_scoring = use_llm_scoring
        self.llm_model = llm_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.cross_encoder = None
        self.llm_scorer = None
        
        self._initialize_models()
        
        # Performance tracking
        self.reranking_stats = {
            "total_reranks": 0,
            "total_candidates": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "llm_scoring_used": 0,
            "cross_encoder_used": 0
        }
        
        logger.info(f"Initialized advanced reranker with model: {model_name}")
    
    def _initialize_models(self):
        """Initialize reranking models"""
        
        # Initialize cross-encoder
        try:
            self.cross_encoder = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded advanced cross-encoder: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
        
        # Initialize LLM scorer if requested
        if self.use_llm_scoring:
            try:
                self.llm_scorer = pipeline(
                    "text-generation",
                    model=self.llm_model,
                    device=0 if self.device == "cuda" else -1,
                    return_full_text=False,
                    max_new_tokens=50
                )
                logger.info(f"Loaded LLM scorer: {self.llm_model}")
            except Exception as e:
                logger.error(f"Failed to load LLM scorer: {e}")
                self.use_llm_scoring = False
    
    def rerank(self, 
               query: str, 
               candidates: List[Dict[str, Any]], 
               top_k: int = 7,
               use_signals: bool = True,
               signal_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Advanced reranking for final result selection
        
        Args:
            query: Search query
            candidates: List of candidate documents (typically ~30)
            top_k: Number of final results to return (3-7)
            use_signals: Whether to incorporate multi-signal scores
            signal_weight: Weight for signal scores vs model scores
        
        Returns:
            Final reranked results
        """
        if not candidates:
            return []
        
        start_time = time.time()
        
        try:
            # Stage 1: Cross-encoder scoring
            cross_encoder_candidates = self._cross_encoder_rerank(query, candidates)
            
            # Stage 2: LLM scoring (if enabled)
            if self.use_llm_scoring and self.llm_scorer:
                llm_scored_candidates = self._llm_score_candidates(query, cross_encoder_candidates[:15])
                self.reranking_stats["llm_scoring_used"] += 1
            else:
                llm_scored_candidates = cross_encoder_candidates
                self.reranking_stats["cross_encoder_used"] += 1
            
            # Stage 3: Signal integration
            if use_signals:
                final_candidates = self._integrate_signals(
                    llm_scored_candidates, signal_weight
                )
            else:
                final_candidates = llm_scored_candidates
            
            # Stage 4: Final ranking and selection
            final_results = self._final_ranking(final_candidates, top_k)
            
            # Add advanced reranking metadata
            for i, result in enumerate(final_results):
                result.update({
                    'advanced_rerank_position': i + 1,
                    'reranker_stage': 'advanced_stage2',
                    'final_selection': True
                })
            
            # Update statistics
            rerank_time = time.time() - start_time
            self._update_stats(len(candidates), rerank_time)
            
            logger.debug(f"Advanced reranked {len(candidates)} → {len(final_results)} candidates in {rerank_time:.3f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Advanced reranking failed: {e}")
            return candidates[:top_k]
    
    def _cross_encoder_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply advanced cross-encoder reranking"""
        
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available, skipping advanced cross-encoder reranking")
            return candidates
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for candidate in candidates:
                doc_text = self._extract_document_text(candidate)
                query_doc_pairs.append([query, doc_text])
            
            # Get advanced cross-encoder scores
            advanced_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Combine with candidates
            scored_candidates = []
            for candidate, score in zip(candidates, advanced_scores):
                enhanced_candidate = candidate.copy()
                enhanced_candidate['advanced_cross_encoder_score'] = float(score)
                scored_candidates.append(enhanced_candidate)
            
            # Sort by advanced cross-encoder score
            scored_candidates.sort(key=lambda x: x['advanced_cross_encoder_score'], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            logger.error(f"Advanced cross-encoder reranking failed: {e}")
            return candidates
    
    def _llm_score_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score candidates using LLM-based relevance assessment"""
        
        if not self.llm_scorer:
            return candidates
        
        try:
            scored_candidates = []
            
            for candidate in candidates:
                doc_text = self._extract_document_text(candidate)
                
                # Create relevance assessment prompt
                prompt = self._create_relevance_prompt(query, doc_text)
                
                # Get LLM relevance score
                llm_score = self._get_llm_relevance_score(prompt)
                
                enhanced_candidate = candidate.copy()
                enhanced_candidate['llm_relevance_score'] = llm_score
                scored_candidates.append(enhanced_candidate)
            
            # Sort by LLM score
            scored_candidates.sort(key=lambda x: x['llm_relevance_score'], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return candidates
    
    def _create_relevance_prompt(self, query: str, document: str) -> str:
        """Create prompt for LLM relevance assessment"""
        
        # Truncate document if too long
        if len(document) > 300:
            document = document[:300] + "..."
        
        prompt = f"""
        Query: {query}
        Document: {document}
        
        Rate the relevance of this document to the query on a scale of 0-10:
        - 0: Completely irrelevant
        - 5: Somewhat relevant
        - 10: Perfectly relevant
        
        Relevance score:"""
        
        return prompt
    
    def _get_llm_relevance_score(self, prompt: str) -> float:
        """Extract relevance score from LLM response"""
        
        try:
            # Generate response
            response = self.llm_scorer(prompt, max_new_tokens=10, temperature=0.1)[0]['generated_text']
            
            # Extract numeric score
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)', response)
            
            if score_match:
                score = float(score_match.group(1))
                # Normalize to 0-1 range
                return min(score / 10.0, 1.0)
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.error(f"LLM relevance scoring failed: {e}")
            return 0.5
    
    def _integrate_signals(self, candidates: List[Dict[str, Any]], signal_weight: float) -> List[Dict[str, Any]]:
        """Integrate multi-signal scores with model scores"""
        
        integrated_candidates = []
        
        for candidate in candidates:
            # Get model scores
            advanced_ce_score = candidate.get('advanced_cross_encoder_score', 0.0)
            llm_score = candidate.get('llm_relevance_score', 0.0)
            
            # Get signal score
            signal_score = candidate.get('combined_signal_score', 0.0)
            
            # Combine scores
            if llm_score > 0:
                model_score = (advanced_ce_score + llm_score) / 2
            else:
                model_score = advanced_ce_score
            
            # Weighted combination
            final_score = (model_score * (1 - signal_weight)) + (signal_score * signal_weight)
            
            enhanced_candidate = candidate.copy()
            enhanced_candidate['integrated_score'] = final_score
            enhanced_candidate['model_score'] = model_score
            enhanced_candidate['signal_contribution'] = signal_score * signal_weight
            
            integrated_candidates.append(enhanced_candidate)
        
        # Sort by integrated score
        integrated_candidates.sort(key=lambda x: x['integrated_score'], reverse=True)
        
        return integrated_candidates
    
    def _final_ranking(self, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Apply final ranking logic and selection"""
        
        if len(candidates) <= top_k:
            return candidates
        
        # Apply diversity filtering to avoid too similar results
        diverse_candidates = self._apply_diversity_filtering(candidates, top_k)
        
        # Apply quality threshold
        quality_filtered = self._apply_quality_threshold(diverse_candidates)
        
        # Ensure we have at least 3 results
        final_count = max(min(len(quality_filtered), top_k), min(3, len(candidates)))
        
        return quality_filtered[:final_count]
    
    def _apply_diversity_filtering(self, candidates: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Apply diversity filtering to avoid redundant results"""
        
        if len(candidates) <= target_count:
            return candidates
        
        diverse_results = []
        used_content_hashes = set()
        
        for candidate in candidates:
            # Create content hash for similarity detection
            content = candidate.get('content', '')
            content_hash = hash(content[:200])  # Use first 200 chars
            
            # Check for similarity with already selected results
            is_similar = False
            for used_hash in used_content_hashes:
                # Simple similarity check based on hash proximity
                if abs(content_hash - used_hash) < 1000:  # Arbitrary threshold
                    is_similar = True
                    break
            
            if not is_similar or len(diverse_results) < 3:  # Always include top 3
                diverse_results.append(candidate)
                used_content_hashes.add(content_hash)
                
                if len(diverse_results) >= target_count:
                    break
        
        return diverse_results
    
    def _apply_quality_threshold(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply quality threshold to filter low-quality results"""
        
        if not candidates:
            return candidates
        
        # Calculate dynamic threshold based on top result
        top_score = candidates[0].get('integrated_score', 0.0)
        
        # Set threshold as percentage of top score
        threshold = max(top_score * 0.6, 0.3)  # At least 60% of top score or 0.3 minimum
        
        quality_filtered = []
        for candidate in candidates:
            score = candidate.get('integrated_score', 0.0)
            if score >= threshold:
                quality_filtered.append(candidate)
        
        # Ensure at least 3 results if available
        if len(quality_filtered) < 3 and len(candidates) >= 3:
            return candidates[:3]
        
        return quality_filtered
    
    def _extract_document_text(self, candidate: Dict[str, Any]) -> str:
        """Extract text content from candidate document"""
        
        # Try different content fields
        content_fields = ['content', 'text', 'body', 'passage', 'document']
        
        for field in content_fields:
            if field in candidate and candidate[field]:
                content = str(candidate[field])
                # Truncate if too long for advanced processing
                if len(content) > self.max_length * 3:
                    content = content[:self.max_length * 3]
                return content
        
        # Fallback
        return candidate.get('title', candidate.get('id', 'No content available'))
    
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
    
    def analyze_reranking_quality(self, 
                                original_candidates: List[Dict[str, Any]], 
                                final_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality improvement from advanced reranking"""
        
        analysis = {
            "input_count": len(original_candidates),
            "output_count": len(final_results),
            "compression_ratio": len(final_results) / len(original_candidates) if original_candidates else 0,
            "score_improvements": [],
            "ranking_changes": [],
            "quality_metrics": {}
        }
        
        # Analyze score improvements
        for result in final_results:
            original_score = result.get('similarity_score', 0.0)
            final_score = result.get('integrated_score', 0.0)
            improvement = final_score - original_score
            
            analysis["score_improvements"].append({
                "doc_id": result.get('id', 'unknown'),
                "original_score": original_score,
                "final_score": final_score,
                "improvement": improvement
            })
        
        # Calculate quality metrics
        if final_results:
            final_scores = [r.get('integrated_score', 0.0) for r in final_results]
            
            analysis["quality_metrics"] = {
                "avg_final_score": np.mean(final_scores),
                "min_final_score": min(final_scores),
                "max_final_score": max(final_scores),
                "score_std": np.std(final_scores),
                "high_quality_count": sum(1 for score in final_scores if score > 0.8)
            }
        
        return analysis
    
    def explain_final_ranking(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Explain why a result was selected in final ranking"""
        
        explanation = {
            "document_id": result.get('id', 'unknown'),
            "final_position": result.get('advanced_rerank_position', 0),
            "integrated_score": result.get('integrated_score', 0.0),
            "score_breakdown": {},
            "selection_factors": []
        }
        
        # Score breakdown
        explanation["score_breakdown"] = {
            "model_score": result.get('model_score', 0.0),
            "signal_contribution": result.get('signal_contribution', 0.0),
            "advanced_cross_encoder": result.get('advanced_cross_encoder_score', 0.0),
            "llm_relevance": result.get('llm_relevance_score', 0.0)
        }
        
        # Selection factors
        integrated_score = explanation["integrated_score"]
        
        if integrated_score > 0.8:
            explanation["selection_factors"].append("High integrated relevance score")
        
        if result.get('advanced_cross_encoder_score', 0) > 0.7:
            explanation["selection_factors"].append("Strong cross-encoder relevance")
        
        if result.get('llm_relevance_score', 0) > 0.7:
            explanation["selection_factors"].append("High LLM-assessed relevance")
        
        if result.get('combined_signal_score', 0) > 0.6:
            explanation["selection_factors"].append("Strong multi-signal alignment")
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        return {
            "cross_encoder_model": self.model_name,
            "llm_model": self.llm_model if self.use_llm_scoring else None,
            "use_llm_scoring": self.use_llm_scoring,
            "device": self.device,
            "models_loaded": {
                "cross_encoder": self.cross_encoder is not None,
                "llm_scorer": self.llm_scorer is not None
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranking statistics"""
        return self.reranking_stats.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Check reranker health"""
        
        health = {
            "status": "healthy",
            "issues": []
        }
        
        if not self.cross_encoder:
            health["status"] = "warning"
            health["issues"].append("Advanced cross-encoder not loaded")
        
        if self.use_llm_scoring and not self.llm_scorer:
            health["status"] = "warning"
            health["issues"].append("LLM scorer requested but not loaded")
        
        # Test models
        try:
            if self.cross_encoder:
                test_pairs = [["test query", "test document"]]
                scores = self.cross_encoder.predict(test_pairs)
                if len(scores) != 1:
                    health["issues"].append("Cross-encoder prediction test failed")
        except Exception as e:
            health["status"] = "unhealthy"
            health["issues"].append(f"Cross-encoder test failed: {str(e)}")
        
        return health