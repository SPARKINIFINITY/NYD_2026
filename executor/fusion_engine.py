"""
Fusion Engine - Result Combination and Ranking

Implements various fusion algorithms to combine results from multiple retrievers
including Reciprocal Rank Fusion (RRF) and weighted sum approaches.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from collections import defaultdict, Counter
import math

logger = logging.getLogger(__name__)

class FusionEngine:
    """Combines and ranks results from multiple retrievers"""
    
    def __init__(self, 
                 default_method: str = "rrf",
                 rrf_k: int = 60,
                 score_normalization: str = "min_max"):
        
        self.default_method = default_method
        self.rrf_k = rrf_k
        self.score_normalization = score_normalization
        
        # Available fusion methods
        self.fusion_methods = {
            'rrf': self._reciprocal_rank_fusion,
            'weighted_sum': self._weighted_sum_fusion,
            'borda_count': self._borda_count_fusion,
            'combsum': self._combsum_fusion,
            'combmnz': self._combmnz_fusion,
            'max_score': self._max_score_fusion,
            'min_score': self._min_score_fusion
        }
        
        # Performance tracking
        self.fusion_stats = {
            "total_fusions": 0,
            "method_usage": defaultdict(int),
            "average_time": 0.0,
            "total_time": 0.0
        }
        
        logger.info(f"Initialized fusion engine with default method: {default_method}")
    
    def fuse_results(self, 
                    retriever_results: Dict[str, List[Dict[str, Any]]], 
                    weights: Dict[str, float] = None,
                    method: str = None,
                    k: int = 10) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple retrievers
        
        Args:
            retriever_results: Dict mapping retriever names to their results
            weights: Optional weights for each retriever
            method: Fusion method to use (overrides default)
            k: Number of final results to return
        
        Returns:
            List of fused and ranked results
        """
        start_time = time.time()
        
        try:
            method = method or self.default_method
            
            if method not in self.fusion_methods:
                logger.error(f"Unknown fusion method: {method}")
                return []
            
            if not retriever_results:
                logger.warning("No retriever results provided for fusion")
                return []
            
            # Normalize scores if needed
            normalized_results = self._normalize_scores(retriever_results)
            
            # Apply fusion method
            fusion_func = self.fusion_methods[method]
            
            if method == "weighted_sum" and weights:
                fused_results = fusion_func(normalized_results, weights, k)
            else:
                fused_results = fusion_func(normalized_results, k)
            
            # Add fusion metadata
            for i, result in enumerate(fused_results):
                result.update({
                    'fusion_rank': i + 1,
                    'fusion_method': method,
                    'fusion_timestamp': time.time(),
                    'contributing_retrievers': self._get_contributing_retrievers(result, retriever_results)
                })
            
            # Update statistics
            fusion_time = time.time() - start_time
            self.fusion_stats["total_fusions"] += 1
            self.fusion_stats["method_usage"][method] += 1
            self.fusion_stats["total_time"] += fusion_time
            self.fusion_stats["average_time"] = (
                self.fusion_stats["total_time"] / self.fusion_stats["total_fusions"]
            )
            
            logger.debug(f"Fusion completed in {fusion_time:.3f}s using {method}, returned {len(fused_results)} results")
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            return []
    
    def _normalize_scores(self, retriever_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize scores across retrievers"""
        if self.score_normalization == "none":
            return retriever_results
        
        normalized_results = {}
        
        for retriever_name, results in retriever_results.items():
            if not results:
                normalized_results[retriever_name] = results
                continue
            
            # Extract scores
            scores = [r.get('similarity_score', 0.0) for r in results]
            
            if not scores or all(s == 0 for s in scores):
                normalized_results[retriever_name] = results
                continue
            
            # Apply normalization
            if self.score_normalization == "min_max":
                min_score, max_score = min(scores), max(scores)
                if max_score > min_score:
                    normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
                else:
                    normalized_scores = [1.0] * len(scores)
            
            elif self.score_normalization == "z_score":
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                if std_score > 0:
                    normalized_scores = [(s - mean_score) / std_score for s in scores]
                else:
                    normalized_scores = [0.0] * len(scores)
            
            elif self.score_normalization == "softmax":
                exp_scores = np.exp(np.array(scores) - np.max(scores))  # Numerical stability
                normalized_scores = exp_scores / np.sum(exp_scores)
                normalized_scores = normalized_scores.tolist()
            
            else:
                normalized_scores = scores
            
            # Update results with normalized scores
            normalized_result_list = []
            for result, norm_score in zip(results, normalized_scores):
                norm_result = result.copy()
                norm_result['normalized_score'] = norm_score
                norm_result['original_score'] = result.get('similarity_score', 0.0)
                normalized_result_list.append(norm_result)
            
            normalized_results[retriever_name] = normalized_result_list
        
        return normalized_results
    
    def _reciprocal_rank_fusion(self, retriever_results: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF)"""
        document_scores = defaultdict(float)
        document_data = {}
        
        for retriever_name, results in retriever_results.items():
            for rank, result in enumerate(results, 1):
                doc_id = self._get_document_id(result)
                
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (self.rrf_k + rank)
                document_scores[doc_id] += rrf_score
                
                # Store document data (use first occurrence)
                if doc_id not in document_data:
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['rrf_contributions'] = []
                
                document_data[doc_id]['rrf_contributions'].append({
                    'retriever': retriever_name,
                    'rank': rank,
                    'rrf_score': rrf_score,
                    'original_score': result.get('similarity_score', 0.0)
                })
        
        # Sort by RRF score and return top-k
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            result = document_data[doc_id].copy()
            result.update({
                'similarity_score': rrf_score,
                'fusion_score': rrf_score,
                'fusion_method_details': {
                    'rrf_k': self.rrf_k,
                    'contributions': result['rrf_contributions']
                }
            })
            fused_results.append(result)
        
        return fused_results
    
    def _weighted_sum_fusion(self, retriever_results: Dict[str, List[Dict[str, Any]]], 
                           weights: Dict[str, float], k: int) -> List[Dict[str, Any]]:
        """Weighted sum fusion"""
        document_scores = defaultdict(float)
        document_data = {}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {name: w / total_weight for name, w in weights.items()}
        else:
            normalized_weights = {name: 1.0 / len(weights) for name in weights}
        
        for retriever_name, results in retriever_results.items():
            weight = normalized_weights.get(retriever_name, 0.0)
            
            for result in results:
                doc_id = self._get_document_id(result)
                score = result.get('normalized_score', result.get('similarity_score', 0.0))
                
                weighted_score = score * weight
                document_scores[doc_id] += weighted_score
                
                if doc_id not in document_data:
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['weighted_contributions'] = []
                
                document_data[doc_id]['weighted_contributions'].append({
                    'retriever': retriever_name,
                    'weight': weight,
                    'score': score,
                    'weighted_score': weighted_score
                })
        
        # Sort and return top-k
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for doc_id, final_score in sorted_docs:
            result = document_data[doc_id].copy()
            result.update({
                'similarity_score': final_score,
                'fusion_score': final_score,
                'fusion_method_details': {
                    'weights_used': normalized_weights,
                    'contributions': result['weighted_contributions']
                }
            })
            fused_results.append(result)
        
        return fused_results
    
    def _borda_count_fusion(self, retriever_results: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """Borda count fusion"""
        document_scores = defaultdict(int)
        document_data = {}
        
        for retriever_name, results in retriever_results.items():
            n_results = len(results)
            
            for rank, result in enumerate(results):
                doc_id = self._get_document_id(result)
                
                # Borda count: n - rank (higher rank gets higher score)
                borda_score = n_results - rank
                document_scores[doc_id] += borda_score
                
                if doc_id not in document_data:
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['borda_contributions'] = []
                
                document_data[doc_id]['borda_contributions'].append({
                    'retriever': retriever_name,
                    'rank': rank + 1,
                    'borda_score': borda_score
                })
        
        # Sort and return top-k
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for doc_id, borda_score in sorted_docs:
            result = document_data[doc_id].copy()
            result.update({
                'similarity_score': float(borda_score),
                'fusion_score': float(borda_score),
                'fusion_method_details': {
                    'contributions': result['borda_contributions']
                }
            })
            fused_results.append(result)
        
        return fused_results
    
    def _combsum_fusion(self, retriever_results: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """CombSUM fusion - sum of normalized scores"""
        document_scores = defaultdict(float)
        document_data = {}
        
        for retriever_name, results in retriever_results.items():
            for result in results:
                doc_id = self._get_document_id(result)
                score = result.get('normalized_score', result.get('similarity_score', 0.0))
                
                document_scores[doc_id] += score
                
                if doc_id not in document_data:
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['combsum_contributions'] = []
                
                document_data[doc_id]['combsum_contributions'].append({
                    'retriever': retriever_name,
                    'score': score
                })
        
        # Sort and return top-k
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for doc_id, sum_score in sorted_docs:
            result = document_data[doc_id].copy()
            result.update({
                'similarity_score': sum_score,
                'fusion_score': sum_score,
                'fusion_method_details': {
                    'contributions': result['combsum_contributions']
                }
            })
            fused_results.append(result)
        
        return fused_results
    
    def _combmnz_fusion(self, retriever_results: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """CombMNZ fusion - sum of scores multiplied by number of non-zero retrievers"""
        document_scores = defaultdict(float)
        document_counts = defaultdict(int)
        document_data = {}
        
        for retriever_name, results in retriever_results.items():
            for result in results:
                doc_id = self._get_document_id(result)
                score = result.get('normalized_score', result.get('similarity_score', 0.0))
                
                if score > 0:
                    document_scores[doc_id] += score
                    document_counts[doc_id] += 1
                
                if doc_id not in document_data:
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['combmnz_contributions'] = []
                
                document_data[doc_id]['combmnz_contributions'].append({
                    'retriever': retriever_name,
                    'score': score
                })
        
        # Apply MNZ multiplication
        for doc_id in document_scores:
            document_scores[doc_id] *= document_counts[doc_id]
        
        # Sort and return top-k
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for doc_id, mnz_score in sorted_docs:
            result = document_data[doc_id].copy()
            result.update({
                'similarity_score': mnz_score,
                'fusion_score': mnz_score,
                'fusion_method_details': {
                    'non_zero_count': document_counts[doc_id],
                    'contributions': result['combmnz_contributions']
                }
            })
            fused_results.append(result)
        
        return fused_results
    
    def _max_score_fusion(self, retriever_results: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """Max score fusion - take maximum score across retrievers"""
        document_scores = defaultdict(float)
        document_data = {}
        
        for retriever_name, results in retriever_results.items():
            for result in results:
                doc_id = self._get_document_id(result)
                score = result.get('normalized_score', result.get('similarity_score', 0.0))
                
                if score > document_scores[doc_id]:
                    document_scores[doc_id] = score
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['max_score_source'] = retriever_name
        
        # Sort and return top-k
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for doc_id, max_score in sorted_docs:
            result = document_data[doc_id].copy()
            result.update({
                'similarity_score': max_score,
                'fusion_score': max_score,
                'fusion_method_details': {
                    'max_score_source': result['max_score_source']
                }
            })
            fused_results.append(result)
        
        return fused_results
    
    def _min_score_fusion(self, retriever_results: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """Min score fusion - take minimum score across retrievers (conservative)"""
        document_scores = defaultdict(lambda: float('inf'))
        document_data = {}
        
        for retriever_name, results in retriever_results.items():
            for result in results:
                doc_id = self._get_document_id(result)
                score = result.get('normalized_score', result.get('similarity_score', 0.0))
                
                if score < document_scores[doc_id]:
                    document_scores[doc_id] = score
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['min_score_source'] = retriever_name
        
        # Filter out infinite scores and sort
        valid_docs = [(doc_id, score) for doc_id, score in document_scores.items() if score != float('inf')]
        sorted_docs = sorted(valid_docs, key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for doc_id, min_score in sorted_docs:
            result = document_data[doc_id].copy()
            result.update({
                'similarity_score': min_score,
                'fusion_score': min_score,
                'fusion_method_details': {
                    'min_score_source': result['min_score_source']
                }
            })
            fused_results.append(result)
        
        return fused_results
    
    def _get_document_id(self, result: Dict[str, Any]) -> str:
        """Extract document ID from result"""
        return result.get('id') or result.get('doc_id') or str(hash(result.get('content', '')[:100]))
    
    def _get_contributing_retrievers(self, result: Dict[str, Any], 
                                   retriever_results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Get list of retrievers that contributed to this result"""
        doc_id = self._get_document_id(result)
        contributing = []
        
        for retriever_name, results in retriever_results.items():
            for r in results:
                if self._get_document_id(r) == doc_id:
                    contributing.append(retriever_name)
                    break
        
        return contributing
    
    def analyze_fusion_quality(self, fused_results: List[Dict[str, Any]], 
                             retriever_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze the quality of fusion results"""
        analysis = {
            'total_fused_results': len(fused_results),
            'retriever_coverage': {},
            'score_distribution': {},
            'rank_correlation': {},
            'diversity_metrics': {}
        }
        
        # Retriever coverage analysis
        for retriever_name in retriever_results.keys():
            coverage = sum(1 for result in fused_results 
                          if retriever_name in result.get('contributing_retrievers', []))
            analysis['retriever_coverage'][retriever_name] = {
                'count': coverage,
                'percentage': coverage / len(fused_results) if fused_results else 0
            }
        
        # Score distribution
        if fused_results:
            scores = [r.get('fusion_score', 0) for r in fused_results]
            analysis['score_distribution'] = {
                'min': min(scores),
                'max': max(scores),
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores)
            }
        
        # Diversity metrics
        unique_sources = set()
        for result in fused_results:
            unique_sources.update(result.get('contributing_retrievers', []))
        
        analysis['diversity_metrics'] = {
            'unique_retriever_sources': len(unique_sources),
            'avg_retrievers_per_result': (
                sum(len(r.get('contributing_retrievers', [])) for r in fused_results) / 
                len(fused_results) if fused_results else 0
            )
        }
        
        return analysis
    
    def get_fusion_explanation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Explain how a specific result was fused"""
        explanation = {
            'document_id': self._get_document_id(result),
            'final_score': result.get('fusion_score', result.get('similarity_score', 0)),
            'fusion_method': result.get('fusion_method', 'unknown'),
            'contributing_retrievers': result.get('contributing_retrievers', []),
            'method_details': result.get('fusion_method_details', {})
        }
        
        return explanation
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fusion statistics"""
        stats = self.fusion_stats.copy()
        stats.update({
            'default_method': self.default_method,
            'rrf_k': self.rrf_k,
            'score_normalization': self.score_normalization,
            'available_methods': list(self.fusion_methods.keys())
        })
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check fusion engine health"""
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        # Check if default method is valid
        if self.default_method not in self.fusion_methods:
            health['status'] = 'unhealthy'
            health['issues'].append(f'Invalid default method: {self.default_method}')
        
        return health