"""
Confidence Estimator

Estimates answer confidence based on evidence quality, model certainty,
and consistency metrics for grounded answer generation.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import json

logger = logging.getLogger(__name__)

class ConfidenceEstimator:
    """Estimates confidence scores for generated answers"""
    
    def __init__(self, 
                 evidence_weight: float = 0.4,
                 citation_weight: float = 0.3,
                 consistency_weight: float = 0.2,
                 model_weight: float = 0.1):
        
        self.evidence_weight = evidence_weight
        self.citation_weight = citation_weight
        self.consistency_weight = consistency_weight
        self.model_weight = model_weight
        
        # Confidence calculation statistics
        self.confidence_stats = {
            "total_estimations": 0,
            "avg_confidence": 0.0,
            "confidence_distribution": {
                "high": 0,    # > 0.8
                "medium": 0,  # 0.5 - 0.8
                "low": 0      # < 0.5
            }
        }
        
        logger.info("Initialized confidence estimator")
    
    def estimate_confidence(self, 
                          answer: Dict[str, Any], 
                          evidence_documents: List[Dict[str, Any]], 
                          query: str,
                          model_metadata: Dict[str, Any] = None) -> float:
        """
        Estimate confidence score for generated answer
        
        Args:
            answer: Generated answer with summary, detailed_answer, references
            evidence_documents: List of evidence documents used
            query: Original query
            model_metadata: Model generation metadata
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        
        # Calculate individual confidence components
        evidence_confidence = self._calculate_evidence_confidence(evidence_documents)
        citation_confidence = self._calculate_citation_confidence(answer, evidence_documents)
        consistency_confidence = self._calculate_consistency_confidence(answer, query)
        model_confidence = self._calculate_model_confidence(model_metadata)
        
        # Weighted combination
        total_confidence = (
            evidence_confidence * self.evidence_weight +
            citation_confidence * self.citation_weight +
            consistency_confidence * self.consistency_weight +
            model_confidence * self.model_weight
        )
        
        # Ensure confidence is in valid range
        final_confidence = max(0.0, min(1.0, total_confidence))
        
        # Update statistics
        self._update_confidence_stats(final_confidence)
        
        logger.debug(f"Confidence components - Evidence: {evidence_confidence:.3f}, "
                    f"Citation: {citation_confidence:.3f}, Consistency: {consistency_confidence:.3f}, "
                    f"Model: {model_confidence:.3f}, Final: {final_confidence:.3f}")
        
        return final_confidence
    
    def _calculate_evidence_confidence(self, evidence_documents: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on evidence quality"""
        
        if not evidence_documents:
            return 0.1  # Very low confidence with no evidence
        
        # Factors for evidence confidence
        similarity_scores = [doc.get('similarity_score', 0.0) for doc in evidence_documents]
        content_lengths = [len(doc.get('content', '')) for doc in evidence_documents]
        
        # Average similarity score (primary factor)
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Evidence diversity (number of different sources)
        unique_sources = len(set(doc.get('source', 'unknown') for doc in evidence_documents))
        diversity_score = min(1.0, unique_sources / 3.0)  # Normalize to max 3 sources
        
        # Content quality (sufficient content length)
        avg_content_length = np.mean(content_lengths) if content_lengths else 0
        content_quality = min(1.0, avg_content_length / 200.0)  # Normalize to 200 chars
        
        # Top evidence quality (best evidence score)
        top_similarity = max(similarity_scores) if similarity_scores else 0.0
        
        # Combine factors
        evidence_confidence = (
            avg_similarity * 0.4 +
            top_similarity * 0.3 +
            diversity_score * 0.2 +
            content_quality * 0.1
        )
        
        return min(1.0, evidence_confidence)
    
    def _calculate_citation_confidence(self, 
                                    answer: Dict[str, Any], 
                                    evidence_documents: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on citation quality"""
        
        detailed_answer = answer.get('detailed_answer', '')
        references = answer.get('references', [])
        
        if not detailed_answer:
            return 0.0
        
        # Extract citations from answer text
        citation_pattern = r'\[([a-zA-Z0-9_-]+)\]'
        citations_in_text = re.findall(citation_pattern, detailed_answer)
        
        # Citation coverage (how many claims are cited)
        sentences = self._split_into_sentences(detailed_answer)
        cited_sentences = sum(1 for sentence in sentences if re.search(citation_pattern, sentence))
        citation_coverage = cited_sentences / len(sentences) if sentences else 0.0
        
        # Citation accuracy (citations match references)
        reference_doc_ids = set(ref.get('doc_id', '') for ref in references)
        valid_citations = sum(1 for citation in citations_in_text if citation in reference_doc_ids)
        citation_accuracy = valid_citations / len(citations_in_text) if citations_in_text else 0.0
        
        # Reference quality (average relevance score of references)
        reference_scores = [ref.get('relevance_score', 0.0) for ref in references]
        avg_reference_quality = np.mean(reference_scores) if reference_scores else 0.0
        
        # Citation density (appropriate number of citations)
        citation_density = len(citations_in_text) / len(sentences) if sentences else 0.0
        optimal_density = 0.3  # Optimal: ~30% of sentences cited
        density_score = 1.0 - abs(citation_density - optimal_density) / optimal_density
        density_score = max(0.0, density_score)
        
        # Combine citation factors
        citation_confidence = (
            citation_coverage * 0.3 +
            citation_accuracy * 0.3 +
            avg_reference_quality * 0.25 +
            density_score * 0.15
        )
        
        return min(1.0, citation_confidence)
    
    def _calculate_consistency_confidence(self, answer: Dict[str, Any], query: str) -> float:
        """Calculate confidence based on answer consistency"""
        
        summary = answer.get('summary', '')
        detailed_answer = answer.get('detailed_answer', '')
        
        if not summary or not detailed_answer:
            return 0.0
        
        # Query relevance (answer addresses the query)
        query_relevance = self._calculate_text_relevance(detailed_answer, query)
        
        # Summary-detail consistency (summary aligns with detailed answer)
        summary_consistency = self._calculate_text_relevance(summary, detailed_answer)
        
        # Answer completeness (sufficient detail)
        completeness_score = min(1.0, len(detailed_answer) / 300.0)  # Normalize to 300 chars
        
        # Answer coherence (logical flow and structure)
        coherence_score = self._calculate_coherence_score(detailed_answer)
        
        # Factual consistency (no contradictions)
        factual_consistency = self._check_factual_consistency(summary, detailed_answer)
        
        # Combine consistency factors
        consistency_confidence = (
            query_relevance * 0.3 +
            summary_consistency * 0.25 +
            completeness_score * 0.2 +
            coherence_score * 0.15 +
            factual_consistency * 0.1
        )
        
        return min(1.0, consistency_confidence)
    
    def _calculate_model_confidence(self, model_metadata: Dict[str, Any] = None) -> float:
        """Calculate confidence based on model performance metadata"""
        
        if not model_metadata:
            return 0.5  # Neutral confidence without metadata
        
        # Generation time (faster might indicate more confident generation)
        generation_time = model_metadata.get('generation_time', 5.0)
        time_score = max(0.0, 1.0 - (generation_time - 1.0) / 10.0)  # Penalize very slow generation
        
        # Model size/capability (larger models generally more confident)
        model_name = model_metadata.get('model_used', '')
        model_capability_score = self._get_model_capability_score(model_name)
        
        # Generation attempt (first attempt is more confident)
        attempt = model_metadata.get('attempt', 1)
        attempt_score = max(0.0, 1.0 - (attempt - 1) * 0.2)
        
        # Evidence utilization (model used available evidence)
        evidence_count = model_metadata.get('evidence_count', 0)
        evidence_utilization = min(1.0, evidence_count / 5.0)  # Normalize to 5 evidence docs
        
        # Combine model factors
        model_confidence = (
            time_score * 0.2 +
            model_capability_score * 0.4 +
            attempt_score * 0.2 +
            evidence_utilization * 0.2
        )
        
        return min(1.0, model_confidence)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _calculate_text_relevance(self, text1: str, text2: str) -> float:
        """Calculate relevance between two texts using word overlap"""
        
        # Simple word overlap calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate coherence score based on text structure"""
        
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < 2:
            return 0.5  # Neutral for very short text
        
        # Check for transition words/phrases
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'similarly', 'in contrast', 'on the other hand',
            'first', 'second', 'finally', 'in conclusion', 'as a result'
        }
        
        transition_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in transition_words):
                transition_count += 1
        
        # Coherence based on transitions and sentence flow
        transition_score = min(1.0, transition_count / (len(sentences) * 0.3))
        
        # Check for consistent tense and style
        consistency_score = 0.7  # Default assumption of reasonable consistency
        
        # Average sentence length (not too short or too long)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        length_score = 1.0 - abs(avg_sentence_length - 15) / 15  # Optimal ~15 words
        length_score = max(0.0, length_score)
        
        coherence = (transition_score * 0.4 + consistency_score * 0.3 + length_score * 0.3)
        
        return min(1.0, coherence)
    
    def _check_factual_consistency(self, summary: str, detailed_answer: str) -> float:
        """Check for factual consistency between summary and detailed answer"""
        
        # Simple consistency check based on key terms
        summary_terms = set(summary.lower().split())
        detail_terms = set(detailed_answer.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        summary_terms -= stop_words
        detail_terms -= stop_words
        
        # Check for contradictory terms (simple heuristic)
        contradictory_pairs = [
            ('yes', 'no'), ('true', 'false'), ('correct', 'incorrect'),
            ('increase', 'decrease'), ('more', 'less'), ('higher', 'lower')
        ]
        
        contradiction_penalty = 0.0
        for term1, term2 in contradictory_pairs:
            if term1 in summary_terms and term2 in detail_terms:
                contradiction_penalty += 0.2
            elif term2 in summary_terms and term1 in detail_terms:
                contradiction_penalty += 0.2
        
        # Base consistency score
        base_consistency = 0.8
        
        return max(0.0, base_consistency - contradiction_penalty)
    
    def _get_model_capability_score(self, model_name: str) -> float:
        """Get capability score based on model name"""
        
        model_scores = {
            'gpt-4': 0.95,
            'gpt-3.5': 0.85,
            'DialoGPT-large': 0.75,
            'DialoGPT-medium': 0.65,
            'DialoGPT-small': 0.55,
            'blenderbot': 0.70,
            'flan-t5-large': 0.80,
            'flan-t5-base': 0.70,
            'gpt-neo-2.7B': 0.75,
            'gpt-neo-1.3B': 0.65,
            'GODEL': 0.70
        }
        
        # Find matching model
        for model_key, score in model_scores.items():
            if model_key.lower() in model_name.lower():
                return score
        
        # Default score for unknown models
        return 0.6
    
    def _update_confidence_stats(self, confidence: float):
        """Update confidence statistics"""
        
        self.confidence_stats["total_estimations"] += 1
        
        # Update average confidence
        total_confidence = (self.confidence_stats["avg_confidence"] * 
                          (self.confidence_stats["total_estimations"] - 1) + confidence)
        self.confidence_stats["avg_confidence"] = total_confidence / self.confidence_stats["total_estimations"]
        
        # Update distribution
        if confidence > 0.8:
            self.confidence_stats["confidence_distribution"]["high"] += 1
        elif confidence > 0.5:
            self.confidence_stats["confidence_distribution"]["medium"] += 1
        else:
            self.confidence_stats["confidence_distribution"]["low"] += 1
    
    def estimate_batch_confidence(self, 
                                answers: List[Dict[str, Any]], 
                                evidence_batches: List[List[Dict[str, Any]]], 
                                queries: List[str],
                                model_metadata_batch: List[Dict[str, Any]] = None) -> List[float]:
        """Estimate confidence for batch of answers"""
        
        if not model_metadata_batch:
            model_metadata_batch = [None] * len(answers)
        
        confidences = []
        
        for answer, evidence_docs, query, metadata in zip(answers, evidence_batches, queries, model_metadata_batch):
            confidence = self.estimate_confidence(answer, evidence_docs, query, metadata)
            confidences.append(confidence)
        
        return confidences
    
    def calibrate_confidence(self, 
                           predicted_confidences: List[float], 
                           actual_accuracies: List[float]) -> Dict[str, float]:
        """Calibrate confidence estimator based on actual performance"""
        
        if len(predicted_confidences) != len(actual_accuracies):
            raise ValueError("Predicted confidences and actual accuracies must have same length")
        
        # Calculate calibration metrics
        predicted = np.array(predicted_confidences)
        actual = np.array(actual_accuracies)
        
        # Mean absolute error
        mae = np.mean(np.abs(predicted - actual))
        
        # Root mean square error
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        
        # Correlation
        correlation = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else 0.0
        
        # Calibration curve (binned analysis)
        calibration_error = self._calculate_calibration_error(predicted, actual)
        
        calibration_metrics = {
            "mean_absolute_error": mae,
            "root_mean_square_error": rmse,
            "correlation": correlation,
            "calibration_error": calibration_error
        }
        
        logger.info(f"Confidence calibration metrics: {calibration_metrics}")
        
        return calibration_metrics
    
    def _calculate_calibration_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calculate expected calibration error"""
        
        # Bin predictions
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        calibration_error = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = (predicted > bin_lower) & (predicted <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Average confidence and accuracy in bin
                avg_confidence = np.mean(predicted[in_bin])
                avg_accuracy = np.mean(actual[in_bin])
                
                # Weighted calibration error
                bin_weight = np.sum(in_bin) / len(predicted)
                calibration_error += bin_weight * abs(avg_confidence - avg_accuracy)
        
        return calibration_error
    
    def update_weights(self, 
                      evidence_weight: float = None,
                      citation_weight: float = None,
                      consistency_weight: float = None,
                      model_weight: float = None):
        """Update confidence estimation weights"""
        
        if evidence_weight is not None:
            self.evidence_weight = evidence_weight
        if citation_weight is not None:
            self.citation_weight = citation_weight
        if consistency_weight is not None:
            self.consistency_weight = consistency_weight
        if model_weight is not None:
            self.model_weight = model_weight
        
        # Normalize weights to sum to 1.0
        total_weight = (self.evidence_weight + self.citation_weight + 
                       self.consistency_weight + self.model_weight)
        
        if total_weight > 0:
            self.evidence_weight /= total_weight
            self.citation_weight /= total_weight
            self.consistency_weight /= total_weight
            self.model_weight /= total_weight
        
        logger.info(f"Updated confidence weights - Evidence: {self.evidence_weight:.3f}, "
                   f"Citation: {self.citation_weight:.3f}, Consistency: {self.consistency_weight:.3f}, "
                   f"Model: {self.model_weight:.3f}")
    
    def get_confidence_stats(self) -> Dict[str, Any]:
        """Get confidence estimation statistics"""
        
        stats = self.confidence_stats.copy()
        
        # Calculate distribution percentages
        total = stats["total_estimations"]
        if total > 0:
            for category in stats["confidence_distribution"]:
                stats["confidence_distribution"][category + "_percentage"] = (
                    stats["confidence_distribution"][category] / total * 100
                )
        
        return stats
    
    def clear_stats(self):
        """Clear confidence estimation statistics"""
        
        self.confidence_stats = {
            "total_estimations": 0,
            "avg_confidence": 0.0,
            "confidence_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        logger.info("Cleared confidence estimation statistics")