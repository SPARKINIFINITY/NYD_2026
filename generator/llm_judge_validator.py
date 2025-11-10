"""
LLM-as-Judge Validator with Natural Language Inference (NLI)

This component validates generated answers against evidence using:
1. Claim extraction from generated answers
2. NLI-based entailment checking (DeBERTa-large-mnli)
3. Per-claim and overall verdict assignment
4. Orchestrator feedback for answer refinement
"""

import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, AutoModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class EntailmentLabel(Enum):
    """NLI entailment labels"""
    ENTAILED = "entailed"
    NEUTRAL = "neutral" 
    CONTRADICTED = "contradicted"

class OverallVerdict(Enum):
    """Overall validation verdicts"""
    SATISFACTORY = "satisfactory"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_REGENERATION = "needs_regeneration"
    NEEDS_EXPANDED_RETRIEVAL = "needs_expanded_retrieval"

@dataclass
class Claim:
    """Individual claim extracted from answer"""
    text: str
    claim_id: str
    sentence_index: int
    confidence: float = 0.0
    is_factual: bool = True

@dataclass
class ClaimValidation:
    """Validation result for individual claim"""
    claim: Claim
    entailment_label: EntailmentLabel
    entailment_score: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    evidence_spans: List[Dict[str, Any]]

@dataclass
class ValidationResult:
    """Complete validation result"""
    overall_verdict: OverallVerdict
    overall_confidence: float
    claim_validations: List[ClaimValidation]
    unsupported_claims: List[Claim]
    contradicted_claims: List[Claim]
    evidence_coverage: float
    recommendations: List[str]
    metadata: Dict[str, Any]

class LLMJudgeValidator:
    """LLM-as-Judge validator with NLI capabilities"""
    
    def __init__(self,
                 nli_model_name: str = "microsoft/deberta-large-mnli",
                 claim_extractor_model: str = "microsoft/DialoGPT-small",
                 device: str = "auto",
                 batch_size: int = 8,
                 confidence_threshold: float = 0.7,
                 entailment_threshold: float = 0.5):
        
        self.nli_model_name = nli_model_name
        self.claim_extractor_model = claim_extractor_model
        self.device = self._get_device(device)
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.entailment_threshold = entailment_threshold
        
        # Initialize models
        self.nli_pipeline = None
        self.claim_extractor = None
        self.tokenizer = None
        
        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "satisfactory_answers": 0,
            "needs_clarification": 0,
            "needs_regeneration": 0,
            "needs_expanded_retrieval": 0,
            "avg_validation_time": 0.0,
            "avg_claims_per_answer": 0.0,
            "avg_entailment_score": 0.0
        }
        
        # Initialize components
        self._initialize_models()
        
        logger.info(f"Initialized LLMJudgeValidator with NLI model: {nli_model_name}")
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device"""
        if device == "auto":
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_models(self):
        """Initialize NLI and claim extraction models"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Initialize NLI pipeline
                self.nli_pipeline = pipeline(
                    "text-classification",
                    model=self.nli_model_name,
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                
                # Initialize tokenizer for text processing
                self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
                
                logger.info("Successfully initialized transformer models")
            else:
                logger.warning("Transformers not available - using mock validation")
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.nli_pipeline = None
            self.tokenizer = None
    
    def validate_answer(self,
                       generated_answer: Dict[str, Any],
                       evidence_documents: List[Dict[str, Any]],
                       query: str,
                       intent: str = "fact") -> ValidationResult:
        """
        Validate generated answer against evidence using NLI
        
        Args:
            generated_answer: Generated answer with summary, detailed_answer, etc.
            evidence_documents: Retrieved evidence documents
            query: Original user query
            intent: Query intent
            
        Returns:
            ValidationResult with per-claim and overall validation
        """
        
        start_time = time.time()
        self.validation_stats["total_validations"] += 1
        
        try:
            # Step 1: Extract claims from generated answer
            claims = self._extract_claims(generated_answer, query)
            
            # Step 2: Validate each claim against evidence
            claim_validations = self._validate_claims_with_nli(claims, evidence_documents)
            
            # Step 3: Determine overall verdict
            overall_verdict, overall_confidence = self._determine_overall_verdict(
                claim_validations, generated_answer, evidence_documents
            )
            
            # Step 4: Analyze unsupported and contradicted claims
            unsupported_claims = [cv.claim for cv in claim_validations 
                                if cv.entailment_label == EntailmentLabel.NEUTRAL]
            contradicted_claims = [cv.claim for cv in claim_validations 
                                 if cv.entailment_label == EntailmentLabel.CONTRADICTED]
            
            # Step 5: Calculate evidence coverage
            evidence_coverage = self._calculate_evidence_coverage(claim_validations, evidence_documents)
            
            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(
                overall_verdict, claim_validations, unsupported_claims, contradicted_claims
            )
            
            # Create validation result
            validation_result = ValidationResult(
                overall_verdict=overall_verdict,
                overall_confidence=overall_confidence,
                claim_validations=claim_validations,
                unsupported_claims=unsupported_claims,
                contradicted_claims=contradicted_claims,
                evidence_coverage=evidence_coverage,
                recommendations=recommendations,
                metadata={
                    "validation_time": time.time() - start_time,
                    "total_claims": len(claims),
                    "entailed_claims": len([cv for cv in claim_validations 
                                          if cv.entailment_label == EntailmentLabel.ENTAILED]),
                    "neutral_claims": len(unsupported_claims),
                    "contradicted_claims": len(contradicted_claims),
                    "nli_model": self.nli_model_name,
                    "query": query,
                    "intent": intent
                }
            )
            
            # Update statistics
            self._update_validation_stats(validation_result, time.time() - start_time)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._create_error_validation_result(str(e), generated_answer, evidence_documents)
    
    def _extract_claims(self, generated_answer: Dict[str, Any], query: str) -> List[Claim]:
        """Extract individual claims from generated answer"""
        
        detailed_answer = generated_answer.get("detailed_answer", "")
        summary = generated_answer.get("summary", "")
        
        # Ensure we have strings, not lists or other types
        if isinstance(detailed_answer, list):
            detailed_answer = " ".join(str(item) for item in detailed_answer)
        elif not isinstance(detailed_answer, str):
            detailed_answer = str(detailed_answer)
            
        if isinstance(summary, list):
            summary = " ".join(str(item) for item in summary)
        elif not isinstance(summary, str):
            summary = str(summary)
        
        # Combine text sources
        full_text = f"{summary} {detailed_answer}".strip()
        
        if not full_text:
            return []
        
        claims = []
        
        try:
            if TRANSFORMERS_AVAILABLE and self.tokenizer:
                # Use advanced claim extraction
                claims = self._extract_claims_advanced(full_text, query)
            else:
                # Use simple sentence-based extraction
                claims = self._extract_claims_simple(full_text)
                
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            # Fallback to simple extraction
            claims = self._extract_claims_simple(full_text)
        
        return claims
    
    def _extract_claims_simple(self, text: str) -> List[Claim]:
        """Simple sentence-based claim extraction"""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        claims = []
        for i, sentence in enumerate(sentences):
            if len(sentence) > 10:  # Filter very short sentences
                claim = Claim(
                    text=sentence,
                    claim_id=f"claim_{i+1}",
                    sentence_index=i,
                    confidence=0.8,  # Default confidence
                    is_factual=self._is_factual_sentence(sentence)
                )
                claims.append(claim)
        
        return claims
    
    def _extract_claims_advanced(self, text: str, query: str) -> List[Claim]:
        """Advanced claim extraction using LLM"""
        
        # For now, use simple extraction but with better factual detection
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        claims = []
        for i, sentence in enumerate(sentences):
            if len(sentence) > 10:
                # Better factual detection
                is_factual = self._is_factual_sentence_advanced(sentence, query)
                confidence = self._estimate_claim_confidence(sentence, query)
                
                claim = Claim(
                    text=sentence,
                    claim_id=f"claim_{i+1}",
                    sentence_index=i,
                    confidence=confidence,
                    is_factual=is_factual
                )
                claims.append(claim)
        
        return claims
    
    def _is_factual_sentence(self, sentence: str) -> bool:
        """Simple factual sentence detection"""
        
        # Patterns that indicate factual claims
        factual_patterns = [
            r'\b(is|are|was|were|has|have|contains|includes)\b',
            r'\b(according to|based on|research shows|studies indicate)\b',
            r'\b\d+\b',  # Numbers often indicate facts
            r'\b(percent|percentage|ratio|rate)\b'
        ]
        
        # Patterns that indicate non-factual content
        non_factual_patterns = [
            r'\b(I think|I believe|perhaps|maybe|possibly|might)\b',
            r'\b(opinion|suggest|recommend|should|could)\b',
            r'\?$'  # Questions
        ]
        
        sentence_lower = sentence.lower()
        
        # Check for non-factual patterns first
        for pattern in non_factual_patterns:
            if re.search(pattern, sentence_lower):
                return False
        
        # Check for factual patterns
        for pattern in factual_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        # Default to factual if no clear indicators
        return True
    
    def _is_factual_sentence_advanced(self, sentence: str, query: str) -> bool:
        """Advanced factual sentence detection"""
        
        # Use simple detection for now, can be enhanced with ML models
        return self._is_factual_sentence(sentence)
    
    def _estimate_claim_confidence(self, sentence: str, query: str) -> float:
        """Estimate confidence in claim extraction"""
        
        # Simple heuristics for claim confidence
        confidence = 0.7  # Base confidence
        
        # Boost confidence for specific patterns
        if re.search(r'\b(according to|research shows|studies indicate)\b', sentence.lower()):
            confidence += 0.2
        
        if re.search(r'\b\d+\b', sentence):  # Contains numbers
            confidence += 0.1
        
        # Reduce confidence for uncertain language
        if re.search(r'\b(might|could|possibly|perhaps)\b', sentence.lower()):
            confidence -= 0.2
        
        return max(0.1, min(0.95, confidence))
    
    def _validate_claims_with_nli(self, 
                                claims: List[Claim], 
                                evidence_documents: List[Dict[str, Any]]) -> List[ClaimValidation]:
        """Validate claims against evidence using NLI"""
        
        claim_validations = []
        
        for claim in claims:
            if not claim.is_factual:
                # Skip non-factual claims
                validation = ClaimValidation(
                    claim=claim,
                    entailment_label=EntailmentLabel.NEUTRAL,
                    entailment_score=0.5,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    evidence_spans=[]
                )
                claim_validations.append(validation)
                continue
            
            # Validate against evidence
            validation = self._validate_single_claim(claim, evidence_documents)
            claim_validations.append(validation)
        
        return claim_validations
    
    def _validate_single_claim(self, 
                             claim: Claim, 
                             evidence_documents: List[Dict[str, Any]]) -> ClaimValidation:
        """Validate single claim against evidence using NLI"""
        
        supporting_evidence = []
        contradicting_evidence = []
        evidence_spans = []
        entailment_scores = []
        
        try:
            if TRANSFORMERS_AVAILABLE and self.nli_pipeline:
                # Use actual NLI model
                for doc in evidence_documents:
                    evidence_text = doc.get('content', '')
                    if not evidence_text:
                        continue
                    
                    # Run NLI
                    nli_result = self._run_nli(claim.text, evidence_text)
                    
                    entailment_label = nli_result['label']
                    entailment_score = nli_result['score']
                    entailment_scores.append(entailment_score)
                    
                    # Categorize evidence based on entailment
                    if entailment_label == EntailmentLabel.ENTAILED and entailment_score > self.entailment_threshold:
                        supporting_evidence.append(evidence_text)
                        evidence_spans.append({
                            'doc_id': doc.get('id', ''),
                            'content': evidence_text,
                            'entailment_score': entailment_score,
                            'label': 'supporting'
                        })
                    elif entailment_label == EntailmentLabel.CONTRADICTED and entailment_score > self.entailment_threshold:
                        contradicting_evidence.append(evidence_text)
                        evidence_spans.append({
                            'doc_id': doc.get('id', ''),
                            'content': evidence_text,
                            'entailment_score': entailment_score,
                            'label': 'contradicting'
                        })
            else:
                # Use mock NLI validation
                mock_validation = self._mock_nli_validation(claim, evidence_documents)
                supporting_evidence = mock_validation['supporting']
                contradicting_evidence = mock_validation['contradicting']
                evidence_spans = mock_validation['spans']
                entailment_scores = mock_validation['scores']
        
        except Exception as e:
            logger.error(f"NLI validation failed for claim '{claim.text}': {e}")
            # Return neutral validation on error
            entailment_scores = [0.5]
        
        # Determine overall entailment for this claim
        if entailment_scores:
            avg_score = sum(entailment_scores) / len(entailment_scores)
            
            if supporting_evidence and not contradicting_evidence:
                final_label = EntailmentLabel.ENTAILED
                final_score = max(entailment_scores)
            elif contradicting_evidence:
                final_label = EntailmentLabel.CONTRADICTED
                final_score = max(entailment_scores)
            else:
                final_label = EntailmentLabel.NEUTRAL
                final_score = avg_score
        else:
            final_label = EntailmentLabel.NEUTRAL
            final_score = 0.5
        
        return ClaimValidation(
            claim=claim,
            entailment_label=final_label,
            entailment_score=final_score,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            evidence_spans=evidence_spans
        )
    
    def _run_nli(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Run NLI model on premise-hypothesis pair"""
        
        try:
            # Format input for NLI model
            nli_input = f"{premise} [SEP] {hypothesis}"
            
            # Run NLI pipeline
            results = self.nli_pipeline(nli_input)
            
            # Parse results (DeBERTa-mnli format)
            label_mapping = {
                'ENTAILMENT': EntailmentLabel.ENTAILED,
                'NEUTRAL': EntailmentLabel.NEUTRAL,
                'CONTRADICTION': EntailmentLabel.CONTRADICTED
            }
            
            # Handle different result formats
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict):
                    # Results is a list of dictionaries
                    best_result = max(results, key=lambda x: x['score'])
                else:
                    # Results is a list of other objects, use first one
                    best_result = results[0]
                    if hasattr(best_result, 'label') and hasattr(best_result, 'score'):
                        best_result = {'label': best_result.label, 'score': best_result.score}
                    else:
                        # Fallback
                        best_result = {'label': 'NEUTRAL', 'score': 0.5}
            else:
                # Single result or unexpected format
                if isinstance(results, dict):
                    best_result = results
                else:
                    best_result = {'label': 'NEUTRAL', 'score': 0.5}
            
            return {
                'label': label_mapping.get(best_result['label'], EntailmentLabel.NEUTRAL),
                'score': best_result['score'],
                'all_scores': results
            }
            
        except Exception as e:
            logger.error(f"NLI execution failed: {e}")
            return {
                'label': EntailmentLabel.NEUTRAL,
                'score': 0.5,
                'all_scores': []
            }
    
    def _mock_nli_validation(self, 
                           claim: Claim, 
                           evidence_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock NLI validation for testing"""
        
        supporting = []
        contradicting = []
        spans = []
        scores = []
        
        claim_words = set(claim.text.lower().split())
        
        for doc in evidence_documents:
            evidence_text = doc.get('content', '')
            evidence_words = set(evidence_text.lower().split())
            
            # Simple word overlap scoring
            overlap = len(claim_words.intersection(evidence_words))
            total_words = len(claim_words.union(evidence_words))
            
            if total_words > 0:
                overlap_score = overlap / len(claim_words)
                
                if overlap_score > 0.3:  # High overlap = supporting
                    supporting.append(evidence_text)
                    spans.append({
                        'doc_id': doc.get('id', ''),
                        'content': evidence_text,
                        'entailment_score': overlap_score,
                        'label': 'supporting'
                    })
                    scores.append(overlap_score)
                elif overlap_score < 0.1 and len(evidence_words) > 10:  # Low overlap = potentially contradicting
                    contradicting.append(evidence_text)
                    spans.append({
                        'doc_id': doc.get('id', ''),
                        'content': evidence_text,
                        'entailment_score': 0.7,  # Mock contradiction score
                        'label': 'contradicting'
                    })
                    scores.append(0.7)
                else:
                    scores.append(0.5)  # Neutral
        
        return {
            'supporting': supporting,
            'contradicting': contradicting,
            'spans': spans,
            'scores': scores if scores else [0.5]
        }
    
    def _determine_overall_verdict(self,
                                 claim_validations: List[ClaimValidation],
                                 generated_answer: Dict[str, Any],
                                 evidence_documents: List[Dict[str, Any]]) -> Tuple[OverallVerdict, float]:
        """Determine overall validation verdict"""
        
        if not claim_validations:
            return OverallVerdict.NEEDS_REGENERATION, 0.1
        
        # Count claim types
        entailed_count = sum(1 for cv in claim_validations 
                           if cv.entailment_label == EntailmentLabel.ENTAILED)
        neutral_count = sum(1 for cv in claim_validations 
                          if cv.entailment_label == EntailmentLabel.NEUTRAL)
        contradicted_count = sum(1 for cv in claim_validations 
                               if cv.entailment_label == EntailmentLabel.CONTRADICTED)
        
        total_claims = len(claim_validations)
        
        # Calculate ratios
        entailed_ratio = entailed_count / total_claims
        contradicted_ratio = contradicted_count / total_claims
        neutral_ratio = neutral_count / total_claims
        
        # Calculate overall confidence
        avg_entailment_score = sum(cv.entailment_score for cv in claim_validations) / total_claims
        
        # Determine verdict based on ratios and thresholds
        if contradicted_ratio > 0.2:  # More than 20% contradicted
            return OverallVerdict.NEEDS_REGENERATION, avg_entailment_score * 0.5
        
        elif entailed_ratio >= 0.7:  # At least 70% entailed
            return OverallVerdict.SATISFACTORY, avg_entailment_score
        
        elif neutral_ratio > 0.5:  # More than 50% neutral (unsupported)
            if len(evidence_documents) < 3:
                return OverallVerdict.NEEDS_EXPANDED_RETRIEVAL, avg_entailment_score * 0.7
            else:
                return OverallVerdict.NEEDS_CLARIFICATION, avg_entailment_score * 0.8
        
        else:
            return OverallVerdict.NEEDS_CLARIFICATION, avg_entailment_score * 0.8
    
    def _calculate_evidence_coverage(self,
                                   claim_validations: List[ClaimValidation],
                                   evidence_documents: List[Dict[str, Any]]) -> float:
        """Calculate how well evidence covers the claims"""
        
        if not evidence_documents or not claim_validations:
            return 0.0
        
        # Count how many evidence documents were used
        used_doc_ids = set()
        for cv in claim_validations:
            for span in cv.evidence_spans:
                used_doc_ids.add(span.get('doc_id', ''))
        
        coverage = len(used_doc_ids) / len(evidence_documents)
        return min(1.0, coverage)
    
    def _generate_recommendations(self,
                                overall_verdict: OverallVerdict,
                                claim_validations: List[ClaimValidation],
                                unsupported_claims: List[Claim],
                                contradicted_claims: List[Claim]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        
        recommendations = []
        
        if overall_verdict == OverallVerdict.SATISFACTORY:
            recommendations.append("Answer validation passed. Response is well-supported by evidence.")
        
        elif overall_verdict == OverallVerdict.NEEDS_CLARIFICATION:
            recommendations.append("Ask user for clarification on ambiguous aspects of the query.")
            if unsupported_claims:
                recommendations.append(f"Remove or qualify {len(unsupported_claims)} unsupported claims.")
        
        elif overall_verdict == OverallVerdict.NEEDS_REGENERATION:
            recommendations.append("Regenerate answer with better grounding in evidence.")
            if contradicted_claims:
                recommendations.append(f"Address {len(contradicted_claims)} contradicted claims.")
        
        elif overall_verdict == OverallVerdict.NEEDS_EXPANDED_RETRIEVAL:
            recommendations.append("Expand retrieval to find more relevant evidence.")
            recommendations.append("Consider using different retrieval strategies or query expansion.")
        
        # Specific recommendations for claim issues
        if len(contradicted_claims) > 0:
            recommendations.append(f"Critical: {len(contradicted_claims)} claims contradict evidence - immediate attention required.")
        
        if len(unsupported_claims) > len(claim_validations) * 0.3:
            recommendations.append("High number of unsupported claims - consider more conservative answer generation.")
        
        return recommendations
    
    def _create_error_validation_result(self,
                                      error_message: str,
                                      generated_answer: Dict[str, Any],
                                      evidence_documents: List[Dict[str, Any]]) -> ValidationResult:
        """Create error validation result"""
        
        return ValidationResult(
            overall_verdict=OverallVerdict.NEEDS_REGENERATION,
            overall_confidence=0.0,
            claim_validations=[],
            unsupported_claims=[],
            contradicted_claims=[],
            evidence_coverage=0.0,
            recommendations=[f"Validation failed: {error_message}"],
            metadata={
                "error": error_message,
                "validation_time": 0.0,
                "total_claims": 0,
                "nli_model": self.nli_model_name
            }
        )
    
    def _update_validation_stats(self, validation_result: ValidationResult, processing_time: float):
        """Update validation statistics"""
        
        # Update verdict counts
        verdict_key = f"{validation_result.overall_verdict.value}"
        if verdict_key in self.validation_stats:
            self.validation_stats[verdict_key] += 1
        
        # Update averages
        total_validations = self.validation_stats["total_validations"]
        
        # Average validation time
        total_time = (self.validation_stats["avg_validation_time"] * (total_validations - 1) + processing_time)
        self.validation_stats["avg_validation_time"] = total_time / total_validations
        
        # Average claims per answer
        total_claims = (self.validation_stats["avg_claims_per_answer"] * (total_validations - 1) + 
                       validation_result.metadata.get("total_claims", 0))
        self.validation_stats["avg_claims_per_answer"] = total_claims / total_validations
        
        # Average entailment score
        if validation_result.claim_validations:
            avg_score = sum(cv.entailment_score for cv in validation_result.claim_validations) / len(validation_result.claim_validations)
            total_score = (self.validation_stats["avg_entailment_score"] * (total_validations - 1) + avg_score)
            self.validation_stats["avg_entailment_score"] = total_score / total_validations
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        stats = self.validation_stats.copy()
        
        # Calculate derived metrics
        if stats["total_validations"] > 0:
            stats["satisfactory_rate"] = stats["satisfactory_answers"] / stats["total_validations"]
        
        return stats
    
    def clear_stats(self):
        """Clear validation statistics"""
        
        self.validation_stats = {
            "total_validations": 0,
            "satisfactory_answers": 0,
            "needs_clarification": 0,
            "needs_regeneration": 0,
            "needs_expanded_retrieval": 0,
            "avg_validation_time": 0.0,
            "avg_claims_per_answer": 0.0,
            "avg_entailment_score": 0.0
        }
    
    def update_thresholds(self, 
                         confidence_threshold: float = None,
                         entailment_threshold: float = None):
        """Update validation thresholds"""
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        if entailment_threshold is not None:
            self.entailment_threshold = entailment_threshold
        
        logger.info(f"Updated thresholds - confidence: {self.confidence_threshold}, entailment: {self.entailment_threshold}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        
        return {
            "nli_model": self.nli_model_name,
            "claim_extractor_model": self.claim_extractor_model,
            "device": self.device,
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
            "entailment_threshold": self.entailment_threshold,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "models_loaded": self.nli_pipeline is not None
        }