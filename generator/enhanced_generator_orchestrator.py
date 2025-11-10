"""
Enhanced Generator Orchestrator with LLM-as-Judge Validation

This orchestrator integrates the LLM-as-Judge validator to provide:
1. Answer validation with NLI-based claim checking
2. Automatic answer refinement based on validation results
3. Iterative improvement with retrieval expansion
4. Confidence-aware response generation
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .generator_orchestrator import GeneratorOrchestrator
from .llm_judge_validator import (
    LLMJudgeValidator, ValidationResult, OverallVerdict, 
    EntailmentLabel, ClaimValidation
)
from .enhanced_response_formatter import (
    EnhancedResponseFormatter, FeedbackType, OutputFormat
)
from .feedback_learning_system import FeedbackLearningSystem

logger = logging.getLogger(__name__)

class EnhancedGeneratorOrchestrator(GeneratorOrchestrator):
    """Enhanced generator orchestrator with LLM-as-Judge validation"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "auto",
                 max_workers: int = 4,
                 enable_async: bool = True,
                 enable_validation: bool = True,
                 nli_model_name: str = "microsoft/deberta-large-mnli",
                 max_refinement_iterations: int = 2,
                 validation_threshold: float = 0.7):
        
        # Initialize base orchestrator
        super().__init__(model_name, device, max_workers, enable_async)
        
        # Validation configuration
        self.enable_validation = enable_validation
        self.max_refinement_iterations = max_refinement_iterations
        self.validation_threshold = validation_threshold
        
        # Initialize LLM-as-Judge validator
        self.validator = None
        if enable_validation:
            try:
                self.validator = LLMJudgeValidator(
                    nli_model_name=nli_model_name,
                    device=device,
                    confidence_threshold=validation_threshold
                )
                logger.info("LLM-as-Judge validator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize validator: {e}")
                self.enable_validation = False
        
        # Initialize Enhanced Response Formatter with Feedback Collection
        self.enhanced_formatter = EnhancedResponseFormatter(
            enable_feedback_collection=True,
            max_summary_length=300,
            max_answer_length=2000,
            max_references=10
        )
        
        # Initialize Feedback Learning System
        self.learning_system = FeedbackLearningSystem(
            learning_rate=0.1,
            min_feedback_threshold=3,  # Lower threshold for demo
            enable_auto_adjustments=True
        )
        
        # Enhanced statistics
        self.enhanced_stats = {
            "validated_generations": 0,
            "refinement_iterations": 0,
            "satisfactory_first_attempt": 0,
            "needed_clarification": 0,
            "needed_regeneration": 0,
            "needed_expanded_retrieval": 0,
            "avg_validation_time": 0.0,
            "avg_refinement_iterations": 0.0
        }
        
        logger.info(f"Enhanced Generator Orchestrator initialized with validation: {enable_validation}")
    
    def generate_validated_answer(self,
                                query: str,
                                intent: str,
                                evidence_documents: List[Dict[str, Any]],
                                previous_context: str = None,
                                use_cache: bool = True,
                                orchestrator_callback = None) -> Dict[str, Any]:
        """
        Generate answer with LLM-as-Judge validation and iterative refinement
        
        Args:
            query: User query
            intent: Query intent
            evidence_documents: Retrieved evidence documents
            previous_context: Previous conversation context
            use_cache: Whether to use response caching
            orchestrator_callback: Callback function for orchestrator communication
            
        Returns:
            Validated and potentially refined answer
        """
        
        start_time = time.time()
        self.enhanced_stats["validated_generations"] += 1
        
        # Track refinement iterations
        iteration = 0
        current_evidence = evidence_documents.copy()
        validation_history = []
        
        while iteration < self.max_refinement_iterations:
            iteration += 1
            
            # Generate answer using base orchestrator
            generated_answer = super().generate_grounded_answer(
                query=query,
                intent=intent,
                evidence_documents=current_evidence,
                previous_context=previous_context,
                use_cache=use_cache and iteration == 1  # Only use cache for first iteration
            )
            
            # Skip validation if disabled or validator unavailable
            if not self.enable_validation or not self.validator:
                logger.info("Validation disabled - returning answer without validation")
                generated_answer["metadata"]["validation_status"] = "skipped"
                return generated_answer
            
            # Validate the generated answer
            validation_start = time.time()
            validation_result = self.validator.validate_answer(
                generated_answer=generated_answer,
                evidence_documents=current_evidence,
                query=query,
                intent=intent
            )
            validation_time = time.time() - validation_start
            
            # Store validation in history
            validation_history.append({
                "iteration": iteration,
                "validation_result": validation_result,
                "validation_time": validation_time
            })
            
            # Update statistics
            self._update_enhanced_stats(validation_result, validation_time, iteration)
            
            # Check if answer is satisfactory
            if validation_result.overall_verdict == OverallVerdict.SATISFACTORY:
                logger.info(f"Answer validated successfully on iteration {iteration}")
                
                # Add validation metadata to answer
                generated_answer["metadata"]["validation"] = {
                    "status": "passed",
                    "iterations": iteration,
                    "overall_verdict": validation_result.overall_verdict.value,
                    "confidence": validation_result.overall_confidence,
                    "claim_count": len(validation_result.claim_validations),
                    "evidence_coverage": validation_result.evidence_coverage,
                    "validation_time": validation_time,
                    "validation_history": validation_history
                }
                
                if iteration == 1:
                    self.enhanced_stats["satisfactory_first_attempt"] += 1
                
                return generated_answer
            
            # Handle different validation verdicts
            if validation_result.overall_verdict == OverallVerdict.NEEDS_CLARIFICATION:
                logger.info("Answer needs clarification - asking orchestrator")
                self.enhanced_stats["needed_clarification"] += 1
                
                if orchestrator_callback:
                    clarification_request = self._create_clarification_request(
                        validation_result, query, generated_answer
                    )
                    
                    # Try to get clarification from orchestrator
                    clarification_response = orchestrator_callback(clarification_request)
                    
                    if clarification_response and clarification_response.get("expanded_query"):
                        # Use expanded query for next iteration
                        query = clarification_response["expanded_query"]
                        logger.info(f"Using expanded query: {query}")
                        continue
                
                # If no clarification available, return with warning
                generated_answer["metadata"]["validation"] = {
                    "status": "needs_clarification",
                    "iterations": iteration,
                    "overall_verdict": validation_result.overall_verdict.value,
                    "recommendations": validation_result.recommendations,
                    "validation_history": validation_history
                }
                return generated_answer
            
            elif validation_result.overall_verdict == OverallVerdict.NEEDS_REGENERATION:
                logger.info("Answer needs regeneration - refining generation")
                self.enhanced_stats["needed_regeneration"] += 1
                
                # Remove contradicted claims and regenerate
                if iteration < self.max_refinement_iterations:
                    # Modify generation parameters for next iteration
                    self._adjust_generation_for_refinement(validation_result)
                    continue
                
            elif validation_result.overall_verdict == OverallVerdict.NEEDS_EXPANDED_RETRIEVAL:
                logger.info("Answer needs expanded retrieval")
                self.enhanced_stats["needed_expanded_retrieval"] += 1
                
                if orchestrator_callback:
                    retrieval_request = self._create_retrieval_expansion_request(
                        validation_result, query, current_evidence
                    )
                    
                    # Try to get more evidence from orchestrator
                    retrieval_response = orchestrator_callback(retrieval_request)
                    
                    if retrieval_response and retrieval_response.get("additional_evidence"):
                        # Add new evidence for next iteration
                        additional_evidence = retrieval_response["additional_evidence"]
                        current_evidence.extend(additional_evidence)
                        logger.info(f"Added {len(additional_evidence)} additional evidence documents")
                        continue
                
                # If no additional retrieval available, continue with current evidence
                logger.warning("Could not expand retrieval - continuing with current evidence")
        
        # Max iterations reached - return best attempt with validation metadata
        logger.warning(f"Max refinement iterations ({self.max_refinement_iterations}) reached")
        
        generated_answer["metadata"]["validation"] = {
            "status": "max_iterations_reached",
            "iterations": iteration,
            "overall_verdict": validation_result.overall_verdict.value if validation_result else "unknown",
            "recommendations": validation_result.recommendations if validation_result else [],
            "validation_history": validation_history
        }
        
        return generated_answer
    
    def generate_enhanced_response(self,
                                 query: str,
                                 intent: str,
                                 evidence_documents: List[Dict[str, Any]],
                                 previous_context: str = None,
                                 output_formats: List[OutputFormat] = None,
                                 enable_feedback: bool = True,
                                 orchestrator_callback = None) -> Dict[str, Any]:
        """
        Generate response with validation, enhanced formatting, and feedback collection
        
        Args:
            query: User query
            intent: Query intent
            evidence_documents: Retrieved evidence documents
            previous_context: Previous conversation context
            output_formats: Desired output formats (JSON, text, markdown, HTML)
            enable_feedback: Whether to enable feedback collection
            orchestrator_callback: Callback function for orchestrator communication
            
        Returns:
            Enhanced response with multiple formats and feedback capabilities
        """
        
        # Generate validated answer
        validated_answer = self.generate_validated_answer(
            query=query,
            intent=intent,
            evidence_documents=evidence_documents,
            previous_context=previous_context,
            orchestrator_callback=orchestrator_callback
        )
        
        # Default output formats
        if output_formats is None:
            output_formats = [OutputFormat.JSON, OutputFormat.TEXT, OutputFormat.HTML]
        
        # Format response with enhanced formatter
        enhanced_response = self.enhanced_formatter.format_enhanced_response(
            raw_answer=validated_answer,
            evidence_documents=evidence_documents,
            metadata=validated_answer.get("metadata", {}),
            query=query,
            intent=intent,
            output_formats=output_formats,
            enable_feedback=enable_feedback
        )
        
        # Add validation information to response
        validation_metadata = validated_answer.get("metadata", {}).get("validation", {})
        if validation_metadata:
            enhanced_response["validation_summary"] = {
                "status": validation_metadata.get("status"),
                "verdict": validation_metadata.get("overall_verdict"),
                "confidence": validation_metadata.get("confidence", 0.0),
                "iterations": validation_metadata.get("iterations", 0)
            }
        
        return enhanced_response
    
    def collect_response_feedback(self,
                                response_id: str,
                                feedback_type: FeedbackType,
                                rating: Optional[int] = None,
                                comment: Optional[str] = None,
                                correction: Optional[str] = None,
                                user_id: Optional[str] = None,
                                session_id: Optional[str] = None) -> str:
        """
        Collect feedback for a generated response
        
        Args:
            response_id: ID of the response being rated
            feedback_type: Type of feedback (thumbs up/down, not correct, etc.)
            rating: Optional 1-5 rating
            comment: Optional text comment
            correction: Optional correction text
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            Feedback entry ID
        """
        
        return self.enhanced_formatter.collect_feedback(
            response_id=response_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            correction=correction,
            user_id=user_id,
            session_id=session_id
        )
    
    def get_response_feedback_summary(self, response_id: str) -> Dict[str, Any]:
        """Get feedback summary for a specific response"""
        
        return self.enhanced_formatter.get_feedback_summary(response_id)
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights from collected feedback"""
        
        return self.enhanced_formatter.get_learning_insights()
    
    def export_feedback_data(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export feedback data for analysis"""
        
        return self.enhanced_formatter.export_feedback_data(format_type)
    
    def analyze_feedback_and_learn(self, session_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze collected feedback and generate learning insights
        
        Args:
            session_context: Optional session context for pattern analysis
            
        Returns:
            Learning analysis results and improvement recommendations
        """
        
        # Get all feedback entries and response traces
        all_feedback_entries = []
        all_response_traces = []
        
        for response_id, feedback_list in self.enhanced_formatter.feedback_entries.items():
            all_feedback_entries.extend(feedback_list)
            
            if response_id in self.enhanced_formatter.response_traces:
                trace = self.enhanced_formatter.response_traces[response_id]
                # Add trace for each feedback entry
                for _ in feedback_list:
                    all_response_traces.append(trace)
        
        if not all_feedback_entries:
            return {"message": "No feedback available for learning", "insights": []}
        
        # Analyze feedback and generate insights
        insights = self.learning_system.analyze_feedback_batch(
            feedback_entries=all_feedback_entries,
            response_traces=all_response_traces,
            session_context=session_context
        )
        
        # Get learning recommendations
        recommendations = self.learning_system.get_learning_recommendations()
        
        # Generate improvement plan
        improvement_plan = self.learning_system.generate_improvement_plan()
        
        # Apply learned adjustments to system components
        self._apply_learning_adjustments(insights)
        
        return {
            "insights_generated": len(insights),
            "insights": [
                {
                    "type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "priority": insight.priority,
                    "recommended_actions": [action.value for action in insight.recommended_actions],
                    "affected_components": insight.affected_components
                }
                for insight in insights
            ],
            "recommendations": recommendations,
            "improvement_plan": improvement_plan,
            "system_health": improvement_plan["overall_health"],
            "learning_summary": self.learning_system.get_learning_summary()
        }
    
    def _apply_learning_adjustments(self, insights: List[Any]):
        """Apply learning-based adjustments to system components"""
        
        for insight in insights:
            if insight.confidence >= 0.8 and insight.priority >= 4:
                # Apply high-confidence, high-priority adjustments
                
                if "confidence_threshold" in insight.description.lower():
                    # Adjust confidence estimation
                    if hasattr(self.confidence_estimator, 'update_weights'):
                        if "overconfident" in insight.description:
                            self.confidence_estimator.update_weights(evidence_weight=1.2)
                        elif "underconfident" in insight.description:
                            self.confidence_estimator.update_weights(evidence_weight=0.9)
                
                if "validation" in insight.description.lower() and self.validator:
                    # Adjust validation thresholds
                    if "lenient" in insight.description:
                        current_threshold = getattr(self.validator, 'entailment_threshold', 0.5)
                        self.validator.update_thresholds(entailment_threshold=min(0.8, current_threshold + 0.1))
                    elif "strict" in insight.description:
                        current_threshold = getattr(self.validator, 'entailment_threshold', 0.5)
                        self.validator.update_thresholds(entailment_threshold=max(0.3, current_threshold - 0.1))
                
                if "generation" in insight.description.lower():
                    # Adjust generation parameters
                    if "incomplete" in insight.description or "unclear" in insight.description:
                        self.update_generation_config(
                            max_length=min(2500, getattr(self.answer_generator, "max_length", 2000) + 200),
                            temperature=max(0.3, getattr(self.answer_generator, "temperature", 0.7) - 0.1)
                        )
                    elif "inaccurate" in insight.description:
                        self.update_generation_config(
                            temperature=max(0.2, getattr(self.answer_generator, "temperature", 0.7) - 0.15)
                        )
        
        logger.info(f"Applied learning adjustments based on {len(insights)} insights")
    
    def get_learning_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive learning dashboard data"""
        
        learning_summary = self.learning_system.get_learning_summary()
        formatter_stats = self.enhanced_formatter.get_enhanced_stats()
        
        return {
            "learning_system": learning_summary,
            "response_formatting": formatter_stats,
            "recent_adjustments": self.learning_system.adjustment_history[-10:],  # Last 10 adjustments
            "performance_metrics": {
                name: {
                    "current": metric.current_value,
                    "target": metric.target_value,
                    "trend": metric.trend,
                    "last_updated": metric.last_updated.isoformat()
                }
                for name, metric in self.learning_system.performance_metrics.items()
            }
        }
    
    def _create_clarification_request(self, 
                                    validation_result: ValidationResult,
                                    query: str,
                                    generated_answer: Dict[str, Any]) -> Dict[str, Any]:
        """Create clarification request for orchestrator"""
        
        unsupported_claims = [claim.text for claim in validation_result.unsupported_claims]
        
        return {
            "type": "clarification_request",
            "original_query": query,
            "unsupported_claims": unsupported_claims,
            "recommendations": validation_result.recommendations,
            "suggested_expansions": [
                f"Can you provide more details about {claim}?" 
                for claim in unsupported_claims[:3]
            ]
        }
    
    def _create_retrieval_expansion_request(self,
                                          validation_result: ValidationResult,
                                          query: str,
                                          current_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create retrieval expansion request for orchestrator"""
        
        # Extract key terms from unsupported claims
        expansion_terms = []
        for claim in validation_result.unsupported_claims:
            # Simple keyword extraction
            words = claim.text.split()
            expansion_terms.extend([word for word in words if len(word) > 3])
        
        return {
            "type": "retrieval_expansion",
            "original_query": query,
            "expansion_terms": list(set(expansion_terms))[:10],  # Top 10 unique terms
            "current_evidence_count": len(current_evidence),
            "requested_additional_count": min(10, len(validation_result.unsupported_claims) * 2)
        }
    
    def _adjust_generation_for_refinement(self, validation_result: ValidationResult):
        """Adjust generation parameters based on validation results"""
        
        # Lower temperature for more conservative generation
        self.update_generation_config(
            temperature=max(0.3, getattr(self.answer_generator, "temperature", 0.7) - 0.2),
            do_sample=True,
            top_p=0.8
        )
        
        # Increase confidence threshold
        if hasattr(self.confidence_estimator, 'update_weights'):
            self.confidence_estimator.update_weights(
                evidence_weight=1.2,  # Increase evidence importance
                consistency_weight=1.1
            )
        
        logger.info("Adjusted generation parameters for refinement")
    
    def _update_enhanced_stats(self, 
                             validation_result: ValidationResult, 
                             validation_time: float, 
                             iteration: int):
        """Update enhanced statistics"""
        
        # Update iteration stats
        self.enhanced_stats["refinement_iterations"] += iteration - 1
        
        # Update validation time
        total_validations = self.enhanced_stats["validated_generations"]
        total_time = (self.enhanced_stats["avg_validation_time"] * (total_validations - 1) + validation_time)
        self.enhanced_stats["avg_validation_time"] = total_time / total_validations
        
        # Update average refinement iterations
        total_refinement = (self.enhanced_stats["avg_refinement_iterations"] * (total_validations - 1) + 
                          (iteration - 1))
        self.enhanced_stats["avg_refinement_iterations"] = total_refinement / total_validations
    
    def batch_validate_answers(self,
                             queries: List[str],
                             intents: List[str],
                             evidence_batches: List[List[Dict[str, Any]]],
                             previous_contexts: List[str] = None,
                             use_parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Generate and validate answers for multiple queries in batch
        
        Args:
            queries: List of user queries
            intents: List of query intents
            evidence_batches: List of evidence document lists
            previous_contexts: List of previous contexts
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of validated answers
        """
        
        if not previous_contexts:
            previous_contexts = [None] * len(queries)
        
        if use_parallel and self.enable_async and len(queries) > 1:
            return self._batch_validate_parallel(queries, intents, evidence_batches, previous_contexts)
        else:
            return self._batch_validate_sequential(queries, intents, evidence_batches, previous_contexts)
    
    def _batch_validate_sequential(self,
                                 queries: List[str],
                                 intents: List[str],
                                 evidence_batches: List[List[Dict[str, Any]]],
                                 previous_contexts: List[str]) -> List[Dict[str, Any]]:
        """Generate and validate batch answers sequentially"""
        
        results = []
        
        for query, intent, evidence_docs, context in zip(queries, intents, evidence_batches, previous_contexts):
            result = self.generate_validated_answer(query, intent, evidence_docs, context)
            results.append(result)
        
        return results
    
    def _batch_validate_parallel(self,
                               queries: List[str],
                               intents: List[str],
                               evidence_batches: List[List[Dict[str, Any]]],
                               previous_contexts: List[str]) -> List[Dict[str, Any]]:
        """Generate and validate batch answers in parallel"""
        
        futures = []
        
        for query, intent, evidence_docs, context in zip(queries, intents, evidence_batches, previous_contexts):
            future = self.executor.submit(
                self.generate_validated_answer,
                query, intent, evidence_docs, context
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=120)  # 2 minute timeout for validation
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel validation failed: {str(e)}")
                error_result = self._create_error_response("", str(e), [])
                error_result["metadata"]["validation"] = {"status": "error", "error": str(e)}
                results.append(error_result)
        
        return results
    
    def update_validation_config(self,
                               enable_validation: bool = None,
                               max_refinement_iterations: int = None,
                               validation_threshold: float = None,
                               nli_threshold: float = None):
        """Update validation configuration"""
        
        if enable_validation is not None:
            self.enable_validation = enable_validation
        
        if max_refinement_iterations is not None:
            self.max_refinement_iterations = max_refinement_iterations
        
        if validation_threshold is not None:
            self.validation_threshold = validation_threshold
        
        if self.validator and nli_threshold is not None:
            self.validator.update_thresholds(entailment_threshold=nli_threshold)
        
        logger.info(f"Updated validation config - enabled: {self.enable_validation}, "
                   f"max_iterations: {self.max_refinement_iterations}")
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced orchestrator statistics"""
        
        base_stats = self.get_orchestrator_stats()
        enhanced_stats = self.enhanced_stats.copy()
        
        # Calculate derived metrics
        if enhanced_stats["validated_generations"] > 0:
            enhanced_stats["satisfactory_first_attempt_rate"] = (
                enhanced_stats["satisfactory_first_attempt"] / enhanced_stats["validated_generations"]
            )
            enhanced_stats["clarification_rate"] = (
                enhanced_stats["needed_clarification"] / enhanced_stats["validated_generations"]
            )
            enhanced_stats["regeneration_rate"] = (
                enhanced_stats["needed_regeneration"] / enhanced_stats["validated_generations"]
            )
            enhanced_stats["expanded_retrieval_rate"] = (
                enhanced_stats["needed_expanded_retrieval"] / enhanced_stats["validated_generations"]
            )
        
        # Get validation stats if available
        validation_stats = {}
        if self.validator:
            validation_stats = self.validator.get_validation_stats()
        
        return {
            "base_orchestrator": base_stats,
            "enhanced_orchestrator": enhanced_stats,
            "validation": validation_stats,
            "system_info": {
                "validation_enabled": self.enable_validation,
                "max_refinement_iterations": self.max_refinement_iterations,
                "validation_threshold": self.validation_threshold,
                "validator_available": self.validator is not None
            }
        }
    
    def get_validation_report(self, query: str, answer: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed validation report for a specific answer"""
        
        if not self.enable_validation or not self.validator:
            return {"error": "Validation not available"}
        
        validation_metadata = answer.get("metadata", {}).get("validation", {})
        
        if not validation_metadata:
            return {"error": "No validation metadata found"}
        
        report = {
            "query": query,
            "validation_status": validation_metadata.get("status"),
            "iterations": validation_metadata.get("iterations", 0),
            "overall_verdict": validation_metadata.get("overall_verdict"),
            "confidence": validation_metadata.get("confidence", 0.0),
            "evidence_coverage": validation_metadata.get("evidence_coverage", 0.0),
            "recommendations": validation_metadata.get("recommendations", [])
        }
        
        # Add detailed validation history if available
        validation_history = validation_metadata.get("validation_history", [])
        if validation_history:
            report["validation_details"] = []
            
            for hist in validation_history:
                validation_result = hist.get("validation_result")
                if validation_result:
                    detail = {
                        "iteration": hist.get("iteration"),
                        "validation_time": hist.get("validation_time"),
                        "claim_count": len(validation_result.claim_validations),
                        "entailed_claims": len([cv for cv in validation_result.claim_validations 
                                              if cv.entailment_label == EntailmentLabel.ENTAILED]),
                        "neutral_claims": len(validation_result.unsupported_claims),
                        "contradicted_claims": len(validation_result.contradicted_claims)
                    }
                    report["validation_details"].append(detail)
        
        return report
    
    def clear_enhanced_stats(self):
        """Clear enhanced statistics"""
        
        self.enhanced_stats = {
            "validated_generations": 0,
            "refinement_iterations": 0,
            "satisfactory_first_attempt": 0,
            "needed_clarification": 0,
            "needed_regeneration": 0,
            "needed_expanded_retrieval": 0,
            "avg_validation_time": 0.0,
            "avg_refinement_iterations": 0.0
        }
        
        if self.validator:
            self.validator.clear_stats()
        
        # Also clear base stats
        self.clear_all_stats()
        
        logger.info("Cleared enhanced orchestrator statistics")
    
    def shutdown(self):
        """Shutdown enhanced orchestrator"""
        
        # Shutdown base orchestrator
        super().shutdown()
        
        # Additional cleanup for validator
        if self.validator:
            logger.info("Shutting down LLM-as-Judge validator")
        
        logger.info("Enhanced Generator Orchestrator shutdown complete")