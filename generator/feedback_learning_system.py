"""
Feedback Learning System

Implements continuous learning from user feedback to improve RAG pipeline performance.
Integrates with session memory and response formatting for comprehensive learning.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum

from .enhanced_response_formatter import FeedbackType, FeedbackEntry, ResponseTrace

logger = logging.getLogger(__name__)

class LearningAction(Enum):
    """Types of learning actions that can be taken"""
    ADJUST_CONFIDENCE_THRESHOLD = "adjust_confidence_threshold"
    MODIFY_RETRIEVAL_STRATEGY = "modify_retrieval_strategy"
    UPDATE_GENERATION_PARAMS = "update_generation_params"
    ENHANCE_VALIDATION_RULES = "enhance_validation_rules"
    IMPROVE_ENTITY_EXTRACTION = "improve_entity_extraction"
    REFINE_INTENT_DETECTION = "refine_intent_detection"
    OPTIMIZE_RERANKING = "optimize_reranking"

@dataclass
class LearningInsight:
    """Individual learning insight from feedback analysis"""
    insight_id: str
    insight_type: str
    description: str
    confidence: float
    evidence_count: int
    recommended_actions: List[LearningAction]
    affected_components: List[str]
    priority: int  # 1-5, 5 being highest
    timestamp: datetime

@dataclass
class PerformanceMetric:
    """Performance metric tracking"""
    metric_name: str
    current_value: float
    target_value: float
    trend: str  # "improving", "declining", "stable"
    last_updated: datetime

class FeedbackLearningSystem:
    """Comprehensive feedback learning system for RAG pipeline improvement"""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 min_feedback_threshold: int = 5,
                 confidence_adjustment_factor: float = 0.05,
                 enable_auto_adjustments: bool = True):
        
        self.learning_rate = learning_rate
        self.min_feedback_threshold = min_feedback_threshold
        self.confidence_adjustment_factor = confidence_adjustment_factor
        self.enable_auto_adjustments = enable_auto_adjustments
        
        # Learning state
        self.learning_insights: List[LearningInsight] = []
        self.performance_metrics: Dict[str, PerformanceMetric] = {}
        self.adjustment_history: List[Dict[str, Any]] = []
        
        # Component performance tracking
        self.component_performance = {
            "retrieval": defaultdict(list),
            "reranking": defaultdict(list),
            "generation": defaultdict(list),
            "validation": defaultdict(list),
            "overall": defaultdict(list)
        }
        
        # Learning patterns
        self.feedback_patterns = {
            "intent_performance": defaultdict(lambda: {"positive": 0, "negative": 0, "ratings": []}),
            "confidence_accuracy": defaultdict(list),
            "correction_topics": defaultdict(int),
            "user_preferences": defaultdict(int),
            "session_patterns": defaultdict(list)
        }
        
        # Initialize performance metrics
        self._initialize_performance_metrics()
        
        logger.info("Feedback Learning System initialized")
    
    def _initialize_performance_metrics(self):
        """Initialize performance metrics tracking"""
        
        metrics = [
            ("user_satisfaction", 0.0, 0.8),
            ("response_accuracy", 0.0, 0.9),
            ("confidence_calibration", 0.0, 0.85),
            ("feedback_rate", 0.0, 0.3),
            ("correction_rate", 0.0, 0.1),
            ("average_rating", 0.0, 4.0),
            ("validation_success_rate", 0.0, 0.9),
            ("retrieval_precision", 0.0, 0.8)
        ]
        
        for metric_name, current, target in metrics:
            self.performance_metrics[metric_name] = PerformanceMetric(
                metric_name=metric_name,
                current_value=current,
                target_value=target,
                trend="stable",
                last_updated=datetime.now()
            )
    
    def analyze_feedback_batch(self, 
                             feedback_entries: List[FeedbackEntry],
                             response_traces: List[ResponseTrace],
                             session_context: Optional[Dict[str, Any]] = None) -> List[LearningInsight]:
        """
        Analyze a batch of feedback to generate learning insights
        
        Args:
            feedback_entries: List of feedback entries to analyze
            response_traces: Corresponding response traces
            session_context: Optional session context for pattern analysis
            
        Returns:
            List of learning insights
        """
        
        if len(feedback_entries) < self.min_feedback_threshold:
            logger.info(f"Insufficient feedback for learning (need {self.min_feedback_threshold}, got {len(feedback_entries)})")
            return []
        
        insights = []
        
        # 1. Analyze rating patterns
        rating_insights = self._analyze_rating_patterns(feedback_entries, response_traces)
        insights.extend(rating_insights)
        
        # 2. Analyze correction patterns
        correction_insights = self._analyze_correction_patterns(feedback_entries, response_traces)
        insights.extend(correction_insights)
        
        # 3. Analyze confidence calibration
        confidence_insights = self._analyze_confidence_calibration(feedback_entries, response_traces)
        insights.extend(confidence_insights)
        
        # 4. Analyze intent-specific performance
        intent_insights = self._analyze_intent_performance(feedback_entries, response_traces)
        insights.extend(intent_insights)
        
        # 5. Analyze validation effectiveness
        validation_insights = self._analyze_validation_effectiveness(feedback_entries, response_traces)
        insights.extend(validation_insights)
        
        # 6. Analyze session patterns if context provided
        if session_context:
            session_insights = self._analyze_session_patterns(feedback_entries, response_traces, session_context)
            insights.extend(session_insights)
        
        # Store insights
        self.learning_insights.extend(insights)
        
        # Update performance metrics
        self._update_performance_metrics(feedback_entries, response_traces)
        
        # Apply automatic adjustments if enabled
        if self.enable_auto_adjustments:
            self._apply_automatic_adjustments(insights)
        
        return insights
    
    def _analyze_rating_patterns(self, 
                               feedback_entries: List[FeedbackEntry],
                               response_traces: List[ResponseTrace]) -> List[LearningInsight]:
        """Analyze rating patterns to identify improvement opportunities"""
        
        insights = []
        
        # Group feedback by rating
        ratings = [f.rating for f in feedback_entries if f.rating is not None]
        if not ratings:
            return insights
        
        avg_rating = sum(ratings) / len(ratings)
        low_ratings = [r for r in ratings if r <= 2]
        high_ratings = [r for r in ratings if r >= 4]
        
        # Low rating analysis
        if len(low_ratings) / len(ratings) > 0.3:  # More than 30% low ratings
            insight = LearningInsight(
                insight_id=f"rating_analysis_{int(time.time())}",
                insight_type="rating_pattern",
                description=f"High percentage of low ratings ({len(low_ratings)/len(ratings):.1%}). Average rating: {avg_rating:.1f}",
                confidence=0.8,
                evidence_count=len(low_ratings),
                recommended_actions=[
                    LearningAction.ADJUST_CONFIDENCE_THRESHOLD,
                    LearningAction.ENHANCE_VALIDATION_RULES,
                    LearningAction.UPDATE_GENERATION_PARAMS
                ],
                affected_components=["generation", "validation", "overall"],
                priority=4,
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        # High confidence but low rating analysis
        low_rated_traces = [trace for trace, feedback in zip(response_traces, feedback_entries) 
                           if feedback.rating and feedback.rating <= 2]
        
        high_conf_low_rating = [trace for trace in low_rated_traces 
                               if trace.generated_response.get("confidence", 0) > 0.7]
        
        if len(high_conf_low_rating) > 2:
            insight = LearningInsight(
                insight_id=f"confidence_calibration_{int(time.time())}",
                insight_type="confidence_calibration",
                description=f"High confidence responses receiving low ratings ({len(high_conf_low_rating)} cases)",
                confidence=0.9,
                evidence_count=len(high_conf_low_rating),
                recommended_actions=[
                    LearningAction.ADJUST_CONFIDENCE_THRESHOLD,
                    LearningAction.ENHANCE_VALIDATION_RULES
                ],
                affected_components=["confidence_estimation", "validation"],
                priority=5,
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _analyze_correction_patterns(self,
                                   feedback_entries: List[FeedbackEntry],
                                   response_traces: List[ResponseTrace]) -> List[LearningInsight]:
        """Analyze correction patterns to identify knowledge gaps"""
        
        insights = []
        
        # Extract corrections
        corrections = [f.correction for f in feedback_entries if f.correction]
        if len(corrections) < 3:
            return insights
        
        # Simple keyword analysis of corrections
        correction_keywords = Counter()
        correction_topics = Counter()
        
        for correction in corrections:
            words = correction.lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    correction_keywords[word] += 1
        
        # Identify common correction topics
        topic_keywords = {
            "outdated": ["outdated", "old", "recent", "latest", "current"],
            "incomplete": ["missing", "incomplete", "more", "additional", "also"],
            "inaccurate": ["wrong", "incorrect", "false", "error", "mistake"],
            "unclear": ["unclear", "confusing", "explain", "clarify", "better"]
        }
        
        for topic, keywords in topic_keywords.items():
            count = sum(correction_keywords[keyword] for keyword in keywords if keyword in correction_keywords)
            if count > 0:
                correction_topics[topic] = count
        
        # Generate insights for common correction topics
        for topic, count in correction_topics.most_common(3):
            if count >= 2:  # At least 2 occurrences
                actions = []
                components = []
                
                if topic == "outdated":
                    actions = [LearningAction.MODIFY_RETRIEVAL_STRATEGY, LearningAction.OPTIMIZE_RERANKING]
                    components = ["retrieval", "reranking"]
                elif topic == "incomplete":
                    actions = [LearningAction.UPDATE_GENERATION_PARAMS, LearningAction.MODIFY_RETRIEVAL_STRATEGY]
                    components = ["generation", "retrieval"]
                elif topic == "inaccurate":
                    actions = [LearningAction.ENHANCE_VALIDATION_RULES, LearningAction.ADJUST_CONFIDENCE_THRESHOLD]
                    components = ["validation", "generation"]
                elif topic == "unclear":
                    actions = [LearningAction.UPDATE_GENERATION_PARAMS, LearningAction.REFINE_INTENT_DETECTION]
                    components = ["generation", "intent_detection"]
                
                insight = LearningInsight(
                    insight_id=f"correction_{topic}_{int(time.time())}",
                    insight_type="correction_pattern",
                    description=f"Common correction pattern: {topic} ({count} occurrences)",
                    confidence=0.7 + (count * 0.1),
                    evidence_count=count,
                    recommended_actions=actions,
                    affected_components=components,
                    priority=3 + min(count, 2),
                    timestamp=datetime.now()
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_confidence_calibration(self,
                                      feedback_entries: List[FeedbackEntry],
                                      response_traces: List[ResponseTrace]) -> List[LearningInsight]:
        """Analyze confidence calibration accuracy"""
        
        insights = []
        
        # Collect confidence vs rating pairs
        conf_rating_pairs = []
        for feedback, trace in zip(feedback_entries, response_traces):
            if feedback.rating is not None:
                confidence = trace.generated_response.get("confidence", 0.0)
                conf_rating_pairs.append((confidence, feedback.rating))
        
        if len(conf_rating_pairs) < 5:
            return insights
        
        # Analyze calibration
        overconfident_cases = [(conf, rating) for conf, rating in conf_rating_pairs 
                              if conf > 0.8 and rating <= 2]
        underconfident_cases = [(conf, rating) for conf, rating in conf_rating_pairs 
                               if conf < 0.5 and rating >= 4]
        
        # Overconfidence insight
        if len(overconfident_cases) >= 2:
            insight = LearningInsight(
                insight_id=f"overconfidence_{int(time.time())}",
                insight_type="confidence_calibration",
                description=f"System is overconfident: {len(overconfident_cases)} high-confidence responses received low ratings",
                confidence=0.8,
                evidence_count=len(overconfident_cases),
                recommended_actions=[
                    LearningAction.ADJUST_CONFIDENCE_THRESHOLD,
                    LearningAction.ENHANCE_VALIDATION_RULES
                ],
                affected_components=["confidence_estimation", "validation"],
                priority=4,
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        # Underconfidence insight
        if len(underconfident_cases) >= 2:
            insight = LearningInsight(
                insight_id=f"underconfidence_{int(time.time())}",
                insight_type="confidence_calibration",
                description=f"System is underconfident: {len(underconfident_cases)} low-confidence responses received high ratings",
                confidence=0.7,
                evidence_count=len(underconfident_cases),
                recommended_actions=[
                    LearningAction.ADJUST_CONFIDENCE_THRESHOLD
                ],
                affected_components=["confidence_estimation"],
                priority=3,
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _analyze_intent_performance(self,
                                  feedback_entries: List[FeedbackEntry],
                                  response_traces: List[ResponseTrace]) -> List[LearningInsight]:
        """Analyze performance by query intent"""
        
        insights = []
        
        # Group by intent
        intent_performance = defaultdict(lambda: {"ratings": [], "feedback_types": []})
        
        for feedback, trace in zip(feedback_entries, response_traces):
            intent = trace.intent
            if feedback.rating:
                intent_performance[intent]["ratings"].append(feedback.rating)
            intent_performance[intent]["feedback_types"].append(feedback.feedback_type)
        
        # Analyze each intent
        for intent, data in intent_performance.items():
            if len(data["ratings"]) >= 3:  # Minimum samples
                avg_rating = sum(data["ratings"]) / len(data["ratings"])
                negative_feedback = sum(1 for ft in data["feedback_types"] 
                                      if ft in [FeedbackType.THUMBS_DOWN, FeedbackType.NOT_CORRECT])
                
                if avg_rating < 3.0 or negative_feedback / len(data["feedback_types"]) > 0.4:
                    actions = []
                    components = []
                    
                    if intent == "fact":
                        actions = [LearningAction.OPTIMIZE_RERANKING, LearningAction.ENHANCE_VALIDATION_RULES]
                        components = ["reranking", "validation"]
                    elif intent == "explain":
                        actions = [LearningAction.UPDATE_GENERATION_PARAMS, LearningAction.MODIFY_RETRIEVAL_STRATEGY]
                        components = ["generation", "retrieval"]
                    elif intent == "compare":
                        actions = [LearningAction.MODIFY_RETRIEVAL_STRATEGY, LearningAction.UPDATE_GENERATION_PARAMS]
                        components = ["retrieval", "generation"]
                    else:
                        actions = [LearningAction.REFINE_INTENT_DETECTION, LearningAction.UPDATE_GENERATION_PARAMS]
                        components = ["intent_detection", "generation"]
                    
                    insight = LearningInsight(
                        insight_id=f"intent_{intent}_{int(time.time())}",
                        insight_type="intent_performance",
                        description=f"Poor performance for {intent} queries (avg rating: {avg_rating:.1f})",
                        confidence=0.8,
                        evidence_count=len(data["ratings"]),
                        recommended_actions=actions,
                        affected_components=components,
                        priority=4,
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
        
        return insights
    
    def _analyze_validation_effectiveness(self,
                                        feedback_entries: List[FeedbackEntry],
                                        response_traces: List[ResponseTrace]) -> List[LearningInsight]:
        """Analyze effectiveness of LLM-as-Judge validation"""
        
        insights = []
        
        # Collect validation vs feedback data
        validation_feedback_pairs = []
        for feedback, trace in zip(feedback_entries, response_traces):
            validation_results = trace.validation_results
            if validation_results and feedback.rating is not None:
                validation_status = validation_results.get("status")
                validation_verdict = validation_results.get("overall_verdict")
                validation_confidence = validation_results.get("confidence", 0.0)
                
                validation_feedback_pairs.append({
                    "validation_status": validation_status,
                    "validation_verdict": validation_verdict,
                    "validation_confidence": validation_confidence,
                    "user_rating": feedback.rating,
                    "feedback_type": feedback.feedback_type
                })
        
        if len(validation_feedback_pairs) < 5:
            return insights
        
        # Analyze validation accuracy
        passed_validation = [pair for pair in validation_feedback_pairs 
                           if pair["validation_status"] == "passed"]
        failed_validation = [pair for pair in validation_feedback_pairs 
                           if pair["validation_status"] in ["needs_clarification", "needs_regeneration"]]
        
        # Check if validation is too lenient
        passed_low_rating = [pair for pair in passed_validation if pair["user_rating"] <= 2]
        if len(passed_low_rating) >= 2:
            insight = LearningInsight(
                insight_id=f"validation_lenient_{int(time.time())}",
                insight_type="validation_effectiveness",
                description=f"Validation may be too lenient: {len(passed_low_rating)} validated responses received low ratings",
                confidence=0.7,
                evidence_count=len(passed_low_rating),
                recommended_actions=[
                    LearningAction.ENHANCE_VALIDATION_RULES,
                    LearningAction.ADJUST_CONFIDENCE_THRESHOLD
                ],
                affected_components=["validation"],
                priority=3,
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        # Check if validation is too strict
        failed_high_rating = [pair for pair in failed_validation if pair["user_rating"] >= 4]
        if len(failed_high_rating) >= 2:
            insight = LearningInsight(
                insight_id=f"validation_strict_{int(time.time())}",
                insight_type="validation_effectiveness",
                description=f"Validation may be too strict: {len(failed_high_rating)} rejected responses received high ratings",
                confidence=0.6,
                evidence_count=len(failed_high_rating),
                recommended_actions=[
                    LearningAction.ENHANCE_VALIDATION_RULES
                ],
                affected_components=["validation"],
                priority=2,
                timestamp=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _analyze_session_patterns(self,
                                feedback_entries: List[FeedbackEntry],
                                response_traces: List[ResponseTrace],
                                session_context: Dict[str, Any]) -> List[LearningInsight]:
        """Analyze session-level patterns"""
        
        insights = []
        
        # Analyze session satisfaction trend
        session_ratings = [f.rating for f in feedback_entries if f.rating is not None]
        if len(session_ratings) >= 3:
            # Check for declining satisfaction
            first_half = session_ratings[:len(session_ratings)//2]
            second_half = session_ratings[len(session_ratings)//2:]
            
            if first_half and second_half:
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                if first_avg - second_avg > 1.0:  # Significant decline
                    insight = LearningInsight(
                        insight_id=f"session_decline_{int(time.time())}",
                        insight_type="session_pattern",
                        description=f"User satisfaction declining within session (from {first_avg:.1f} to {second_avg:.1f})",
                        confidence=0.7,
                        evidence_count=len(session_ratings),
                        recommended_actions=[
                            LearningAction.MODIFY_RETRIEVAL_STRATEGY,
                            LearningAction.UPDATE_GENERATION_PARAMS
                        ],
                        affected_components=["session_management", "generation"],
                        priority=3,
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
        
        return insights  
  
    def _update_performance_metrics(self,
                                  feedback_entries: List[FeedbackEntry],
                                  response_traces: List[ResponseTrace]):
        """Update performance metrics based on feedback"""
        
        # User satisfaction (based on ratings)
        ratings = [f.rating for f in feedback_entries if f.rating is not None]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            satisfaction = (avg_rating - 1) / 4  # Normalize to 0-1
            self._update_metric("user_satisfaction", satisfaction)
            self._update_metric("average_rating", avg_rating)
        
        # Feedback rate
        total_responses = len(response_traces)
        feedback_rate = len(feedback_entries) / total_responses if total_responses > 0 else 0
        self._update_metric("feedback_rate", feedback_rate)
        
        # Correction rate
        corrections = [f for f in feedback_entries if f.correction]
        correction_rate = len(corrections) / len(feedback_entries) if feedback_entries else 0
        self._update_metric("correction_rate", correction_rate)
        
        # Validation success rate
        validated_responses = [trace for trace in response_traces 
                             if trace.validation_results and trace.validation_results.get("status") == "passed"]
        validation_success_rate = len(validated_responses) / len(response_traces) if response_traces else 0
        self._update_metric("validation_success_rate", validation_success_rate)
    
    def _update_metric(self, metric_name: str, new_value: float):
        """Update a performance metric with trend analysis"""
        
        if metric_name in self.performance_metrics:
            metric = self.performance_metrics[metric_name]
            old_value = metric.current_value
            
            # Update value with learning rate
            updated_value = old_value + self.learning_rate * (new_value - old_value)
            metric.current_value = updated_value
            
            # Determine trend
            if abs(new_value - old_value) < 0.01:
                metric.trend = "stable"
            elif new_value > old_value:
                metric.trend = "improving"
            else:
                metric.trend = "declining"
            
            metric.last_updated = datetime.now()
    
    def _apply_automatic_adjustments(self, insights: List[LearningInsight]):
        """Apply automatic adjustments based on learning insights"""
        
        adjustments_made = []
        
        for insight in insights:
            if insight.confidence >= 0.8 and insight.priority >= 4:
                for action in insight.recommended_actions:
                    adjustment = self._execute_learning_action(action, insight)
                    if adjustment:
                        adjustments_made.append(adjustment)
        
        # Log adjustments
        if adjustments_made:
            self.adjustment_history.extend(adjustments_made)
            logger.info(f"Applied {len(adjustments_made)} automatic adjustments based on feedback learning")
    
    def _execute_learning_action(self, action: LearningAction, insight: LearningInsight) -> Optional[Dict[str, Any]]:
        """Execute a specific learning action"""
        
        adjustment = {
            "action": action.value,
            "insight_id": insight.insight_id,
            "timestamp": datetime.now().isoformat(),
            "confidence": insight.confidence,
            "description": ""
        }
        
        if action == LearningAction.ADJUST_CONFIDENCE_THRESHOLD:
            # Adjust confidence threshold based on calibration issues
            if "overconfidence" in insight.description:
                new_threshold = max(0.1, self.confidence_adjustment_factor - 0.05)
                adjustment["description"] = f"Lowered confidence threshold to {new_threshold}"
                adjustment["parameters"] = {"confidence_threshold": new_threshold}
            elif "underconfidence" in insight.description:
                new_threshold = min(0.9, self.confidence_adjustment_factor + 0.05)
                adjustment["description"] = f"Raised confidence threshold to {new_threshold}"
                adjustment["parameters"] = {"confidence_threshold": new_threshold}
            
            return adjustment
        
        elif action == LearningAction.UPDATE_GENERATION_PARAMS:
            # Adjust generation parameters
            if "incomplete" in insight.description or "unclear" in insight.description:
                adjustment["description"] = "Increased generation length and reduced temperature"
                adjustment["parameters"] = {
                    "max_length": "+20%",
                    "temperature": "-0.1"
                }
            elif "inaccurate" in insight.description:
                adjustment["description"] = "Reduced temperature for more conservative generation"
                adjustment["parameters"] = {"temperature": "-0.15"}
            
            return adjustment
        
        elif action == LearningAction.ENHANCE_VALIDATION_RULES:
            # Enhance validation rules
            if "lenient" in insight.description:
                adjustment["description"] = "Tightened validation rules"
                adjustment["parameters"] = {"validation_threshold": "+0.1"}
            elif "strict" in insight.description:
                adjustment["description"] = "Relaxed validation rules"
                adjustment["parameters"] = {"validation_threshold": "-0.1"}
            
            return adjustment
        
        elif action == LearningAction.MODIFY_RETRIEVAL_STRATEGY:
            # Modify retrieval strategy
            if "outdated" in insight.description:
                adjustment["description"] = "Increased retrieval count and added recency weighting"
                adjustment["parameters"] = {
                    "retrieval_k": "+10",
                    "recency_weight": "+0.2"
                }
            elif "incomplete" in insight.description:
                adjustment["description"] = "Increased retrieval diversity"
                adjustment["parameters"] = {
                    "retrieval_k": "+15",
                    "diversity_threshold": "+0.1"
                }
            
            return adjustment
        
        # For other actions, return placeholder (would need specific implementation)
        adjustment["description"] = f"Action {action.value} identified but not auto-implemented"
        return adjustment
    
    def get_learning_recommendations(self, priority_threshold: int = 3) -> List[Dict[str, Any]]:
        """Get actionable learning recommendations"""
        
        recommendations = []
        
        # Filter insights by priority
        high_priority_insights = [insight for insight in self.learning_insights 
                                if insight.priority >= priority_threshold]
        
        # Group by component
        component_issues = defaultdict(list)
        for insight in high_priority_insights:
            for component in insight.affected_components:
                component_issues[component].append(insight)
        
        # Generate recommendations
        for component, insights in component_issues.items():
            if len(insights) >= 2:  # Multiple issues in same component
                recommendation = {
                    "component": component,
                    "issue_count": len(insights),
                    "priority": max(insight.priority for insight in insights),
                    "description": f"Multiple issues detected in {component}",
                    "insights": [insight.description for insight in insights],
                    "recommended_actions": list(set(action.value for insight in insights 
                                                 for action in insight.recommended_actions)),
                    "confidence": sum(insight.confidence for insight in insights) / len(insights)
                }
                recommendations.append(recommendation)
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
        
        return recommendations
    
    def generate_improvement_plan(self) -> Dict[str, Any]:
        """Generate comprehensive improvement plan based on learning"""
        
        recommendations = self.get_learning_recommendations()
        
        # Categorize improvements
        immediate_actions = []
        short_term_goals = []
        long_term_objectives = []
        
        for rec in recommendations:
            if rec["priority"] >= 4:
                immediate_actions.append(rec)
            elif rec["priority"] >= 3:
                short_term_goals.append(rec)
            else:
                long_term_objectives.append(rec)
        
        # Calculate ROI estimates
        performance_gaps = {}
        for metric_name, metric in self.performance_metrics.items():
            gap = metric.target_value - metric.current_value
            if gap > 0:
                performance_gaps[metric_name] = {
                    "current": metric.current_value,
                    "target": metric.target_value,
                    "gap": gap,
                    "trend": metric.trend
                }
        
        improvement_plan = {
            "generated_at": datetime.now().isoformat(),
            "overall_health": self._calculate_system_health(),
            "performance_gaps": performance_gaps,
            "immediate_actions": immediate_actions,
            "short_term_goals": short_term_goals,
            "long_term_objectives": long_term_objectives,
            "estimated_impact": self._estimate_improvement_impact(recommendations),
            "implementation_priority": self._prioritize_implementations(recommendations)
        }
        
        return improvement_plan
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        
        health_scores = {}
        overall_score = 0
        
        for metric_name, metric in self.performance_metrics.items():
            if metric.target_value > 0:
                score = min(1.0, metric.current_value / metric.target_value)
                health_scores[metric_name] = score
                overall_score += score
        
        overall_score = overall_score / len(self.performance_metrics) if self.performance_metrics else 0
        
        # Determine health status
        if overall_score >= 0.8:
            status = "excellent"
        elif overall_score >= 0.6:
            status = "good"
        elif overall_score >= 0.4:
            status = "fair"
        else:
            status = "needs_attention"
        
        return {
            "overall_score": overall_score,
            "status": status,
            "component_scores": health_scores,
            "trending_up": len([m for m in self.performance_metrics.values() if m.trend == "improving"]),
            "trending_down": len([m for m in self.performance_metrics.values() if m.trend == "declining"])
        }
    
    def _estimate_improvement_impact(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate impact of implementing recommendations"""
        
        impact_estimates = {}
        
        # Simple heuristic-based impact estimation
        for rec in recommendations:
            component = rec["component"]
            priority = rec["priority"]
            confidence = rec["confidence"]
            
            # Base impact on priority and confidence
            estimated_impact = (priority / 5.0) * confidence * 0.2  # Max 20% improvement
            
            # Adjust based on component
            if component in ["generation", "validation"]:
                estimated_impact *= 1.2  # Higher impact for core components
            elif component in ["retrieval", "reranking"]:
                estimated_impact *= 1.1
            
            impact_estimates[component] = impact_estimates.get(component, 0) + estimated_impact
        
        return impact_estimates
    
    def _prioritize_implementations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize implementation order of recommendations"""
        
        # Score each recommendation
        scored_recommendations = []
        
        for rec in recommendations:
            # Implementation difficulty (inverse of ease)
            difficulty_scores = {
                "confidence_estimation": 2,
                "validation": 3,
                "generation": 4,
                "retrieval": 3,
                "reranking": 2,
                "intent_detection": 3,
                "overall": 5
            }
            
            difficulty = difficulty_scores.get(rec["component"], 3)
            
            # Calculate implementation score (higher = more urgent)
            implementation_score = (
                rec["priority"] * 0.4 +
                rec["confidence"] * 0.3 +
                (6 - difficulty) * 0.2 +  # Easier implementations get higher score
                rec["issue_count"] * 0.1
            )
            
            scored_rec = rec.copy()
            scored_rec["implementation_score"] = implementation_score
            scored_rec["estimated_difficulty"] = difficulty
            scored_recommendations.append(scored_rec)
        
        # Sort by implementation score
        scored_recommendations.sort(key=lambda x: x["implementation_score"], reverse=True)
        
        return scored_recommendations
    
    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for analysis and persistence"""
        
        return {
            "export_timestamp": datetime.now().isoformat(),
            "learning_insights": [asdict(insight) for insight in self.learning_insights],
            "performance_metrics": {
                name: asdict(metric) for name, metric in self.performance_metrics.items()
            },
            "adjustment_history": self.adjustment_history,
            "feedback_patterns": {
                "intent_performance": dict(self.feedback_patterns["intent_performance"]),
                "confidence_accuracy": dict(self.feedback_patterns["confidence_accuracy"]),
                "correction_topics": dict(self.feedback_patterns["correction_topics"]),
                "user_preferences": dict(self.feedback_patterns["user_preferences"])
            },
            "system_health": self._calculate_system_health(),
            "improvement_plan": self.generate_improvement_plan()
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        
        recent_insights = [insight for insight in self.learning_insights 
                          if (datetime.now() - insight.timestamp).days <= 7]
        
        return {
            "total_insights": len(self.learning_insights),
            "recent_insights": len(recent_insights),
            "adjustments_made": len(self.adjustment_history),
            "system_health": self._calculate_system_health(),
            "top_issues": [insight.description for insight in 
                          sorted(self.learning_insights, key=lambda x: x.priority, reverse=True)[:3]],
            "performance_trends": {
                name: metric.trend for name, metric in self.performance_metrics.items()
            },
            "learning_rate": self.learning_rate,
            "auto_adjustments_enabled": self.enable_auto_adjustments
        }