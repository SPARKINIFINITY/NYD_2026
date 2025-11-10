"""
Agentic Planner - Main Controller

The main planning controller that orchestrates query analysis, strategy selection,
plan generation, and optimization for intelligent retrieval execution.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging

from .session_memory import SessionMemoryManager
from .planning_strategies import PlanningStrategies
from .plan_optimizer import ExecutionPlanOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgenticPlanner:
    """
    Main Agentic Planning Controller
    
    Coordinates all planning components to generate optimal execution plans
    based on query analysis, session context, and learned preferences.
    """
    
    def __init__(self, 
                 enable_optimization: bool = True,
                 enable_learning: bool = True,
                 session_timeout: int = 3600):
        
        # Initialize components
        self.session_manager = SessionMemoryManager(session_timeout=session_timeout)
        self.strategies = PlanningStrategies()
        self.optimizer = ExecutionPlanOptimizer() if enable_optimization else None
        
        # Configuration
        self.enable_optimization = enable_optimization
        self.enable_learning = enable_learning
        
        # Planning statistics
        self.planning_stats = {
            "total_plans_generated": 0,
            "strategy_usage": {},
            "optimization_usage": {},
            "average_plan_confidence": 0.0,
            "session_count": 0
        }
        
        # Current session
        self.current_session_id = None
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new planning session"""
        session_id = self.session_manager.start_session(session_id)
        self.current_session_id = session_id
        self.planning_stats["session_count"] += 1
        
        logger.info(f"Started planning session: {session_id}")
        return session_id
    
    def generate_execution_plan(self, 
                              query: str,
                              intent: str,
                              entities: List[str],
                              optimization_goal: str = "balanced",
                              constraints: Dict[str, Any] = None,
                              strategy_override: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive execution plan
        
        Args:
            query: User query text
            intent: Detected intent (fact, explain, compare, etc.)
            entities: Extracted entities from query
            optimization_goal: Optimization objective (speed, accuracy, balanced, resource_efficient)
            constraints: Resource/performance constraints
            strategy_override: Force specific strategy (optional)
        
        Returns:
            Complete execution plan with metadata
        """
        
        # Ensure session is active
        if not self.session_manager.is_session_active():
            self.start_session()
        
        # Gather context
        context = self._gather_planning_context(query, intent, entities)
        
        # Generate base plan using strategies
        strategy_name = strategy_override or "adaptive"
        base_plan = self.strategies.generate_plan(query, intent, entities, context, strategy_name)
        
        # Optimize plan if enabled
        if self.enable_optimization and self.optimizer:
            optimized_plan = self.optimizer.optimize_plan(
                base_plan, 
                constraints or {}, 
                optimization_goal
            )
        else:
            optimized_plan = base_plan
        
        # Validate and finalize plan
        final_plan = self.strategies.validate_plan(optimized_plan)
        
        # Add comprehensive metadata
        final_plan.update({
            "planner_metadata": {
                "session_id": self.current_session_id,
                "generated_at": datetime.now().isoformat(),
                "query": query,
                "intent": intent,
                "entities": entities,
                "optimization_goal": optimization_goal,
                "strategy_used": strategy_name,
                "optimization_applied": self.enable_optimization,
                "context_used": bool(context.get("recent_context", {}).get("recent_queries"))
            }
        })
        
        # Update session memory
        self.session_manager.add_query_context(query, intent, entities, final_plan)
        
        # Update statistics
        self._update_planning_stats(final_plan, strategy_name, optimization_goal)
        
        logger.info(f"Generated execution plan for query: '{query[:50]}...' with strategy: {strategy_name}")
        
        return final_plan
    
    def _gather_planning_context(self, query: str, intent: str, entities: List[str]) -> Dict[str, Any]:
        """Gather comprehensive context for planning"""
        
        context = {
            "timestamp": datetime.now().isoformat(),
            "query_characteristics": {
                "query": query,
                "intent": intent,
                "entities": entities,
                "query_length": len(query.split()),
                "entity_count": len(entities)
            }
        }
        
        # Add session context
        if self.session_manager.is_session_active():
            context["recent_context"] = self.session_manager.get_recent_context()
            context["user_preferences"] = self.session_manager.get_user_preferences()
            context["conversation_patterns"] = self.session_manager.detect_conversation_patterns()
            context["session_summary"] = self.session_manager.get_session_summary()
        
        return context
    
    def _update_planning_stats(self, plan: Dict[str, Any], strategy_name: str, optimization_goal: str):
        """Update planning statistics"""
        self.planning_stats["total_plans_generated"] += 1
        
        # Update strategy usage
        if strategy_name not in self.planning_stats["strategy_usage"]:
            self.planning_stats["strategy_usage"][strategy_name] = 0
        self.planning_stats["strategy_usage"][strategy_name] += 1
        
        # Update optimization usage
        if optimization_goal not in self.planning_stats["optimization_usage"]:
            self.planning_stats["optimization_usage"][optimization_goal] = 0
        self.planning_stats["optimization_usage"][optimization_goal] += 1
        
        # Update average confidence
        confidence = plan.get("confidence", 0.5)
        total_plans = self.planning_stats["total_plans_generated"]
        current_avg = self.planning_stats["average_plan_confidence"]
        self.planning_stats["average_plan_confidence"] = (
            (current_avg * (total_plans - 1) + confidence) / total_plans
        )
    
    def analyze_query_complexity(self, query: str, intent: str, entities: List[str]) -> Dict[str, Any]:
        """Analyze query complexity and provide planning recommendations"""
        
        analysis = {
            "complexity_score": 0.0,
            "complexity_factors": [],
            "recommended_strategy": "adaptive",
            "recommended_optimization": "balanced",
            "estimated_difficulty": "medium"
        }
        
        # Calculate complexity score
        query_length = len(query.split())
        entity_count = len(entities)
        
        # Base complexity from length and entities
        length_score = min(query_length / 20, 1.0)  # Normalize to 0-1
        entity_score = min(entity_count / 5, 1.0)   # Normalize to 0-1
        
        complexity_score = (length_score * 0.4) + (entity_score * 0.3)
        
        # Intent-based complexity
        intent_complexity = {
            "fact": 0.2,
            "explain": 0.6,
            "compare": 0.8,
            "table": 0.4,
            "code": 0.5,
            "multi-hop": 1.0,
            "clarify": 0.3,
            "irrelevant": 0.1
        }
        
        intent_score = intent_complexity.get(intent, 0.5)
        complexity_score += intent_score * 0.3
        
        analysis["complexity_score"] = min(complexity_score, 1.0)
        
        # Identify complexity factors
        if query_length > 15:
            analysis["complexity_factors"].append("long_query")
        if entity_count > 3:
            analysis["complexity_factors"].append("many_entities")
        if intent in ["multi-hop", "compare", "explain"]:
            analysis["complexity_factors"].append("complex_intent")
        
        # Determine difficulty level
        if complexity_score < 0.3:
            analysis["estimated_difficulty"] = "easy"
            analysis["recommended_optimization"] = "speed"
        elif complexity_score > 0.7:
            analysis["estimated_difficulty"] = "hard"
            analysis["recommended_optimization"] = "accuracy"
            analysis["recommended_strategy"] = "multi-hop" if intent == "multi-hop" else "adaptive"
        
        # Add session context if available
        if self.session_manager.is_session_active():
            patterns = self.session_manager.detect_conversation_patterns()
            if "complex_reasoning_session" in patterns.get("patterns", []):
                analysis["complexity_factors"].append("complex_session")
                analysis["recommended_optimization"] = "accuracy"
        
        return analysis
    
    def get_planning_recommendations(self, query: str, intent: str, entities: List[str]) -> Dict[str, Any]:
        """Get comprehensive planning recommendations"""
        
        # Analyze query complexity
        complexity_analysis = self.analyze_query_complexity(query, intent, entities)
        
        # Get optimizer recommendations if available
        optimizer_recommendations = {}
        if self.optimizer:
            query_characteristics = {
                "query_length": len(query.split()),
                "entity_count": len(entities),
                "intent": intent
            }
            optimizer_recommendations = self.optimizer.get_optimization_recommendations(query_characteristics)
        
        # Combine recommendations
        recommendations = {
            "complexity_analysis": complexity_analysis,
            "strategy_recommendation": {
                "primary": complexity_analysis["recommended_strategy"],
                "alternatives": self._get_alternative_strategies(intent),
                "reasoning": f"Based on intent '{intent}' and complexity score {complexity_analysis['complexity_score']:.2f}"
            },
            "optimization_recommendation": {
                "goal": optimizer_recommendations.get("suggested_goal", complexity_analysis["recommended_optimization"]),
                "constraints": optimizer_recommendations.get("suggested_constraints", {}),
                "reasoning": optimizer_recommendations.get("reasoning", [])
            },
            "execution_estimates": self._estimate_execution_characteristics(complexity_analysis)
        }
        
        return recommendations
    
    def _get_alternative_strategies(self, intent: str) -> List[str]:
        """Get alternative strategies for given intent"""
        alternatives = {
            "fact": ["factual_query", "adaptive"],
            "explain": ["explanation", "adaptive", "multi_hop"],
            "compare": ["comparison", "adaptive", "multi_hop"],
            "table": ["table_data", "adaptive"],
            "code": ["code", "adaptive"],
            "multi-hop": ["multi_hop", "adaptive", "explanation"],
            "clarify": ["clarification", "adaptive"],
            "irrelevant": ["irrelevant"]
        }
        
        return alternatives.get(intent, ["adaptive"])
    
    def _estimate_execution_characteristics(self, complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate execution characteristics based on complexity"""
        
        complexity_score = complexity_analysis["complexity_score"]
        
        # Estimate execution time (in seconds)
        base_time = 1.0 + (complexity_score * 3.0)  # 1-4 seconds range
        
        # Estimate resource usage (relative scale 1-10)
        resource_usage = 3 + (complexity_score * 5)  # 3-8 range
        
        # Estimate accuracy potential (0-1 scale)
        accuracy_potential = 0.7 + (complexity_score * 0.2)  # 0.7-0.9 range
        
        return {
            "estimated_execution_time_seconds": round(base_time, 1),
            "estimated_resource_usage": round(resource_usage, 1),
            "accuracy_potential": round(accuracy_potential, 2),
            "recommended_timeout": round(base_time * 2, 1)  # 2x estimated time
        }
    
    def learn_from_execution_feedback(self, 
                                    plan: Dict[str, Any], 
                                    execution_results: Dict[str, Any],
                                    user_satisfaction: float = None):
        """Learn from execution feedback to improve future planning"""
        
        if not self.enable_learning:
            return
        
        # Extract performance metrics
        performance_metrics = {
            "success": execution_results.get("success", False),
            "execution_time": execution_results.get("execution_time", 0),
            "results_count": execution_results.get("results_count", 0),
            "user_satisfaction": user_satisfaction,
            "timestamp": datetime.now().isoformat()
        }
        
        # Learn in session memory
        self.session_manager.learn_from_feedback(
            plan, 
            performance_metrics["success"], 
            user_satisfaction
        )
        
        # Learn in optimizer
        if self.optimizer:
            self.optimizer.learn_from_feedback(plan, performance_metrics)
        
        logger.info(f"Learned from execution feedback: success={performance_metrics['success']}")
    
    def get_session_insights(self) -> Dict[str, Any]:
        """Get insights about current session"""
        
        if not self.session_manager.is_session_active():
            return {"error": "No active session"}
        
        session_summary = self.session_manager.get_session_summary()
        
        insights = {
            "session_overview": {
                "session_id": session_summary["session_id"],
                "duration_minutes": round(session_summary.get("duration_seconds", 0) / 60, 1),
                "total_queries": session_summary["statistics"]["total_queries"],
                "is_active": session_summary["is_active"]
            },
            "query_patterns": {
                "intent_distribution": dict(session_summary["statistics"]["intent_distribution"]),
                "avg_entities_per_query": round(session_summary["statistics"]["avg_entities_per_query"], 1),
                "multihop_percentage": round(
                    (session_summary["statistics"]["multihop_queries"] / 
                     max(session_summary["statistics"]["total_queries"], 1)) * 100, 1
                )
            },
            "conversation_analysis": session_summary["conversation_patterns"],
            "user_preferences": session_summary["user_preferences"],
            "recommendations": self._generate_session_recommendations(session_summary)
        }
        
        return insights
    
    def _generate_session_recommendations(self, session_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on session analysis"""
        
        recommendations = []
        stats = session_summary["statistics"]
        patterns = session_summary["conversation_patterns"]
        
        # Analyze query patterns
        if stats["clarification_requests"] > stats["total_queries"] * 0.3:
            recommendations.append("Consider providing more detailed initial responses")
        
        if stats["multihop_queries"] > stats["total_queries"] * 0.5:
            recommendations.append("User prefers complex reasoning - enable advanced retrieval by default")
        
        # Analyze conversation patterns
        if "frequent_clarifications" in patterns.get("patterns", []):
            recommendations.append("Increase context window for better understanding")
        
        if "topic_switching" in patterns.get("patterns", []):
            recommendations.append("Provide topic transition summaries")
        
        # Analyze preferences
        prefs = session_summary["user_preferences"]
        if prefs["preferred_retrievers"]:
            top_retriever = prefs["preferred_retrievers"][0]
            recommendations.append(f"User shows preference for {top_retriever} retriever")
        
        return recommendations
    
    def export_planning_data(self) -> Dict[str, Any]:
        """Export comprehensive planning data for analysis"""
        
        export_data = {
            "planner_stats": self.planning_stats,
            "session_data": self.session_manager.export_session_data(),
            "strategies_available": self.strategies.get_available_strategies(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        if self.optimizer:
            export_data["optimization_stats"] = self.optimizer.get_optimization_stats()
        
        return export_data
    
    def load_planning_data(self, planning_data: Dict[str, Any]):
        """Load planning data from previous sessions"""
        
        # Load session data
        if "session_data" in planning_data:
            self.session_manager.load_session_data(planning_data["session_data"])
        
        # Load planning stats
        if "planner_stats" in planning_data:
            self.planning_stats.update(planning_data["planner_stats"])
        
        logger.info("Loaded planning data from previous session")
    
    def get_planner_status(self) -> Dict[str, Any]:
        """Get current planner status and health"""
        
        return {
            "planner_health": "healthy",
            "active_session": self.current_session_id,
            "session_active": self.session_manager.is_session_active(),
            "components_status": {
                "session_manager": "active",
                "strategies": "active",
                "optimizer": "active" if self.optimizer else "disabled"
            },
            "configuration": {
                "optimization_enabled": self.enable_optimization,
                "learning_enabled": self.enable_learning
            },
            "statistics": self.planning_stats
        }