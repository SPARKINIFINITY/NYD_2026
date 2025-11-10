"""
Planner Services - Unified Interface

This module provides a single, easy-to-use interface for all planner functionality:
- Intelligent query planning and strategy selection
- Session memory management and context tracking
- Execution plan optimization for different goals
- Learning from feedback and user preferences
- Comprehensive planning analytics and insights

Usage:
    from planner.planner_services import PlannerServices
    
    # Initialize services
    planner = PlannerServices()
    
    # Generate execution plan
    plan = planner.generate_plan("What is karma?", "explain", ["karma"])
    
    # Get planning recommendations
    recommendations = planner.get_recommendations("Compare Rama and Krishna", "compare", ["Rama", "Krishna"])
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import all planner components
from .agentic_planner import AgenticPlanner
from .planning_strategies import PlanningStrategies
from .session_memory import SessionMemoryManager
from .plan_optimizer import ExecutionPlanOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlannerServices:
    """
    Unified interface for all planner functionality.
    Provides easy access to planning, optimization, and session management.
    """
    
    def __init__(self, 
                 enable_optimization: bool = True,
                 enable_learning: bool = True,
                 session_timeout: int = 3600,
                 auto_start_session: bool = True):
        """
        Initialize all planner services.
        
        Args:
            enable_optimization: Enable plan optimization
            enable_learning: Enable learning from feedback
            session_timeout: Session timeout in seconds
            auto_start_session: Automatically start session on first use
        """
        self.initialized = False
        self.auto_start_session = auto_start_session
        
        try:
            logger.info("Initializing Planner Services...")
            
            # Initialize main planner
            self.planner = AgenticPlanner(
                enable_optimization=enable_optimization,
                enable_learning=enable_learning,
                session_timeout=session_timeout
            )
            
            # Initialize individual components for direct access
            self.strategies = PlanningStrategies()
            self.session_manager = SessionMemoryManager(session_timeout=session_timeout)
            self.optimizer = ExecutionPlanOptimizer() if enable_optimization else None
            
            # Configuration
            self.config = {
                "optimization_enabled": enable_optimization,
                "learning_enabled": enable_learning,
                "session_timeout": session_timeout
            }
            
            self.initialized = True
            logger.info("✓ All planner services initialized successfully")
            
        except Exception as e:
            logger.error(f"✗ Error initializing planner services: {e}")
            raise
    
    def generate_plan(self, 
                     query: str,
                     intent: str,
                     entities: List[str],
                     optimization_goal: str = "balanced",
                     constraints: Optional[Dict[str, Any]] = None,
                     strategy_override: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive execution plan.
        
        Args:
            query: User query text
            intent: Detected intent (fact, explain, compare, etc.)
            entities: Extracted entities from query
            optimization_goal: Optimization objective (speed, accuracy, balanced, resource_efficient)
            constraints: Resource/performance constraints
            strategy_override: Force specific strategy (optional)
            context: Additional context (optional)
            
        Returns:
            Complete execution plan with metadata
        """
        if not self.initialized:
            logger.warning("Planner services not initialized")
            return self._get_fallback_plan(query, intent, entities)
        
        # Auto-start session if needed
        if self.auto_start_session and not self.planner.session_manager.is_session_active():
            self.planner.start_session()
        
        try:
            plan = self.planner.generate_execution_plan(
                query=query,
                intent=intent,
                entities=entities,
                optimization_goal=optimization_goal,
                constraints=constraints or {},
                strategy_override=strategy_override
            )
            
            # Add service metadata
            plan["service_metadata"] = {
                "generated_by": "planner_services",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return self._get_fallback_plan(query, intent, entities)
    
    def get_recommendations(self, 
                          query: str,
                          intent: str,
                          entities: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive planning recommendations.
        
        Args:
            query: User query text
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Planning recommendations and analysis
        """
        if not self.initialized:
            logger.warning("Planner services not initialized")
            return {"error": "Planner services not initialized"}
        
        try:
            return self.planner.get_planning_recommendations(query, intent, entities)
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {"error": str(e)}
    
    def analyze_complexity(self, 
                          query: str,
                          intent: str,
                          entities: List[str]) -> Dict[str, Any]:
        """
        Analyze query complexity and provide insights.
        
        Args:
            query: User query text
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Complexity analysis and recommendations
        """
        if not self.initialized:
            logger.warning("Planner services not initialized")
            return {"complexity_score": 0.5, "estimated_difficulty": "unknown"}
        
        try:
            return self.planner.analyze_query_complexity(query, intent, entities)
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            return {"error": str(e)}
    
    def optimize_plan(self, 
                     plan: Dict[str, Any],
                     optimization_goal: str = "balanced",
                     constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize an existing execution plan.
        
        Args:
            plan: Execution plan to optimize
            optimization_goal: Optimization objective
            constraints: Resource/performance constraints
            
        Returns:
            Optimized execution plan
        """
        if not self.initialized or not self.optimizer:
            logger.warning("Optimizer not available")
            return plan
        
        try:
            return self.optimizer.optimize_plan(plan, constraints or {}, optimization_goal)
        except Exception as e:
            logger.error(f"Error optimizing plan: {e}")
            return plan
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new planning session.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Session ID
        """
        if not self.initialized:
            logger.warning("Planner services not initialized")
            return "error_session"
        
        try:
            return self.planner.start_session(session_id)
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return "error_session"
    
    def get_session_insights(self) -> Dict[str, Any]:
        """
        Get insights about current session.
        
        Returns:
            Session insights and analytics
        """
        if not self.initialized:
            logger.warning("Planner services not initialized")
            return {"error": "Planner services not initialized"}
        
        try:
            return self.planner.get_session_insights()
        except Exception as e:
            logger.error(f"Error getting session insights: {e}")
            return {"error": str(e)}
    
    def learn_from_feedback(self, 
                           plan: Dict[str, Any],
                           execution_results: Dict[str, Any],
                           user_satisfaction: Optional[float] = None):
        """
        Learn from execution feedback to improve future planning.
        
        Args:
            plan: Execution plan that was used
            execution_results: Results from plan execution
            user_satisfaction: User satisfaction score (0-1)
        """
        if not self.initialized or not self.config["learning_enabled"]:
            return
        
        try:
            self.planner.learn_from_execution_feedback(plan, execution_results, user_satisfaction)
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available planning strategies.
        
        Returns:
            List of strategy names
        """
        if not self.initialized:
            return ["adaptive"]
        
        try:
            return self.strategies.get_available_strategies()
        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            return ["adaptive"]
    
    def validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize an execution plan.
        
        Args:
            plan: Plan to validate
            
        Returns:
            Validated plan
        """
        if not self.initialized:
            return plan
        
        try:
            return self.strategies.validate_plan(plan)
        except Exception as e:
            logger.error(f"Error validating plan: {e}")
            return plan
    
    def estimate_execution_time(self, plan: Dict[str, Any]) -> float:
        """
        Estimate execution time for a plan.
        
        Args:
            plan: Execution plan
            
        Returns:
            Estimated execution time in seconds
        """
        if not self.initialized or not self.optimizer:
            return 2.0  # Default estimate
        
        try:
            return self.optimizer._estimate_execution_time(plan)
        except Exception as e:
            logger.error(f"Error estimating execution time: {e}")
            return 2.0
    
    def get_optimization_recommendations(self, 
                                       query_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimization recommendations based on query characteristics.
        
        Args:
            query_characteristics: Query analysis results
            
        Returns:
            Optimization recommendations
        """
        if not self.initialized or not self.optimizer:
            return {"suggested_goal": "balanced", "reasoning": []}
        
        try:
            return self.optimizer.get_optimization_recommendations(query_characteristics)
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {e}")
            return {"suggested_goal": "balanced", "reasoning": []}
    
    def batch_generate_plans(self, 
                           queries_data: List[Dict[str, Any]],
                           default_optimization_goal: str = "balanced") -> List[Dict[str, Any]]:
        """
        Generate plans for multiple queries in batch.
        
        Args:
            queries_data: List of query data dicts with 'query', 'intent', 'entities'
            default_optimization_goal: Default optimization goal
            
        Returns:
            List of execution plans
        """
        plans = []
        
        logger.info(f"Generating plans for {len(queries_data)} queries in batch...")
        
        for i, query_data in enumerate(queries_data, 1):
            logger.info(f"Processing query {i}/{len(queries_data)}")
            
            plan = self.generate_plan(
                query=query_data.get("query", ""),
                intent=query_data.get("intent", "unknown"),
                entities=query_data.get("entities", []),
                optimization_goal=query_data.get("optimization_goal", default_optimization_goal),
                constraints=query_data.get("constraints"),
                strategy_override=query_data.get("strategy_override")
            )
            
            plans.append(plan)
        
        logger.info("Batch plan generation completed")
        return plans
    
    def get_planner_status(self) -> Dict[str, Any]:
        """
        Get current planner status and health.
        
        Returns:
            Planner status information
        """
        if not self.initialized:
            return {
                "status": "not_initialized",
                "error": "Planner services not initialized"
            }
        
        try:
            status = self.planner.get_planner_status()
            status.update({
                "services_version": "1.0.0",
                "configuration": self.config,
                "initialization_status": "success"
            })
            return status
        except Exception as e:
            logger.error(f"Error getting planner status: {e}")
            return {"status": "error", "error": str(e)}
    
    def export_session_data(self) -> Dict[str, Any]:
        """
        Export comprehensive session and planning data.
        
        Returns:
            Exportable session data
        """
        if not self.initialized:
            return {"error": "Planner services not initialized"}
        
        try:
            return self.planner.export_planning_data()
        except Exception as e:
            logger.error(f"Error exporting session data: {e}")
            return {"error": str(e)}
    
    def load_session_data(self, session_data: Dict[str, Any]) -> bool:
        """
        Load session data from previous sessions.
        
        Args:
            session_data: Previously exported session data
            
        Returns:
            Success status
        """
        if not self.initialized:
            logger.warning("Planner services not initialized")
            return False
        
        try:
            self.planner.load_planning_data(session_data)
            return True
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return False
    
    def get_planning_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive planning analytics and statistics.
        
        Returns:
            Planning analytics data
        """
        if not self.initialized:
            return {"error": "Planner services not initialized"}
        
        try:
            analytics = {
                "planner_status": self.get_planner_status(),
                "session_insights": self.get_session_insights(),
                "available_strategies": self.get_available_strategies()
            }
            
            if self.optimizer:
                analytics["optimization_stats"] = self.optimizer.get_optimization_stats()
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting planning analytics: {e}")
            return {"error": str(e)}
    
    def _get_fallback_plan(self, query: str, intent: str, entities: List[str]) -> Dict[str, Any]:
        """Generate a basic fallback plan when services are unavailable."""
        return {
            "retrievers": ["dense", "bm25"],
            "dense_k": 50,
            "bm25_k": 50,
            "rerank": True,
            "multihop": 1 if intent in ["multi-hop", "compare", "explain"] else 0,
            "strategy": "fallback",
            "confidence": 0.5,
            "reasoning": "Fallback plan - planner services unavailable",
            "service_metadata": {
                "generated_by": "fallback_planner",
                "timestamp": datetime.now().isoformat(),
                "warning": "Full planner services not available"
            }
        }
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy performance data
        """
        if not self.initialized:
            return {"error": "Planner services not initialized"}
        
        try:
            # Extract performance data from session memory and optimizer
            session_summary = self.session_manager.get_session_summary()
            user_prefs = session_summary.get("user_preferences", {})
            
            strategy_success = user_prefs.get("successful_strategies", {}).get(strategy_name, 0)
            
            performance_data = {
                "strategy_name": strategy_name,
                "usage_count": strategy_success,
                "success_rate": "unknown",  # Would need more detailed tracking
                "average_confidence": "unknown",
                "recommended_for": self._get_strategy_recommendations(strategy_name)
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {"error": str(e)}
    
    def _get_strategy_recommendations(self, strategy_name: str) -> List[str]:
        """Get recommendations for when to use a specific strategy."""
        recommendations = {
            "fact": ["Simple factual queries", "Quick lookups", "Direct questions"],
            "explain": ["Explanation requests", "How/why questions", "Concept clarification"],
            "compare": ["Comparison queries", "Versus questions", "Difference analysis"],
            "table": ["Data requests", "List queries", "Structured information"],
            "code": ["Programming questions", "Technical implementation", "Code examples"],
            "multi-hop": ["Complex reasoning", "Multi-step questions", "Deep analysis"],
            "clarify": ["Follow-up questions", "Clarification requests", "Context-dependent queries"],
            "adaptive": ["General queries", "Mixed complexity", "Learning from user patterns"],
            "irrelevant": ["Off-topic queries", "Small talk", "Non-informational requests"]
        }
        
        return recommendations.get(strategy_name, ["General purpose"])


# Convenience functions for quick access
def quick_plan(query: str, intent: str, entities: List[str]) -> Dict[str, Any]:
    """Quick plan generation without initializing full services."""
    try:
        strategies = PlanningStrategies()
        context = {"timestamp": datetime.now().isoformat()}
        return strategies.generate_plan(query, intent, entities, context)
    except Exception as e:
        logger.error(f"Quick planning failed: {e}")
        return {
            "retrievers": ["dense"],
            "dense_k": 50,
            "rerank": True,
            "multihop": 0,
            "strategy": "quick_fallback",
            "confidence": 0.5
        }

def quick_optimize(plan: Dict[str, Any], goal: str = "balanced") -> Dict[str, Any]:
    """Quick plan optimization without initializing full services."""
    try:
        optimizer = ExecutionPlanOptimizer()
        return optimizer.optimize_plan(plan, {}, goal)
    except Exception as e:
        logger.error(f"Quick optimization failed: {e}")
        return plan

def quick_complexity(query: str, intent: str, entities: List[str]) -> float:
    """Quick complexity analysis without initializing full services."""
    try:
        query_length = len(query.split())
        entity_count = len(entities)
        
        # Simple complexity calculation
        length_score = min(query_length / 20, 1.0)
        entity_score = min(entity_count / 5, 1.0)
        
        intent_complexity = {
            "fact": 0.2, "explain": 0.6, "compare": 0.8, "table": 0.4,
            "code": 0.5, "multi-hop": 1.0, "clarify": 0.3, "irrelevant": 0.1
        }
        
        intent_score = intent_complexity.get(intent, 0.5)
        complexity = (length_score * 0.4) + (entity_score * 0.3) + (intent_score * 0.3)
        
        return min(complexity, 1.0)
        
    except Exception as e:
        logger.error(f"Quick complexity analysis failed: {e}")
        return 0.5


# Example usage and testing
if __name__ == "__main__":
    # Initialize services
    planner = PlannerServices()
    
    # Test queries with different intents
    test_cases = [
        {"query": "Who is Rama's father?", "intent": "fact", "entities": ["Rama", "father"]},
        {"query": "Explain the concept of karma", "intent": "explain", "entities": ["karma"]},
        {"query": "Compare Rama and Krishna", "intent": "compare", "entities": ["Rama", "Krishna"]},
        {"query": "Show me Hindu deities in a table", "intent": "table", "entities": ["Hindu", "deities"]},
        {"query": "Python code for sorting a list", "intent": "code", "entities": ["Python", "sorting", "list"]},
        {"query": "Why did Rama go to exile and how did it affect Ayodhya?", "intent": "multi-hop", "entities": ["Rama", "exile", "Ayodhya"]}
    ]
    
    print("=" * 80)
    print("PLANNER SERVICES DEMONSTRATION")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        intent = test_case["intent"]
        entities = test_case["entities"]
        
        print(f"\n[{i}] Processing: '{query}'")
        print("-" * 60)
        
        # Generate plan
        plan = planner.generate_plan(query, intent, entities)
        
        print(f"Strategy: {plan.get('strategy', 'unknown')}")
        print(f"Retrievers: {plan.get('retrievers', [])}")
        print(f"K Values: {[(r, plan.get(f'{r}_k', 0)) for r in plan.get('retrievers', [])]}")
        print(f"Multihop: {plan.get('multihop', 0)}")
        print(f"Rerank: {plan.get('rerank', False)}")
        print(f"Confidence: {plan.get('confidence', 0):.3f}")
        
        # Get recommendations
        recommendations = planner.get_recommendations(query, intent, entities)
        complexity = recommendations.get("complexity_analysis", {})
        print(f"Complexity: {complexity.get('complexity_score', 0):.3f} ({complexity.get('estimated_difficulty', 'unknown')})")
        
        # Estimate execution time
        exec_time = planner.estimate_execution_time(plan)
        print(f"Estimated Time: {exec_time:.1f}s")
    
    # Show session insights
    print(f"\n" + "=" * 60)
    print("SESSION INSIGHTS")
    print("=" * 60)
    
    insights = planner.get_session_insights()
    if "session_overview" in insights:
        overview = insights["session_overview"]
        print(f"Session ID: {overview.get('session_id', 'unknown')}")
        print(f"Total Queries: {overview.get('total_queries', 0)}")
        print(f"Duration: {overview.get('duration_minutes', 0):.1f} minutes")
        
        patterns = insights.get("query_patterns", {})
        print(f"Intent Distribution: {patterns.get('intent_distribution', {})}")
    
    print(f"\n✅ Demonstration completed!")