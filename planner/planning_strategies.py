"""
Planning Strategies

Defines different planning strategies for various query types and contexts.
Each strategy determines optimal retrieval parameters and execution approach.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import math

class BasePlanningStrategy(ABC):
    """Base class for planning strategies"""
    
    @abstractmethod
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan for given inputs"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        pass

class FactualQueryStrategy(BasePlanningStrategy):
    """Strategy for factual queries - fast, precise retrieval"""
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        # Base configuration for factual queries
        plan = {
            "retrievers": ["dense", "bm25"],
            "dense_k": 50,
            "bm25_k": 50,
            "rerank": True,
            "multihop": 0,
            "strategy": "factual_query",
            "confidence": 0.9,
            "reasoning": "Factual queries need precise, direct retrieval"
        }
        
        # Adjust based on entity count
        entity_count = len(entities)
        if entity_count > 3:
            # More entities might need broader search
            plan["dense_k"] = 75
            plan["bm25_k"] = 75
        elif entity_count == 0:
            # No entities - might need broader search
            plan["dense_k"] = 100
            plan["bm25_k"] = 100
            plan["confidence"] = 0.7
        
        # Adjust based on query length
        query_length = len(query.split())
        if query_length > 15:
            # Longer queries might be more complex
            plan["dense_k"] = min(plan["dense_k"] + 25, 100)
            plan["bm25_k"] = min(plan["bm25_k"] + 25, 100)
        
        return plan
    
    def get_strategy_name(self) -> str:
        return "factual_query"

class ExplanationStrategy(BasePlanningStrategy):
    """Strategy for explanation requests - comprehensive retrieval"""
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        plan = {
            "retrievers": ["dense", "bm25"],
            "dense_k": 100,
            "bm25_k": 100,
            "rerank": True,
            "multihop": 0,
            "strategy": "explanation",
            "confidence": 0.85,
            "reasoning": "Explanations need comprehensive context"
        }
        
        # Check if explanation might need multi-hop reasoning
        complex_indicators = ["why", "how", "because", "reason", "cause", "effect"]
        if any(indicator in query.lower() for indicator in complex_indicators):
            plan["multihop"] = 1
            plan["dense_k"] = 150
            plan["bm25_k"] = 150
        
        # Adjust for technical explanations
        tech_entities = ["algorithm", "system", "process", "method", "technique"]
        if any(entity.lower() in tech_entities for entity in entities):
            plan["retrievers"].append("graph")
            plan["graph_k"] = 50
        
        return plan
    
    def get_strategy_name(self) -> str:
        return "explanation"

class ComparisonStrategy(BasePlanningStrategy):
    """Strategy for comparison queries - balanced retrieval"""
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        plan = {
            "retrievers": ["dense", "bm25"],
            "dense_k": 100,
            "bm25_k": 100,
            "rerank": True,
            "multihop": 0,
            "strategy": "comparison",
            "confidence": 0.8,
            "reasoning": "Comparisons need balanced information from multiple sources"
        }
        
        # Comparisons often need more comprehensive retrieval
        entity_count = len(entities)
        if entity_count >= 2:
            # Multiple entities to compare
            plan["dense_k"] = 150
            plan["bm25_k"] = 150
            plan["multihop"] = 1  # Might need to connect information
        
        # Check for complex comparisons
        comparison_words = ["versus", "vs", "difference", "compare", "contrast", "better", "worse"]
        if any(word in query.lower() for word in comparison_words):
            plan["retrievers"].append("graph")
            plan["graph_k"] = 75
        
        return plan
    
    def get_strategy_name(self) -> str:
        return "comparison"

class TableDataStrategy(BasePlanningStrategy):
    """Strategy for tabular data requests"""
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        plan = {
            "retrievers": ["dense", "bm25"],
            "dense_k": 75,
            "bm25_k": 75,
            "rerank": True,
            "multihop": 0,
            "strategy": "table_data",
            "confidence": 0.85,
            "reasoning": "Table requests need structured data retrieval"
        }
        
        # Table queries often benefit from BM25 for exact matches
        plan["bm25_k"] = 100  # Increase BM25 weight
        
        # Check for specific data requests
        data_indicators = ["list", "table", "data", "statistics", "numbers", "values"]
        if any(indicator in query.lower() for indicator in data_indicators):
            plan["dense_k"] = 50  # Reduce dense, increase BM25
            plan["bm25_k"] = 150
        
        return plan
    
    def get_strategy_name(self) -> str:
        return "table_data"

class CodeStrategy(BasePlanningStrategy):
    """Strategy for code/programming queries"""
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        plan = {
            "retrievers": ["dense"],
            "dense_k": 75,
            "rerank": True,
            "multihop": 0,
            "strategy": "code",
            "confidence": 0.9,
            "reasoning": "Code queries benefit from semantic similarity"
        }
        
        # Code queries often have specific technical terms
        tech_terms = ["function", "class", "method", "algorithm", "implementation"]
        if any(term in query.lower() for term in tech_terms):
            plan["dense_k"] = 100
        
        # Check for complex programming concepts
        complex_terms = ["architecture", "design pattern", "framework", "system"]
        if any(term in query.lower() for term in complex_terms):
            plan["retrievers"].append("graph")
            plan["graph_k"] = 50
            plan["multihop"] = 1
        
        return plan
    
    def get_strategy_name(self) -> str:
        return "code"

class MultiHopStrategy(BasePlanningStrategy):
    """Strategy for complex multi-hop reasoning queries"""
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        plan = {
            "retrievers": ["dense", "graph"],
            "dense_k": 150,
            "graph_k": 100,
            "rerank": True,
            "multihop": 2,
            "strategy": "multi_hop",
            "confidence": 0.75,
            "reasoning": "Complex queries need multi-step reasoning and graph traversal"
        }
        
        # Adjust multihop depth based on query complexity
        query_length = len(query.split())
        entity_count = len(entities)
        
        complexity_score = (query_length / 10) + (entity_count / 5)
        
        if complexity_score > 3:
            plan["multihop"] = 3
            plan["dense_k"] = 200
            plan["graph_k"] = 150
        elif complexity_score < 1.5:
            plan["multihop"] = 1
            plan["dense_k"] = 100
            plan["graph_k"] = 75
        
        # Add BM25 for comprehensive coverage
        plan["retrievers"].append("bm25")
        plan["bm25_k"] = 100
        
        return plan
    
    def get_strategy_name(self) -> str:
        return "multi_hop"

class ClarificationStrategy(BasePlanningStrategy):
    """Strategy for clarification requests"""
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        plan = {
            "retrievers": ["dense"],
            "dense_k": 30,
            "rerank": True,
            "multihop": 0,
            "strategy": "clarification",
            "confidence": 0.8,
            "reasoning": "Clarifications need focused, contextual retrieval"
        }
        
        # Use context from previous queries
        recent_context = context.get("recent_context", {})
        if recent_context.get("recent_entities"):
            # Include context entities in search
            plan["dense_k"] = 50
            plan["use_context_entities"] = True
        
        # Check if clarification is about complex topic
        if recent_context.get("current_topic") in ["mythology_religion", "science_academic"]:
            plan["dense_k"] = 75
            plan["retrievers"].append("bm25")
            plan["bm25_k"] = 50
        
        return plan
    
    def get_strategy_name(self) -> str:
        return "clarification"

class IrrelevantStrategy(BasePlanningStrategy):
    """Strategy for irrelevant/small talk queries"""
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        plan = {
            "retrievers": [],
            "dense_k": 0,
            "rerank": False,
            "multihop": 0,
            "strategy": "irrelevant",
            "confidence": 0.95,
            "reasoning": "Irrelevant queries don't need retrieval"
        }
        
        return plan
    
    def get_strategy_name(self) -> str:
        return "irrelevant"

class AdaptiveStrategy(BasePlanningStrategy):
    """Adaptive strategy that learns from user preferences and context"""
    
    def __init__(self):
        self.base_strategies = {
            "fact": FactualQueryStrategy(),
            "explain": ExplanationStrategy(),
            "compare": ComparisonStrategy(),
            "table": TableDataStrategy(),
            "code": CodeStrategy(),
            "multi-hop": MultiHopStrategy(),
            "clarify": ClarificationStrategy(),
            "irrelevant": IrrelevantStrategy()
        }
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        # Get base plan from appropriate strategy
        base_strategy = self.base_strategies.get(intent, self.base_strategies["fact"])
        plan = base_strategy.generate_plan(query, intent, entities, context)
        
        # Adapt based on user preferences
        user_prefs = context.get("user_preferences", {})
        self._adapt_to_user_preferences(plan, user_prefs)
        
        # Adapt based on session context
        session_context = context.get("recent_context", {})
        self._adapt_to_session_context(plan, session_context)
        
        # Adapt based on conversation patterns
        patterns = context.get("conversation_patterns", {})
        self._adapt_to_conversation_patterns(plan, patterns)
        
        plan["strategy"] = "adaptive"
        plan["base_strategy"] = base_strategy.get_strategy_name()
        
        return plan
    
    def _adapt_to_user_preferences(self, plan: Dict[str, Any], user_prefs: Dict[str, Any]):
        """Adapt plan based on learned user preferences"""
        preferred_retrievers = user_prefs.get("preferred_retrievers", [])
        avg_k_values = user_prefs.get("avg_k_values", {})
        
        # Adjust retrievers based on preferences
        if preferred_retrievers:
            # Blend preferred retrievers with strategy requirements
            current_retrievers = set(plan.get("retrievers", []))
            preferred_set = set(preferred_retrievers)
            
            # Add preferred retrievers if not conflicting
            if "dense" in preferred_set and "dense" not in current_retrievers:
                plan["retrievers"].append("dense")
                plan["dense_k"] = avg_k_values.get("dense", 50)
            
            if "bm25" in preferred_set and "bm25" not in current_retrievers:
                plan["retrievers"].append("bm25")
                plan["bm25_k"] = avg_k_values.get("bm25", 50)
        
        # Adjust k values based on user history
        for retriever, avg_k in avg_k_values.items():
            k_key = f"{retriever}_k"
            if k_key in plan:
                # Blend current plan with user preference
                current_k = plan[k_key]
                adapted_k = int((current_k + avg_k) / 2)
                plan[k_key] = adapted_k
    
    def _adapt_to_session_context(self, plan: Dict[str, Any], session_context: Dict[str, Any]):
        """Adapt plan based on current session context"""
        current_topic = session_context.get("current_topic")
        conversation_length = session_context.get("conversation_length", 0)
        
        # Adjust for topic continuity
        if current_topic and conversation_length > 3:
            # In deep conversation, might need more context
            for k_key in ["dense_k", "bm25_k", "graph_k"]:
                if k_key in plan:
                    plan[k_key] = min(plan[k_key] + 25, 200)
        
        # Adjust for topic complexity
        if current_topic in ["science_academic", "technology_programming"]:
            if "graph" not in plan.get("retrievers", []):
                plan["retrievers"].append("graph")
                plan["graph_k"] = 50
    
    def _adapt_to_conversation_patterns(self, plan: Dict[str, Any], patterns: Dict[str, Any]):
        """Adapt plan based on detected conversation patterns"""
        detected_patterns = patterns.get("patterns", [])
        session_complexity = patterns.get("session_complexity", 0)
        
        # Adjust for frequent clarifications
        if "frequent_clarifications" in detected_patterns:
            # Increase retrieval breadth to provide more context
            for k_key in ["dense_k", "bm25_k"]:
                if k_key in plan:
                    plan[k_key] = min(plan[k_key] + 50, 200)
            plan["rerank"] = True
        
        # Adjust for complex reasoning sessions
        if "complex_reasoning_session" in detected_patterns or session_complexity > 0.6:
            if plan.get("multihop", 0) == 0:
                plan["multihop"] = 1
            if "graph" not in plan.get("retrievers", []):
                plan["retrievers"].append("graph")
                plan["graph_k"] = 75
    
    def get_strategy_name(self) -> str:
        return "adaptive"

class PlanningStrategies:
    """Main class that manages all planning strategies"""
    
    def __init__(self):
        self.strategies = {
            "fact": FactualQueryStrategy(),
            "explain": ExplanationStrategy(),
            "compare": ComparisonStrategy(),
            "table": TableDataStrategy(),
            "code": CodeStrategy(),
            "multi-hop": MultiHopStrategy(),
            "clarify": ClarificationStrategy(),
            "irrelevant": IrrelevantStrategy(),
            "adaptive": AdaptiveStrategy()
        }
        
        self.default_strategy = "adaptive"
    
    def get_strategy(self, strategy_name: str) -> BasePlanningStrategy:
        """Get strategy by name"""
        return self.strategies.get(strategy_name, self.strategies[self.default_strategy])
    
    def generate_plan(self, query: str, intent: str, entities: List[str], 
                     context: Dict[str, Any], strategy_name: str = None) -> Dict[str, Any]:
        """Generate execution plan using specified or default strategy"""
        
        if strategy_name is None:
            strategy_name = self.default_strategy
        
        strategy = self.get_strategy(strategy_name)
        plan = strategy.generate_plan(query, intent, entities, context)
        
        # Add metadata
        plan["generated_at"] = context.get("timestamp")
        plan["query_length"] = len(query.split())
        plan["entity_count"] = len(entities)
        
        return plan
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.strategies.keys())
    
    def validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize execution plan"""
        validated_plan = plan.copy()
        
        # Ensure required fields
        if "retrievers" not in validated_plan:
            validated_plan["retrievers"] = ["dense"]
        
        if "rerank" not in validated_plan:
            validated_plan["rerank"] = True
        
        if "multihop" not in validated_plan:
            validated_plan["multihop"] = 0
        
        # Validate k values
        for retriever in validated_plan["retrievers"]:
            k_key = f"{retriever}_k"
            if k_key not in validated_plan:
                validated_plan[k_key] = 50  # Default value
            else:
                # Ensure k values are within reasonable bounds
                validated_plan[k_key] = max(1, min(validated_plan[k_key], 500))
        
        # Validate multihop
        validated_plan["multihop"] = max(0, min(validated_plan["multihop"], 5))
        
        # Ensure confidence is between 0 and 1
        if "confidence" in validated_plan:
            validated_plan["confidence"] = max(0.0, min(validated_plan["confidence"], 1.0))
        
        return validated_plan