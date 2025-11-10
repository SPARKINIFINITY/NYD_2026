"""
Planner Package - Agentic Controller

This package provides intelligent planning capabilities for query execution:
- Analyzes query intent, entities, and session context
- Generates optimal retrieval execution plans
- Adapts strategies based on query complexity and type
- Manages multi-hop reasoning and graph traversal decisions

Components:
- AgenticPlanner: Main planning controller
- PlanningStrategies: Different planning approaches
- SessionMemoryManager: Manages conversation context
- ExecutionPlanOptimizer: Optimizes retrieval parameters

Plan Output Format:
{
    "retrievers": ["dense", "bm25", "graph"],
    "dense_k": 100,
    "bm25_k": 100,
    "graph_k": 50,
    "rerank": true,
    "multihop": 2,
    "strategy": "complex_reasoning",
    "confidence": 0.85
}
"""

from .agentic_planner import AgenticPlanner
from .planning_strategies import PlanningStrategies
from .session_memory import SessionMemoryManager
from .plan_optimizer import ExecutionPlanOptimizer

__version__ = "1.0.0"
__author__ = "Planner Package"

__all__ = [
    'AgenticPlanner',
    'PlanningStrategies', 
    'SessionMemoryManager',
    'ExecutionPlanOptimizer'
]