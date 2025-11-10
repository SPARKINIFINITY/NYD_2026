"""
Reranker Package - Cross-encoder Cascade System

This package provides a two-stage reranking system that maximizes precision
for answer generation through intelligent result refinement:

Stage 1: Fast cross-encoder reduces ~200 results to ~30
Stage 2: Stronger cross-encoder/LLM refines top 30 to final 3-7

Components:
- CrossEncoderReranker: Fast first-stage reranking
- AdvancedReranker: Stronger second-stage reranking  
- SignalProcessor: Multi-signal integration and scoring
- CascadeReranker: Two-stage orchestrator
- RerankerOrchestrator: Integration with retrieval system

Signals Incorporated:
- dense_score: Semantic similarity from dense retrieval
- bm25_score: Keyword relevance from sparse retrieval
- entity_overlap: Entity matching between query and document
- session_relevance: Context relevance from session history
- cluster_match: Topic/cluster alignment scoring

Architecture:
Query + 200 results → Fast Cross-encoder → 30 results → Strong Reranker → 3-7 final results
"""

from .cross_encoder_reranker import CrossEncoderReranker
from .advanced_reranker import AdvancedReranker
from .signal_processor import SignalProcessor
from .cascade_reranker import CascadeReranker
from .reranker_orchestrator import RerankerOrchestrator

__version__ = "1.0.0"
__author__ = "Reranker Package"

__all__ = [
    'CrossEncoderReranker',
    'AdvancedReranker',
    'SignalProcessor', 
    'CascadeReranker',
    'RerankerOrchestrator'
]