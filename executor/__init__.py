"""
Executor/Retriever Layer Package

This package provides parallel retrieval execution with multiple retriever types:
- Dense retrieval using FAISS and embeddings
- Sparse retrieval using TF-IDF/BM25
- Graph retrieval using knowledge graph traversal
- MemoRAG with session cache and context reuse
- Fusion algorithms for combining results

Components:
- RetrievalExecutor: Main orchestrator for parallel retrieval
- DenseRetriever: Embedding-based semantic search
- SparseRetriever: TF-IDF/BM25 keyword search
- GraphRetriever: Knowledge graph traversal
- MemoRAGRetriever: Session-aware cached retrieval
- FusionEngine: Result combination and ranking
- ExecutorOrchestrator: Integration with planner

Algorithms:
- Dense: all-mpnet-base-v2 embeddings with FAISS
- Sparse: TF-IDF with scikit-learn
- Fusion: Reciprocal Rank Fusion (RRF) and weighted sum
- Graph: Neighbor expansion and path traversal
"""

from .retrieval_executor import RetrievalExecutor
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from .graph_retriever import GraphRetriever
from .memorag_retriever import MemoRAGRetriever
from .fusion_engine import FusionEngine
from .executor_orchestrator import ExecutorOrchestrator
from .executor_services import ExecutorServices

__version__ = "1.0.0"
__author__ = "Executor Package"

__all__ = [
    'RetrievalExecutor',
    'DenseRetriever',
    'SparseRetriever', 
    'GraphRetriever',
    'MemoRAGRetriever',
    'FusionEngine',
    'ExecutorOrchestrator',
    'ExecutorServices'
]