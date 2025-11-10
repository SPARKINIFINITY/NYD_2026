"""
Generator Package - LLM-based Answer Generation

This package provides grounded answer generation using open-source LLMs
with structured JSON output, proper citations, and confidence scoring.

Components:
- AnswerGenerator: Main LLM-based answer generation
- PromptTemplateManager: Manages prompt templates for different query types
- CitationManager: Handles reference extraction and formatting
- ConfidenceEstimator: Estimates answer confidence based on evidence
- OutputFormatter: Ensures structured JSON output format
- GeneratorOrchestrator: Integration with retrieval and reranking systems

Output Format:
{
    "summary": "Brief answer summary",
    "detailed_answer": "Comprehensive grounded answer with citations",
    "references": [
        {
            "doc_id": "doc_1",
            "span": "relevant text span",
            "relevance_score": 0.95
        }
    ],
    "confidence": 0.87,
    "metadata": {
        "model_used": "microsoft/DialoGPT-medium",
        "generation_time": 2.34,
        "evidence_count": 3
    }
}

Supported Open-Source Models:
- microsoft/DialoGPT-medium/large
- facebook/blenderbot-400M-distill
- microsoft/GODEL-v1_1-base-seq2seq
- google/flan-t5-base/large
- EleutherAI/gpt-neo-1.3B/2.7B
"""

try:
    from .answer_generator import AnswerGenerator
    from .prompt_template_manager import PromptTemplateManager
    from .citation_manager import CitationManager
    from .confidence_estimator import ConfidenceEstimator
    from .output_formatter import OutputFormatter
    from .generator_orchestrator import GeneratorOrchestrator
except ImportError as e:
    # Handle missing dependencies gracefully
    import logging
    logging.warning(f"Some generator components may not be available: {e}")
    
    # Create placeholder classes if imports fail
    class AnswerGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError("AnswerGenerator requires transformers library")
    
    class PromptTemplateManager:
        def __init__(self, *args, **kwargs):
            pass
    
    class CitationManager:
        def __init__(self, *args, **kwargs):
            pass
    
    class ConfidenceEstimator:
        def __init__(self, *args, **kwargs):
            pass
    
    class OutputFormatter:
        def __init__(self, *args, **kwargs):
            pass
    
    class GeneratorOrchestrator:
        def __init__(self, *args, **kwargs):
            raise ImportError("GeneratorOrchestrator requires all generator components")

__version__ = "1.0.0"
__author__ = "Generator Package"

__all__ = [
    'AnswerGenerator',
    'PromptTemplateManager',
    'CitationManager',
    'ConfidenceEstimator',
    'OutputFormatter',
    'GeneratorOrchestrator'
]