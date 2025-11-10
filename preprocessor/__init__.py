"""
Preprocessor Package

This package provides query preprocessing functionality including:
- Query normalization using LLM
- Intent detection with predefined categories
- Entity and relation extraction using spaCy and LLM

Components:
- DynamicQueryNormalizer: Normalizes and rewrites queries
- UniversalIntentDetector: Detects query intent from predefined categories
- Entity extraction functions: Extract entities and relations from queries

Intent Categories:
- fact: Factual questions
- explain: Explanation requests  
- compare: Comparison requests
- table: Tabular data requests
- code: Code/programming requests
- multi-hop: Complex multi-step reasoning
- clarify: Clarification requests
- irrelevant: Small talk, greetings, off-topic
"""

from .normalizer import DynamicQueryNormalizer
from .intent_detector import UniversalIntentDetector, INTENTS, RETRIEVAL_CONFIGS
from .entity_extractor import extract_entities_and_relations, extract_entities

__version__ = "1.0.0"
__author__ = "Preprocessor Package"

__all__ = [
    'DynamicQueryNormalizer',
    'UniversalIntentDetector', 
    'INTENTS',
    'RETRIEVAL_CONFIGS',
    'extract_entities_and_relations',
    'extract_entities'
]