"""
Data Ingestion Pipeline Package

This package provides a comprehensive data ingestion pipeline that includes:
- File loading (JSON, CSV, TXT)
- Schema extraction
- LLM-based semantic chunking
- Entity and relation extraction
- Knowledge graph creation
- Embedding generation and FAISS storage

Main Components:
- FileLoader: Load and parse different file formats
- SchemaExtractor: Extract schema information from data
- LLMChunker: Create semantic chunks using LLM guidance
- EntityRelationExtractor: Extract entities and relations for knowledge graphs
- EmbeddingStorage: Create embeddings and store in FAISS index
- DataIngestionPipeline: Main orchestrator for the entire pipeline
"""

from .file_loaders import FileLoader
from .schema_extractor import SchemaExtractor
from .llm_chunker import LLMChunker
from .entity_relation_extractor import EntityRelationExtractor
from .embedding_storage import EmbeddingStorage
from .data_pipeline import DataIngestionPipeline

__version__ = "1.0.0"
__author__ = "Data Ingestion Pipeline"

__all__ = [
    'FileLoader',
    'SchemaExtractor', 
    'LLMChunker',
    'EntityRelationExtractor',
    'EmbeddingStorage',
    'DataIngestionPipeline'
]