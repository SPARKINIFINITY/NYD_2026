"""
Preprocessor Services - Unified Interface

This module provides a single, easy-to-use interface for all preprocessor functionality:
- Query normalization using LLM
- Intent detection with confidence scoring
- Entity and relation extraction with multiple strategies
- Comprehensive preprocessing pipeline

Usage:
    from preprocessor.preprocessor_services import PreprocessorServices
    
    # Initialize services
    services = PreprocessorServices()
    
    # Process a single query
    result = services.process_query("rama father king")
    
    # Get specific components
    normalized = services.normalize_query("rama father king")
    intent = services.detect_intent("What is karma?")
    entities = services.extract_entities("Rama defeated Ravana in Lanka")
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import all preprocessor components
from .normalizer import DynamicQueryNormalizer
from .intent_detector import UniversalIntentDetector, INTENTS, RETRIEVAL_CONFIGS
from .entity_extractor import (
    extract_entities_and_relations_comprehensive,
    extract_entities_comprehensive,
    extract_entities,
    get_entity_statistics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessorServices:
    """
    Unified interface for all preprocessor functionality.
    Provides easy access to normalization, intent detection, and entity extraction.
    """
    
    def __init__(self, 
                 normalizer_model: str = "google/flan-t5-large",
                 intent_model: str = "all-mpnet-base-v2",
                 intent_threshold: float = 0.4):
        """
        Initialize all preprocessor services.
        
        Args:
            normalizer_model: Model name for query normalization
            intent_model: Model name for intent detection
            intent_threshold: Confidence threshold for intent detection
        """
        self.initialized = False
        self.normalizer = None
        self.intent_detector = None
        
        try:
            logger.info("Initializing Preprocessor Services...")
            
            # Initialize normalizer
            logger.info("Loading query normalizer...")
            self.normalizer = DynamicQueryNormalizer(model_name=normalizer_model)
            
            # Initialize intent detector
            logger.info("Loading intent detector...")
            self.intent_detector = UniversalIntentDetector(
                INTENTS, 
                RETRIEVAL_CONFIGS, 
                model_name=intent_model,
                threshold=intent_threshold
            )
            
            self.initialized = True
            logger.info("✓ All preprocessor services initialized successfully")
            
        except Exception as e:
            logger.error(f"✗ Error initializing preprocessor services: {e}")
            raise
    
    def normalize_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Normalize a query into clear, grammatically correct English.
        
        Args:
            query: Raw user query
            context: Optional context for normalization
            
        Returns:
            Normalized query string
        """
        if not self.initialized or not self.normalizer:
            logger.warning("Normalizer not initialized, returning original query")
            return query
            
        try:
            return self.normalizer.normalize(query, context)
        except Exception as e:
            logger.error(f"Error normalizing query: {e}")
            return query
    
    def detect_intent(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Detect the intent of a query with confidence scores.
        
        Args:
            query: Query to analyze
            top_k: Number of top intents to return
            
        Returns:
            List of intent results with confidence scores and retrieval configs
        """
        if not self.initialized or not self.intent_detector:
            logger.warning("Intent detector not initialized")
            return [{"intent": "unknown", "confidence": 0.0, "needs_clarification": True}]
            
        try:
            return self.intent_detector.detect_intent(query, top_k)
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            return [{"intent": "unknown", "confidence": 0.0, "needs_clarification": True}]
    
    def extract_entities(self, query: str, comprehensive: bool = True) -> Dict[str, Any]:
        """
        Extract entities from a query.
        
        Args:
            query: Query to analyze
            comprehensive: If True, return detailed entity information
            
        Returns:
            Entity extraction results
        """
        try:
            if comprehensive:
                return extract_entities_comprehensive(query)
            else:
                entities = extract_entities(query)
                return {"entities": entities}
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"entities": []}
    
    def extract_relations(self, query: str, entities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract relations from a query.
        
        Args:
            query: Query to analyze
            entities: Optional pre-extracted entities
            
        Returns:
            Relation extraction results
        """
        try:
            if entities is None:
                # Extract entities first
                entity_result = self.extract_entities(query, comprehensive=False)
                entities = entity_result.get("entities", [])
            
            # Get comprehensive results
            comprehensive_result = extract_entities_and_relations_comprehensive(query)
            return comprehensive_result["relations"]
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return {"syntactic_relations": [], "semantic_relations": [], "llm_relations": ""}
    
    def get_entity_stats(self, query: str) -> Dict[str, Any]:
        """
        Get detailed statistics about entities in a query.
        
        Args:
            query: Query to analyze
            
        Returns:
            Entity statistics
        """
        try:
            return get_entity_statistics(query)
        except Exception as e:
            logger.error(f"Error getting entity statistics: {e}")
            return {"total_entities": 0, "entity_types": {}, "average_confidence": 0.0}
    
    def process_query(self, 
                     query: str, 
                     context: Optional[str] = None,
                     include_stats: bool = False) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for a query.
        
        Args:
            query: Raw user query
            context: Optional context for normalization
            include_stats: Whether to include detailed statistics
            
        Returns:
            Complete preprocessing results
        """
        result = {
            "original_query": query,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "processing_status": "success"
        }
        
        try:
            # Step 1: Normalize query
            normalized_query = self.normalize_query(query, context)
            result["normalized_query"] = normalized_query
            
            # Step 2: Detect intent
            intent_results = self.detect_intent(normalized_query, top_k=2)
            result["intent"] = {
                "primary_intent": intent_results[0]["intent"],
                "confidence": intent_results[0]["confidence"],
                "needs_clarification": intent_results[0]["needs_clarification"],
                "retrieval_config": intent_results[0].get("retrieval_config", {}),
                "all_intents": intent_results
            }
            
            # Step 3: Extract entities and relations
            comprehensive_extraction = extract_entities_and_relations_comprehensive(normalized_query)
            
            result["entities"] = {
                "entities": comprehensive_extraction["entities"],
                "entity_types": comprehensive_extraction["entity_details"]["entity_types"],
                "confidence_scores": comprehensive_extraction["entity_details"]["confidence_scores"],
                "count": comprehensive_extraction["summary"]["entity_count"]
            }
            
            result["relations"] = {
                "syntactic": comprehensive_extraction["relations"]["syntactic_relations"],
                "semantic": comprehensive_extraction["relations"]["semantic_relations"],
                "llm_relations": comprehensive_extraction["relations"]["llm_relations"],
                "syntactic_count": comprehensive_extraction["summary"]["syntactic_relation_count"],
                "semantic_count": comprehensive_extraction["summary"]["semantic_relation_count"]
            }
            
            # Optional: Include detailed statistics
            if include_stats:
                result["statistics"] = self.get_entity_stats(normalized_query)
            
            # Summary
            result["summary"] = {
                "normalization_changed": normalized_query != query,
                "high_confidence_intent": result["intent"]["confidence"] >= 0.7,
                "entities_found": result["entities"]["count"] > 0,
                "relations_found": (result["relations"]["syntactic_count"] + 
                                 result["relations"]["semantic_count"]) > 0,
                "processing_quality": self._assess_processing_quality(result)
            }
            
        except Exception as e:
            logger.error(f"Error in complete preprocessing: {e}")
            result["processing_status"] = "error"
            result["error"] = str(e)
            
        return result
    
    def _assess_processing_quality(self, result: Dict[str, Any]) -> str:
        """
        Assess the overall quality of preprocessing results.
        
        Args:
            result: Processing results
            
        Returns:
            Quality assessment: "high", "medium", "low"
        """
        try:
            confidence = result["intent"]["confidence"]
            entities_found = result["entities"]["count"]
            relations_found = result["relations"]["syntactic_count"] + result["relations"]["semantic_count"]
            
            if confidence >= 0.8 and entities_found >= 2 and relations_found >= 1:
                return "high"
            elif confidence >= 0.6 and entities_found >= 1:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    def batch_process(self, queries: List[str], context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of queries to process
            context: Optional context for all queries
            
        Returns:
            List of processing results
        """
        results = []
        
        logger.info(f"Processing {len(queries)} queries in batch...")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            result = self.process_query(query, context)
            results.append(result)
        
        logger.info("Batch processing completed")
        return results
    
    def get_available_intents(self) -> List[Dict[str, str]]:
        """
        Get list of available intent categories.
        
        Returns:
            List of intent definitions
        """
        return INTENTS
    
    def get_retrieval_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get retrieval configurations for each intent.
        
        Returns:
            Retrieval configuration mapping
        """
        return RETRIEVAL_CONFIGS
    
    def export_results(self, results: Dict[str, Any], filename: str) -> bool:
        """
        Export processing results to JSON file.
        
        Args:
            results: Results to export
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False


# Convenience functions for quick access
def quick_normalize(query: str, context: Optional[str] = None) -> str:
    """Quick query normalization without initializing full services."""
    try:
        normalizer = DynamicQueryNormalizer()
        return normalizer.normalize(query, context)
    except Exception as e:
        logger.error(f"Quick normalization failed: {e}")
        return query

def quick_intent(query: str) -> str:
    """Quick intent detection without initializing full services."""
    try:
        detector = UniversalIntentDetector(INTENTS, RETRIEVAL_CONFIGS)
        results = detector.detect_intent(query, top_k=1)
        return results[0]["intent"] if results else "unknown"
    except Exception as e:
        logger.error(f"Quick intent detection failed: {e}")
        return "unknown"

def quick_entities(query: str) -> List[str]:
    """Quick entity extraction without initializing full services."""
    try:
        return extract_entities(query)
    except Exception as e:
        logger.error(f"Quick entity extraction failed: {e}")
        return []


# Example usage and testing
if __name__ == "__main__":
    # Initialize services
    services = PreprocessorServices()
    
    # Test queries
    test_queries = [
        "rama father king",
        "What is karma and how does it work?",
        "Compare Krishna and Rama",
        "python code sort list",
        "hello how are you"
    ]
    
    print("=" * 80)
    print("PREPROCESSOR SERVICES DEMONSTRATION")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Processing: '{query}'")
        print("-" * 60)
        
        # Complete processing
        result = services.process_query(query, include_stats=True)
        
        print(f"Normalized: {result['normalized_query']}")
        print(f"Intent: {result['intent']['primary_intent']} (confidence: {result['intent']['confidence']:.3f})")
        print(f"Entities: {result['entities']['entities']}")
        
        if result['relations']['semantic']:
            print(f"Relations: {len(result['relations']['semantic'])} semantic, {len(result['relations']['syntactic'])} syntactic")
        
        print(f"Quality: {result['summary']['processing_quality']}")
    
    print(f"\n✅ Demonstration completed!")