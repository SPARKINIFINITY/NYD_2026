"""
Signal Processor - Multi-Signal Integration

Processes and combines multiple relevance signals for enhanced reranking:
- dense_score: Semantic similarity from dense retrieval
- bm25_score: Keyword relevance from sparse retrieval  
- entity_overlap: Entity matching between query and document
- session_relevance: Context relevance from session history
- cluster_match: Topic/cluster alignment scoring
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import re
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalProcessor:
    """Processes and combines multiple relevance signals"""
    
    def __init__(self, 
                 signal_weights: Dict[str, float] = None,
                 normalization_method: str = "min_max"):
        
        # Default signal weights
        self.signal_weights = signal_weights or {
            "dense_score": 0.25,
            "bm25_score": 0.20,
            "entity_overlap": 0.20,
            "session_relevance": 0.15,
            "cluster_match": 0.10,
            "cross_encoder_score": 0.10  # From stage 1
        }
        
        self.normalization_method = normalization_method
        
        # Signal processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "signal_distributions": {},
            "correlation_matrix": {},
            "processing_time": 0.0
        }
        
        logger.info(f"Initialized signal processor with weights: {self.signal_weights}")
    
    def process_signals(self, 
                       query: str,
                       candidates: List[Dict[str, Any]],
                       session_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process all signals for candidates
        
        Args:
            query: Search query
            candidates: List of candidate documents
            session_context: Optional session context for relevance
        
        Returns:
            Candidates with processed signals
        """
        if not candidates:
            return candidates
        
        try:
            # Extract query features
            query_features = self._extract_query_features(query)
            
            # Process each candidate
            processed_candidates = []
            
            for candidate in candidates:
                processed_candidate = candidate.copy()
                
                # Calculate individual signals
                signals = self._calculate_all_signals(
                    query, query_features, candidate, session_context
                )
                
                # Add signals to candidate
                processed_candidate.update(signals)
                
                # Calculate combined signal score
                combined_score = self._combine_signals(signals)
                processed_candidate['combined_signal_score'] = combined_score
                
                processed_candidates.append(processed_candidate)
            
            # Normalize signals across all candidates
            normalized_candidates = self._normalize_signals(processed_candidates)
            
            # Update statistics
            self._update_processing_stats(normalized_candidates)
            
            return normalized_candidates
            
        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
            return candidates
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from query for signal calculation"""
        
        features = {
            "query": query,
            "query_lower": query.lower(),
            "query_words": query.lower().split(),
            "query_length": len(query.split()),
            "entities": self._extract_simple_entities(query),
            "keywords": self._extract_keywords(query),
            "question_type": self._detect_question_type(query)
        }
        
        return features
    
    def _calculate_all_signals(self, 
                             query: str,
                             query_features: Dict[str, Any],
                             candidate: Dict[str, Any],
                             session_context: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate all relevance signals for a candidate"""
        
        signals = {}
        
        # 1. Dense score (from retrieval)
        signals['dense_score'] = self._get_dense_score(candidate)
        
        # 2. BM25 score (from retrieval)
        signals['bm25_score'] = self._get_bm25_score(candidate)
        
        # 3. Entity overlap
        signals['entity_overlap'] = self._calculate_entity_overlap(query_features, candidate)
        
        # 4. Session relevance
        signals['session_relevance'] = self._calculate_session_relevance(
            query_features, candidate, session_context
        )
        
        # 5. Cluster match
        signals['cluster_match'] = self._calculate_cluster_match(query_features, candidate)
        
        # 6. Cross-encoder score (from stage 1)
        signals['cross_encoder_score'] = candidate.get('cross_encoder_score', 0.0)
        
        # Additional derived signals
        signals['text_overlap'] = self._calculate_text_overlap(query_features, candidate)
        signals['length_penalty'] = self._calculate_length_penalty(candidate)
        signals['freshness_score'] = self._calculate_freshness_score(candidate)
        
        return signals
    
    def _get_dense_score(self, candidate: Dict[str, Any]) -> float:
        """Extract dense retrieval score"""
        return candidate.get('similarity_score', 0.0) if candidate.get('retriever_type') == 'dense' else 0.0
    
    def _get_bm25_score(self, candidate: Dict[str, Any]) -> float:
        """Extract BM25 score"""
        return candidate.get('similarity_score', 0.0) if candidate.get('retriever_type') in ['sparse', 'bm25'] else 0.0
    
    def _calculate_entity_overlap(self, query_features: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Calculate entity overlap between query and document"""
        
        query_entities = set(entity.lower() for entity in query_features.get('entities', []))
        
        # Get candidate entities
        candidate_entities = set()
        
        # From explicit entity field
        if 'entities' in candidate:
            entities = candidate['entities']
            if isinstance(entities, list):
                candidate_entities.update(entity.lower() for entity in entities)
            elif isinstance(entities, str):
                candidate_entities.add(entities.lower())
        
        # From matched entities field
        if 'matched_entities' in candidate:
            matched = candidate['matched_entities']
            if isinstance(matched, list):
                candidate_entities.update(entity.lower() for entity in matched)
        
        # Extract from content if no explicit entities
        if not candidate_entities:
            content = candidate.get('content', '')
            candidate_entities = set(self._extract_simple_entities(content))
        
        # Calculate overlap
        if not query_entities or not candidate_entities:
            return 0.0
        
        overlap = len(query_entities.intersection(candidate_entities))
        max_entities = max(len(query_entities), len(candidate_entities))
        
        return overlap / max_entities if max_entities > 0 else 0.0
    
    def _calculate_session_relevance(self, 
                                   query_features: Dict[str, Any],
                                   candidate: Dict[str, Any],
                                   session_context: Dict[str, Any] = None) -> float:
        """Calculate relevance based on session context"""
        
        if not session_context:
            return 0.0
        
        relevance_score = 0.0
        
        # Recent entities from session
        recent_entities = session_context.get('recent_entities', [])
        if recent_entities:
            candidate_entities = set()
            
            # Get candidate entities
            if 'entities' in candidate:
                entities = candidate['entities']
                if isinstance(entities, list):
                    candidate_entities.update(entity.lower() for entity in entities)
            
            if 'matched_entities' in candidate:
                matched = candidate['matched_entities']
                if isinstance(matched, list):
                    candidate_entities.update(entity.lower() for entity in matched)
            
            # Calculate entity overlap with session
            session_entity_set = set(entity.lower() for entity in recent_entities)
            entity_overlap = len(session_entity_set.intersection(candidate_entities))
            
            if session_entity_set:
                relevance_score += (entity_overlap / len(session_entity_set)) * 0.5
        
        # Topic continuity
        current_topic = session_context.get('current_topic')
        candidate_topic = candidate.get('topic')
        
        if current_topic and candidate_topic and current_topic == candidate_topic:
            relevance_score += 0.3
        
        # Recent query similarity
        recent_queries = session_context.get('recent_queries', [])
        if recent_queries:
            query_words = set(query_features['query_words'])
            
            for recent_query in recent_queries[-3:]:  # Last 3 queries
                recent_words = set(recent_query.lower().split())
                overlap = len(query_words.intersection(recent_words))
                
                if query_words:
                    query_similarity = overlap / len(query_words)
                    relevance_score += query_similarity * 0.2
        
        return min(relevance_score, 1.0)
    
    def _calculate_cluster_match(self, query_features: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Calculate topic/cluster alignment score"""
        
        # Simple topic matching
        query_topic = self._infer_topic_from_query(query_features)
        candidate_topic = candidate.get('topic', '')
        
        if query_topic and candidate_topic:
            if query_topic.lower() == candidate_topic.lower():
                return 1.0
            elif self._topics_related(query_topic, candidate_topic):
                return 0.6
        
        # Keyword-based cluster matching
        query_keywords = set(query_features.get('keywords', []))
        content = candidate.get('content', '').lower()
        content_words = set(content.split())
        
        if query_keywords and content_words:
            keyword_overlap = len(query_keywords.intersection(content_words))
            return min(keyword_overlap / len(query_keywords), 1.0)
        
        return 0.0
    
    def _calculate_text_overlap(self, query_features: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Calculate direct text overlap between query and document"""
        
        query_words = set(query_features['query_words'])
        content = candidate.get('content', '').lower()
        content_words = set(content.split())
        
        if not query_words or not content_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)
    
    def _calculate_length_penalty(self, candidate: Dict[str, Any]) -> float:
        """Calculate penalty/bonus based on document length"""
        
        content = candidate.get('content', '')
        length = len(content.split())
        
        # Optimal length range (100-500 words)
        if 100 <= length <= 500:
            return 1.0
        elif length < 100:
            return 0.8  # Too short
        elif length > 1000:
            return 0.7  # Too long
        else:
            return 0.9  # Acceptable
    
    def _calculate_freshness_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate freshness/recency score"""
        
        # If no timestamp, assume neutral
        if 'timestamp' not in candidate:
            return 0.5
        
        try:
            timestamp = candidate['timestamp']
            if isinstance(timestamp, str):
                doc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                doc_time = timestamp
            
            # Calculate age in days
            age_days = (datetime.now() - doc_time).days
            
            # Fresher documents get higher scores
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.9
            elif age_days <= 30:
                return 0.8
            elif age_days <= 90:
                return 0.6
            else:
                return 0.4
                
        except Exception:
            return 0.5
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """Extract simple entities (capitalized words) from text"""
        
        words = text.split()
        entities = []
        
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Consider capitalized words as entities
            if clean_word and (clean_word[0].isupper() or len(clean_word) > 6):
                entities.append(clean_word)
        
        return list(set(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        
        # Simple keyword extraction
        words = text.lower().split()
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'who', 'where', 'when', 'why', 'how'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return list(set(keywords))
    
    def _detect_question_type(self, query: str) -> str:
        """Detect the type of question"""
        
        query_lower = query.lower()
        
        if query_lower.startswith('who'):
            return 'person'
        elif query_lower.startswith('what'):
            return 'definition'
        elif query_lower.startswith('where'):
            return 'location'
        elif query_lower.startswith('when'):
            return 'time'
        elif query_lower.startswith('why'):
            return 'reason'
        elif query_lower.startswith('how'):
            return 'process'
        elif 'compare' in query_lower or 'versus' in query_lower:
            return 'comparison'
        else:
            return 'general'
    
    def _infer_topic_from_query(self, query_features: Dict[str, Any]) -> str:
        """Infer topic from query features"""
        
        keywords = query_features.get('keywords', [])
        entities = query_features.get('entities', [])
        
        all_terms = keywords + entities
        all_terms_lower = [term.lower() for term in all_terms]
        
        # Topic classification based on keywords
        tech_terms = {'python', 'programming', 'algorithm', 'machine', 'learning', 'ai', 'computer', 'software'}
        mythology_terms = {'rama', 'krishna', 'hanuman', 'ramayana', 'mahabharata', 'hindu', 'deity'}
        science_terms = {'physics', 'chemistry', 'biology', 'theory', 'research', 'study'}
        
        if any(term in all_terms_lower for term in tech_terms):
            return 'technology'
        elif any(term in all_terms_lower for term in mythology_terms):
            return 'mythology'
        elif any(term in all_terms_lower for term in science_terms):
            return 'science'
        else:
            return 'general'
    
    def _topics_related(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are related"""
        
        related_topics = {
            'technology': ['programming', 'computer', 'software'],
            'mythology': ['religion', 'hindu', 'spiritual'],
            'science': ['research', 'academic', 'study']
        }
        
        topic1_lower = topic1.lower()
        topic2_lower = topic2.lower()
        
        for main_topic, related in related_topics.items():
            if topic1_lower == main_topic and topic2_lower in related:
                return True
            if topic2_lower == main_topic and topic1_lower in related:
                return True
        
        return False
    
    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Combine individual signals into final score"""
        
        combined_score = 0.0
        total_weight = 0.0
        
        for signal_name, weight in self.signal_weights.items():
            if signal_name in signals:
                signal_value = signals[signal_name]
                combined_score += signal_value * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_score /= total_weight
        
        return min(combined_score, 1.0)
    
    def _normalize_signals(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize signals across all candidates"""
        
        if not candidates:
            return candidates
        
        # Collect all signal values
        signal_values = {}
        for signal_name in self.signal_weights.keys():
            signal_values[signal_name] = [
                candidate.get(signal_name, 0.0) for candidate in candidates
            ]
        
        # Normalize each signal
        normalized_candidates = []
        
        for candidate in candidates:
            normalized_candidate = candidate.copy()
            
            for signal_name, values in signal_values.items():
                if signal_name in candidate:
                    original_value = candidate[signal_name]
                    
                    if self.normalization_method == "min_max":
                        min_val, max_val = min(values), max(values)
                        if max_val > min_val:
                            normalized_value = (original_value - min_val) / (max_val - min_val)
                        else:
                            normalized_value = 1.0 if original_value > 0 else 0.0
                    
                    elif self.normalization_method == "z_score":
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        if std_val > 0:
                            normalized_value = (original_value - mean_val) / std_val
                            # Convert to 0-1 range using sigmoid
                            normalized_value = 1 / (1 + np.exp(-normalized_value))
                        else:
                            normalized_value = 0.5
                    
                    else:  # No normalization
                        normalized_value = original_value
                    
                    normalized_candidate[f'{signal_name}_normalized'] = normalized_value
            
            normalized_candidates.append(normalized_candidate)
        
        return normalized_candidates
    
    def _update_processing_stats(self, candidates: List[Dict[str, Any]]):
        """Update signal processing statistics"""
        
        self.processing_stats["total_processed"] += len(candidates)
        
        # Update signal distributions
        for signal_name in self.signal_weights.keys():
            values = [candidate.get(signal_name, 0.0) for candidate in candidates]
            
            if signal_name not in self.processing_stats["signal_distributions"]:
                self.processing_stats["signal_distributions"][signal_name] = {
                    "count": 0,
                    "sum": 0.0,
                    "sum_squares": 0.0
                }
            
            dist = self.processing_stats["signal_distributions"][signal_name]
            dist["count"] += len(values)
            dist["sum"] += sum(values)
            dist["sum_squares"] += sum(v * v for v in values)
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about signal processing"""
        
        stats = self.processing_stats.copy()
        
        # Calculate means and standard deviations
        for signal_name, dist in stats["signal_distributions"].items():
            if dist["count"] > 0:
                mean = dist["sum"] / dist["count"]
                variance = (dist["sum_squares"] / dist["count"]) - (mean * mean)
                std = np.sqrt(max(variance, 0))
                
                dist["mean"] = mean
                dist["std"] = std
        
        return stats
    
    def analyze_signal_correlations(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between different signals"""
        
        if len(candidates) < 2:
            return {"error": "Need at least 2 candidates for correlation analysis"}
        
        # Extract signal values
        signal_data = {}
        for signal_name in self.signal_weights.keys():
            signal_data[signal_name] = [
                candidate.get(signal_name, 0.0) for candidate in candidates
            ]
        
        # Calculate correlations
        correlations = {}
        signal_names = list(signal_data.keys())
        
        for i, signal1 in enumerate(signal_names):
            for j, signal2 in enumerate(signal_names[i+1:], i+1):
                values1 = np.array(signal_data[signal1])
                values2 = np.array(signal_data[signal2])
                
                if np.std(values1) > 0 and np.std(values2) > 0:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlations[f"{signal1}_vs_{signal2}"] = correlation
        
        return {
            "correlations": correlations,
            "signal_means": {name: np.mean(values) for name, values in signal_data.items()},
            "signal_stds": {name: np.std(values) for name, values in signal_data.items()}
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive signal processor statistics"""
        
        return {
            "signal_weights": self.signal_weights,
            "normalization_method": self.normalization_method,
            "processing_stats": self.processing_stats
        }