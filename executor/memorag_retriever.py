"""
MemoRAG Retriever - Session-aware Cached Retrieval

Implements memory-augmented retrieval that checks session cache first
and reuses previous contexts if similar queries are found.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
from collections import defaultdict
import hashlib
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MemoRAGRetriever:
    """Memory-augmented retrieval with session cache and context reuse"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 cache_ttl_hours: int = 24,
                 max_cache_size: int = 1000,
                 context_window_size: int = 5,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.max_cache_size = max_cache_size
        self.context_window_size = context_window_size
        
        # Initialize embedding model for similarity computation
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model for MemoRAG: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Session cache structure
        self.session_cache = {}  # session_id -> session_data
        self.query_cache = {}    # query_hash -> cached_results
        self.context_cache = {}  # context_hash -> context_data
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "context_reuse": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
        logger.info(f"Initialized MemoRAG retriever with similarity_threshold: {similarity_threshold}")
    
    def start_session(self, session_id: str) -> bool:
        """Start a new session or resume existing one"""
        try:
            if session_id not in self.session_cache:
                self.session_cache[session_id] = {
                    'session_id': session_id,
                    'start_time': datetime.now(),
                    'last_activity': datetime.now(),
                    'query_history': [],
                    'context_history': [],
                    'retrieval_patterns': defaultdict(int),
                    'successful_retrievals': []
                }
                logger.info(f"Started new MemoRAG session: {session_id}")
            else:
                self.session_cache[session_id]['last_activity'] = datetime.now()
                logger.info(f"Resumed MemoRAG session: {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            return False
    
    def retrieve(self, query: str, session_id: str, 
                base_retrievers: List[Any] = None, 
                k: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve with memory augmentation
        
        Args:
            query: Search query
            session_id: Session identifier
            base_retrievers: List of base retrievers to use if cache miss
            k: Number of results to return
        
        Returns:
            Tuple of (results, memo_info)
        """
        start_time = time.time()
        
        try:
            # Ensure session exists
            if session_id not in self.session_cache:
                self.start_session(session_id)
            
            session_data = self.session_cache[session_id]
            
            # Check for cached results first
            cached_results, cache_info = self._check_cache(query, session_id)
            
            if cached_results:
                # Cache hit - return cached results
                self._update_session_activity(session_id, query, cached_results, cache_hit=True)
                
                memo_info = {
                    'cache_hit': True,
                    'cache_type': cache_info['type'],
                    'similarity_score': cache_info.get('similarity', 1.0),
                    'original_query': cache_info.get('original_query', query),
                    'context_reused': cache_info.get('context_reused', False),
                    'retrieval_time': time.time() - start_time
                }
                
                self.retrieval_stats["cache_hits"] += 1
                if cache_info.get('context_reused'):
                    self.retrieval_stats["context_reuse"] += 1
                
                logger.debug(f"MemoRAG cache hit for query: '{query[:50]}...'")
                return cached_results, memo_info
            
            # Cache miss - need to retrieve using base retrievers
            if not base_retrievers:
                logger.warning("No base retrievers provided and no cache hit")
                return [], {'cache_hit': False, 'error': 'No base retrievers available'}
            
            # Get context from session history
            context = self._get_session_context(session_id, query)
            
            # Perform retrieval using base retrievers
            all_results = []
            retriever_info = {}
            
            for retriever in base_retrievers:
                try:
                    if hasattr(retriever, 'retrieve'):
                        results = retriever.retrieve(query, k=k)
                        all_results.extend(results)
                        retriever_info[type(retriever).__name__] = len(results)
                except Exception as e:
                    logger.error(f"Error in base retriever {type(retriever).__name__}: {e}")
            
            # Remove duplicates and limit results
            unique_results = self._deduplicate_results(all_results)[:k]
            
            # Enhance results with context
            enhanced_results = self._enhance_with_context(unique_results, context)
            
            # Cache the results
            self._cache_results(query, session_id, enhanced_results, context)
            
            # Update session
            self._update_session_activity(session_id, query, enhanced_results, cache_hit=False)
            
            memo_info = {
                'cache_hit': False,
                'base_retrievers_used': list(retriever_info.keys()),
                'retriever_results': retriever_info,
                'context_applied': bool(context),
                'total_results_before_dedup': len(all_results),
                'final_results': len(enhanced_results),
                'retrieval_time': time.time() - start_time
            }
            
            # Update stats
            self.retrieval_stats["total_queries"] += 1
            self.retrieval_stats["total_time"] += memo_info['retrieval_time']
            self.retrieval_stats["average_time"] = (
                self.retrieval_stats["total_time"] / self.retrieval_stats["total_queries"]
            )
            self.retrieval_stats["cache_hit_rate"] = (
                self.retrieval_stats["cache_hits"] / self.retrieval_stats["total_queries"]
            )
            
            logger.debug(f"MemoRAG retrieval completed in {memo_info['retrieval_time']:.3f}s")
            
            return enhanced_results, memo_info
            
        except Exception as e:
            logger.error(f"MemoRAG retrieval failed: {e}")
            return [], {'cache_hit': False, 'error': str(e)}
    
    def _check_cache(self, query: str, session_id: str) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Check various cache levels for similar queries"""
        
        # 1. Exact query cache
        query_hash = self._hash_query(query)
        if query_hash in self.query_cache:
            cache_entry = self.query_cache[query_hash]
            if self._is_cache_valid(cache_entry):
                return cache_entry['results'], {
                    'type': 'exact_query',
                    'original_query': query
                }
        
        # 2. Similar query cache (if embedding model available)
        if self.embedding_model:
            similar_results = self._find_similar_cached_query(query, session_id)
            if similar_results:
                return similar_results['results'], {
                    'type': 'similar_query',
                    'similarity': similar_results['similarity'],
                    'original_query': similar_results['original_query']
                }
        
        # 3. Context-based cache
        context_results = self._check_context_cache(query, session_id)
        if context_results:
            return context_results['results'], {
                'type': 'context_based',
                'context_reused': True,
                'original_context': context_results['context_key']
            }
        
        return None, {}
    
    def _find_similar_cached_query(self, query: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Find similar cached queries using embeddings"""
        if not self.embedding_model:
            return None
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Check session-specific queries first
            session_data = self.session_cache.get(session_id, {})
            session_queries = session_data.get('query_history', [])
            
            best_match = None
            best_similarity = 0
            
            # Check recent session queries
            for hist_query in session_queries[-10:]:  # Check last 10 queries
                hist_query_text = hist_query.get('query', '')
                hist_query_hash = self._hash_query(hist_query_text)
                
                if hist_query_hash in self.query_cache:
                    cache_entry = self.query_cache[hist_query_hash]
                    if self._is_cache_valid(cache_entry):
                        hist_embedding = self.embedding_model.encode([hist_query_text])[0]
                        similarity = np.dot(query_embedding, hist_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(hist_embedding)
                        )
                        
                        if similarity > best_similarity and similarity >= self.similarity_threshold:
                            best_similarity = similarity
                            best_match = {
                                'results': cache_entry['results'],
                                'similarity': similarity,
                                'original_query': hist_query_text
                            }
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding similar cached query: {e}")
            return None
    
    def _check_context_cache(self, query: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Check if query can be answered using cached context"""
        session_data = self.session_cache.get(session_id, {})
        context_history = session_data.get('context_history', [])
        
        # Look for relevant context in recent history
        for context_entry in context_history[-5:]:  # Check last 5 contexts
            context_key = context_entry.get('context_key')
            if context_key and context_key in self.context_cache:
                cached_context = self.context_cache[context_key]
                
                # Simple keyword matching for context relevance
                query_words = set(query.lower().split())
                context_words = set(cached_context.get('keywords', []))
                
                overlap = len(query_words.intersection(context_words))
                if overlap >= 2:  # At least 2 word overlap
                    return {
                        'results': cached_context.get('results', []),
                        'context_key': context_key
                    }
        
        return None
    
    def _get_session_context(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """Get relevant context from session history"""
        session_data = self.session_cache.get(session_id, {})
        query_history = session_data.get('query_history', [])
        
        if not query_history:
            return {}
        
        # Get recent queries for context
        recent_queries = query_history[-self.context_window_size:]
        
        context = {
            'recent_queries': [q.get('query', '') for q in recent_queries],
            'recent_entities': [],
            'recent_topics': [],
            'query_patterns': session_data.get('retrieval_patterns', {})
        }
        
        # Extract entities and topics from recent queries
        for query_entry in recent_queries:
            results = query_entry.get('results', [])
            for result in results:
                if 'matched_entities' in result:
                    context['recent_entities'].extend(result['matched_entities'])
                if 'topic' in result:
                    context['recent_topics'].append(result['topic'])
        
        # Remove duplicates
        context['recent_entities'] = list(set(context['recent_entities']))
        context['recent_topics'] = list(set(context['recent_topics']))
        
        return context
    
    def _enhance_with_context(self, results: List[Dict[str, Any]], 
                            context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance retrieval results with session context"""
        if not context or not results:
            return results
        
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add context relevance score
            context_score = 0.0
            
            # Check entity overlap
            result_entities = result.get('matched_entities', [])
            context_entities = context.get('recent_entities', [])
            
            if result_entities and context_entities:
                entity_overlap = len(set(result_entities).intersection(set(context_entities)))
                context_score += entity_overlap * 0.1
            
            # Check topic relevance
            result_topic = result.get('topic', '')
            context_topics = context.get('recent_topics', [])
            
            if result_topic and result_topic in context_topics:
                context_score += 0.2
            
            # Boost score based on context relevance
            original_score = result.get('similarity_score', 0.0)
            enhanced_score = original_score + context_score
            
            enhanced_result.update({
                'similarity_score': enhanced_score,
                'context_boost': context_score,
                'context_enhanced': context_score > 0,
                'memo_rag_processed': True
            })
            
            enhanced_results.append(enhanced_result)
        
        # Re-sort by enhanced scores
        enhanced_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return enhanced_results
    
    def _cache_results(self, query: str, session_id: str, 
                      results: List[Dict[str, Any]], context: Dict[str, Any]):
        """Cache retrieval results for future use"""
        try:
            # Cache query results
            query_hash = self._hash_query(query)
            self.query_cache[query_hash] = {
                'query': query,
                'results': results,
                'timestamp': datetime.now(),
                'session_id': session_id,
                'context': context
            }
            
            # Cache context if significant
            if context and (context.get('recent_entities') or context.get('recent_topics')):
                context_key = self._hash_context(context)
                self.context_cache[context_key] = {
                    'context': context,
                    'results': results,
                    'keywords': self._extract_keywords_from_context(context),
                    'timestamp': datetime.now()
                }
            
            # Cleanup old cache entries if needed
            self._cleanup_cache()
            
        except Exception as e:
            logger.error(f"Error caching results: {e}")
    
    def _update_session_activity(self, session_id: str, query: str, 
                                results: List[Dict[str, Any]], cache_hit: bool):
        """Update session with new activity"""
        session_data = self.session_cache[session_id]
        
        # Add to query history
        query_entry = {
            'query': query,
            'timestamp': datetime.now(),
            'results': results,
            'cache_hit': cache_hit,
            'result_count': len(results)
        }
        
        session_data['query_history'].append(query_entry)
        session_data['last_activity'] = datetime.now()
        
        # Update retrieval patterns
        if results:
            for result in results:
                retriever_type = result.get('retriever_type', 'unknown')
                session_data['retrieval_patterns'][retriever_type] += 1
        
        # Keep only recent history
        if len(session_data['query_history']) > 50:
            session_data['query_history'] = session_data['query_history'][-50:]
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content or ID"""
        seen = set()
        unique_results = []
        
        for result in results:
            # Create identifier based on content or ID
            identifier = result.get('id') or result.get('content', '')[:100]
            
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)
        
        return unique_results
    
    def _hash_query(self, query: str) -> str:
        """Create hash for query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hash for context"""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        timestamp = cache_entry.get('timestamp')
        if not timestamp:
            return False
        
        return datetime.now() - timestamp < self.cache_ttl
    
    def _extract_keywords_from_context(self, context: Dict[str, Any]) -> List[str]:
        """Extract keywords from context for matching"""
        keywords = []
        
        # Add entities
        keywords.extend(context.get('recent_entities', []))
        
        # Add topics
        keywords.extend(context.get('recent_topics', []))
        
        # Add words from recent queries
        for query in context.get('recent_queries', []):
            keywords.extend(query.lower().split())
        
        # Remove duplicates and common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [k for k in set(keywords) if k.lower() not in stop_words and len(k) > 2]
        
        return keywords
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = datetime.now()
        
        # Clean query cache
        expired_queries = [
            query_hash for query_hash, entry in self.query_cache.items()
            if current_time - entry.get('timestamp', current_time) > self.cache_ttl
        ]
        
        for query_hash in expired_queries:
            del self.query_cache[query_hash]
        
        # Clean context cache
        expired_contexts = [
            context_key for context_key, entry in self.context_cache.items()
            if current_time - entry.get('timestamp', current_time) > self.cache_ttl
        ]
        
        for context_key in expired_contexts:
            del self.context_cache[context_key]
        
        # Limit cache size
        if len(self.query_cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_queries = sorted(
                self.query_cache.items(),
                key=lambda x: x[1].get('timestamp', datetime.min)
            )
            
            for query_hash, _ in sorted_queries[:len(self.query_cache) - self.max_cache_size]:
                del self.query_cache[query_hash]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session activity"""
        if session_id not in self.session_cache:
            return {'error': 'Session not found'}
        
        session_data = self.session_cache[session_id]
        query_history = session_data.get('query_history', [])
        
        summary = {
            'session_id': session_id,
            'start_time': session_data.get('start_time'),
            'last_activity': session_data.get('last_activity'),
            'total_queries': len(query_history),
            'cache_hits': sum(1 for q in query_history if q.get('cache_hit', False)),
            'retrieval_patterns': dict(session_data.get('retrieval_patterns', {})),
            'avg_results_per_query': (
                sum(q.get('result_count', 0) for q in query_history) / len(query_history)
                if query_history else 0
            )
        }
        
        summary['cache_hit_rate'] = (
            summary['cache_hits'] / summary['total_queries']
            if summary['total_queries'] > 0 else 0
        )
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        stats = self.retrieval_stats.copy()
        stats.update({
            'active_sessions': len(self.session_cache),
            'query_cache_size': len(self.query_cache),
            'context_cache_size': len(self.context_cache),
            'similarity_threshold': self.similarity_threshold,
            'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600,
            'max_cache_size': self.max_cache_size
        })
        
        return stats
    
    def clear_session(self, session_id: str) -> bool:
        """Clear specific session data"""
        if session_id in self.session_cache:
            del self.session_cache[session_id]
            logger.info(f"Cleared MemoRAG session: {session_id}")
            return True
        return False
    
    def clear_all_cache(self):
        """Clear all cache data"""
        self.query_cache.clear()
        self.context_cache.clear()
        logger.info("Cleared all MemoRAG cache")
    
    def health_check(self) -> Dict[str, Any]:
        """Check retriever health"""
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        if not self.embedding_model:
            health['status'] = 'warning'
            health['issues'].append('Embedding model not available - similarity matching disabled')
        
        # Check cache sizes
        if len(self.query_cache) > self.max_cache_size * 1.2:
            health['status'] = 'warning'
            health['issues'].append('Query cache size exceeding limits')
        
        return health