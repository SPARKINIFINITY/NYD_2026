"""
Sparse Retriever - TF-IDF/BM25 Keyword Search

Implements sparse retrieval using TF-IDF vectorization and BM25 scoring
for keyword-based search.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
import pickle
import re
from collections import Counter
import math

logger = logging.getLogger(__name__)

class BM25:
    """BM25 scoring implementation"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        
    def fit(self, corpus: List[str]):
        """Fit BM25 on corpus"""
        self.corpus = corpus
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # Calculate document frequencies
        df = {}
        for doc in corpus:
            words = set(doc.lower().split())
            for word in words:
                df[word] = df.get(word, 0) + 1
        
        # Calculate IDF
        N = len(corpus)
        for word, freq in df.items():
            self.idf[word] = math.log((N - freq + 0.5) / (freq + 0.5))
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for query and document"""
        doc = self.corpus[doc_idx]
        doc_words = doc.lower().split()
        doc_word_counts = Counter(doc_words)
        
        score = 0
        query_words = query.lower().split()
        
        for word in query_words:
            if word in doc_word_counts:
                tf = doc_word_counts[word]
                idf = self.idf.get(word, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[doc_idx] / self.avgdl))
                score += idf * (numerator / denominator)
        
        return score
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for all documents"""
        return [self.score(query, i) for i in range(len(self.corpus))]

class SparseRetriever:
    """Sparse retrieval using TF-IDF and BM25"""
    
    def __init__(self, 
                 method: str = "tfidf",  # "tfidf" or "bm25"
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.8):
        
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize components
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
        elif method == "bm25":
            self.bm25 = BM25()
            self.vectorizer = None
        
        # Data storage
        self.document_matrix = None
        self.document_metadata = []
        self.corpus = []
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "cache_hits": 0
        }
        
        # Query cache
        self.query_cache = {}
        self.max_cache_size = 1000
        
        logger.info(f"Initialized sparse retriever with method: {method}")
    
    def build_index(self, documents: List[Dict[str, Any]], save_path: str = None) -> bool:
        """
        Build sparse index from documents
        
        Args:
            documents: List of documents with 'content' and metadata
            save_path: Optional path to save the index
        """
        try:
            logger.info(f"Building sparse index for {len(documents)} documents...")
            start_time = time.time()
            
            # Extract text content
            texts = []
            metadata = []
            
            for doc in documents:
                if isinstance(doc, dict):
                    content = doc.get('content', str(doc))
                    texts.append(self._preprocess_text(content))
                    metadata.append(doc)
                else:
                    texts.append(self._preprocess_text(str(doc)))
                    metadata.append({'content': str(doc), 'id': len(metadata)})
            
            self.corpus = texts
            self.document_metadata = metadata
            
            if self.method == "tfidf":
                # Build TF-IDF matrix
                self.document_matrix = self.vectorizer.fit_transform(texts)
                logger.info(f"TF-IDF matrix shape: {self.document_matrix.shape}")
                
            elif self.method == "bm25":
                # Fit BM25
                self.bm25.fit(texts)
                logger.info(f"BM25 fitted on {len(texts)} documents")
            
            build_time = time.time() - start_time
            logger.info(f"Sparse index built in {build_time:.2f}s")
            
            # Save index if path provided
            if save_path:
                self.save_index(save_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build sparse index: {e}")
            return False
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sparse retrieval"""
        # Basic text cleaning
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
        text = text.lower().strip()
        return text
    
    def save_index(self, base_path: str):
        """Save sparse index and metadata"""
        try:
            if self.method == "tfidf":
                # Save vectorizer and matrix
                with open(f"{base_path}_vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                
                with open(f"{base_path}_matrix.pkl", 'wb') as f:
                    pickle.dump(self.document_matrix, f)
                    
            elif self.method == "bm25":
                # Save BM25 model
                with open(f"{base_path}_bm25.pkl", 'wb') as f:
                    pickle.dump(self.bm25, f)
            
            # Save metadata and corpus
            with open(f"{base_path}_metadata.pkl", 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            with open(f"{base_path}_corpus.pkl", 'wb') as f:
                pickle.dump(self.corpus, f)
            
            logger.info(f"Sparse index saved to {base_path}")
            
        except Exception as e:
            logger.error(f"Failed to save sparse index: {e}")
    
    def load_index(self, base_path: str) -> bool:
        """Load sparse index and metadata"""
        try:
            if self.method == "tfidf":
                # Load vectorizer and matrix
                with open(f"{base_path}_vectorizer.pkl", 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(f"{base_path}_matrix.pkl", 'rb') as f:
                    self.document_matrix = pickle.load(f)
                    
            elif self.method == "bm25":
                # Load BM25 model
                with open(f"{base_path}_bm25.pkl", 'rb') as f:
                    self.bm25 = pickle.load(f)
            
            # Load metadata and corpus
            with open(f"{base_path}_metadata.pkl", 'rb') as f:
                self.document_metadata = pickle.load(f)
            
            with open(f"{base_path}_corpus.pkl", 'rb') as f:
                self.corpus = pickle.load(f)
            
            logger.info(f"Sparse index loaded from {base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load sparse index: {e}")
            return False
    
    def retrieve(self, query: str, k: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents using sparse search
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Optional similarity threshold
        
        Returns:
            List of retrieved documents with scores
        """
        if not self.corpus:
            logger.error("Sparse retriever not properly initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{query}_{k}_{threshold}"
            if cache_key in self.query_cache:
                self.retrieval_stats["cache_hits"] += 1
                return self.query_cache[cache_key]
            
            # Preprocess query
            processed_query = self._preprocess_text(query)
            
            if self.method == "tfidf":
                scores = self._tfidf_search(processed_query)
            elif self.method == "bm25":
                scores = self._bm25_search(processed_query)
            else:
                logger.error(f"Unknown method: {self.method}")
                return []
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Process results
            results = []
            for rank, idx in enumerate(top_indices):
                score = scores[idx]
                
                # Apply threshold if specified
                if threshold and score < threshold:
                    continue
                
                # Get document metadata
                if idx < len(self.document_metadata):
                    doc_metadata = self.document_metadata[idx].copy()
                    doc_metadata.update({
                        'retriever_type': 'sparse',
                        'similarity_score': float(score),
                        'rank': rank + 1,
                        'retrieval_method': f'{self.method}_sparse'
                    })
                    results.append(doc_metadata)
            
            # Update cache
            if len(self.query_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            
            self.query_cache[cache_key] = results
            
            # Update stats
            retrieval_time = time.time() - start_time
            self.retrieval_stats["total_queries"] += 1
            self.retrieval_stats["total_time"] += retrieval_time
            self.retrieval_stats["average_time"] = (
                self.retrieval_stats["total_time"] / self.retrieval_stats["total_queries"]
            )
            
            logger.debug(f"Sparse retrieval completed in {retrieval_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def _tfidf_search(self, query: str) -> np.ndarray:
        """Perform TF-IDF search"""
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_matrix).flatten()
        
        return similarities
    
    def _bm25_search(self, query: str) -> List[float]:
        """Perform BM25 search"""
        return self.bm25.get_scores(query)
    
    def batch_retrieve(self, queries: List[str], k: int = 10) -> List[List[Dict[str, Any]]]:
        """Batch retrieval for multiple queries"""
        batch_results = []
        
        for query in queries:
            results = self.retrieve(query, k)
            batch_results.append(results)
        
        return batch_results
    
    def get_term_importance(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most important terms for a query"""
        if self.method != "tfidf" or not self.vectorizer:
            return []
        
        try:
            processed_query = self._preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            
            # Get feature names and scores
            feature_names = self.vectorizer.get_feature_names_out()
            scores = query_vector.toarray()[0]
            
            # Get top terms
            top_indices = np.argsort(scores)[::-1][:top_n]
            top_terms = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return top_terms
            
        except Exception as e:
            logger.error(f"Failed to get term importance: {e}")
            return []
    
    def explain_retrieval(self, query: str, doc_id: str) -> Dict[str, Any]:
        """Explain why a document was retrieved for a query"""
        # Find document index
        doc_idx = None
        for i, doc in enumerate(self.document_metadata):
            if doc.get('id') == doc_id:
                doc_idx = i
                break
        
        if doc_idx is None:
            return {"error": "Document not found"}
        
        explanation = {
            "query": query,
            "document_id": doc_id,
            "method": self.method,
            "matching_terms": [],
            "score_breakdown": {}
        }
        
        try:
            processed_query = self._preprocess_text(query)
            query_terms = set(processed_query.split())
            doc_text = self.corpus[doc_idx]
            doc_terms = set(doc_text.split())
            
            # Find matching terms
            matching_terms = query_terms.intersection(doc_terms)
            explanation["matching_terms"] = list(matching_terms)
            
            if self.method == "tfidf":
                # Get TF-IDF explanation
                query_vector = self.vectorizer.transform([processed_query])
                doc_vector = self.document_matrix[doc_idx]
                
                feature_names = self.vectorizer.get_feature_names_out()
                query_scores = query_vector.toarray()[0]
                doc_scores = doc_vector.toarray()[0]
                
                for term in matching_terms:
                    if term in feature_names:
                        term_idx = list(feature_names).index(term)
                        explanation["score_breakdown"][term] = {
                            "query_tfidf": float(query_scores[term_idx]),
                            "doc_tfidf": float(doc_scores[term_idx])
                        }
            
            elif self.method == "bm25":
                # Get BM25 explanation
                score = self.bm25.score(processed_query, doc_idx)
                explanation["score_breakdown"]["total_bm25_score"] = score
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain retrieval: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        stats = self.retrieval_stats.copy()
        stats.update({
            'method': self.method,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'corpus_size': len(self.corpus),
            'cache_size': len(self.query_cache)
        })
        
        if self.method == "tfidf" and self.document_matrix is not None:
            stats['matrix_shape'] = self.document_matrix.shape
            stats['vocabulary_size'] = len(self.vectorizer.get_feature_names_out())
        
        return stats
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Sparse retriever cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Check retriever health"""
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        if not self.corpus:
            health['status'] = 'unhealthy'
            health['issues'].append('No corpus loaded')
        
        if self.method == "tfidf":
            if not self.vectorizer:
                health['status'] = 'unhealthy'
                health['issues'].append('TF-IDF vectorizer not initialized')
            if self.document_matrix is None:
                health['status'] = 'unhealthy'
                health['issues'].append('Document matrix not built')
        
        elif self.method == "bm25":
            if not hasattr(self, 'bm25') or not self.bm25.corpus:
                health['status'] = 'unhealthy'
                health['issues'].append('BM25 model not fitted')
        
        return health