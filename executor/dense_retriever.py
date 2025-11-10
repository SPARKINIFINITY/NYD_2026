"""
Dense Retriever - Embedding-based Semantic Search

Implements dense retrieval using sentence embeddings and FAISS index
for semantic similarity search.
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
import time
import os
import pickle

logger = logging.getLogger(__name__)

class DenseRetriever:
    """Dense retrieval using embeddings and FAISS"""
    
    def __init__(self, 
                 model_name: str = "all-mpnet-base-v2",
                 index_path: str = None,
                 embeddings_path: str = None,
                 metadata_path: str = None,
                 use_quantized: bool = False):
        
        self.model_name = model_name
        self.use_quantized = use_quantized
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            self.embedding_dim = 384  # Default dimension
        
        # FAISS index and data
        self.index = None
        self.document_embeddings = None
        self.document_metadata = []
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "cache_hits": 0
        }
        
        # Query cache for performance
        self.query_cache = {}
        self.max_cache_size = 1000
        
        # Load existing index if provided
        if index_path and os.path.exists(index_path):
            self.load_index(index_path, embeddings_path, metadata_path)
    
    def build_index(self, documents: List[Dict[str, Any]], save_path: str = None) -> bool:
        """
        Build FAISS index from documents
        
        Args:
            documents: List of documents with 'content' and metadata
            save_path: Optional path to save the index
        """
        if not self.embedding_model:
            logger.error("Embedding model not available")
            return False
        
        try:
            logger.info(f"Building dense index for {len(documents)} documents...")
            start_time = time.time()
            
            # Extract text content
            texts = []
            metadata = []
            
            for doc in documents:
                if isinstance(doc, dict):
                    content = doc.get('content', str(doc))
                    texts.append(content)
                    metadata.append(doc)
                else:
                    texts.append(str(doc))
                    metadata.append({'content': str(doc), 'id': len(metadata)})
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Create FAISS index
            if self.use_quantized:
                # Use quantized index for memory efficiency
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFPQ(
                    quantizer, 
                    self.embedding_dim, 
                    min(100, len(documents) // 10),  # Number of clusters
                    8,  # Number of bits per sub-vector
                    8   # Number of sub-vectors
                )
                self.index.train(embeddings.astype('float32'))
            else:
                # Use flat index for exact search
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Add embeddings to index
            self.index.add(embeddings.astype('float32'))
            self.document_embeddings = embeddings
            self.document_metadata = metadata
            
            build_time = time.time() - start_time
            logger.info(f"Dense index built in {build_time:.2f}s for {len(documents)} documents")
            
            # Save index if path provided
            if save_path:
                self.save_index(save_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build dense index: {e}")
            return False
    
    def save_index(self, base_path: str):
        """Save FAISS index and metadata"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{base_path}.faiss")
            
            # Save embeddings
            np.save(f"{base_path}_embeddings.npy", self.document_embeddings)
            
            # Save metadata
            with open(f"{base_path}_metadata.pkl", 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            logger.info(f"Dense index saved to {base_path}")
            
        except Exception as e:
            logger.error(f"Failed to save dense index: {e}")
    
    def load_index(self, index_path: str, embeddings_path: str = None, metadata_path: str = None):
        """Load FAISS index and metadata"""
        try:
            # Load FAISS index
            if index_path.endswith('.faiss'):
                self.index = faiss.read_index(index_path)
            else:
                self.index = faiss.read_index(f"{index_path}.faiss")
            
            # Load embeddings
            if embeddings_path:
                self.document_embeddings = np.load(embeddings_path)
            elif not index_path.endswith('.faiss'):
                self.document_embeddings = np.load(f"{index_path}_embeddings.npy")
            
            # Load metadata
            if metadata_path:
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            elif not index_path.endswith('.faiss'):
                with open(f"{index_path}_metadata.pkl", 'rb') as f:
                    self.document_metadata = pickle.load(f)
            
            logger.info(f"Dense index loaded from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dense index: {e}")
            return False
    
    def retrieve(self, query: str, k: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents using dense search
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Optional similarity threshold
        
        Returns:
            List of retrieved documents with scores
        """
        if not self.index or not self.embedding_model:
            logger.error("Dense retriever not properly initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{query}_{k}_{threshold}"
            if cache_key in self.query_cache:
                self.retrieval_stats["cache_hits"] += 1
                return self.query_cache[cache_key]
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, k)
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                # Convert distance to similarity score
                similarity = 1.0 / (1.0 + distance)
                
                # Apply threshold if specified
                if threshold and similarity < threshold:
                    continue
                
                # Get document metadata
                if idx < len(self.document_metadata):
                    doc_metadata = self.document_metadata[idx].copy()
                    doc_metadata.update({
                        'retriever_type': 'dense',
                        'similarity_score': float(similarity),
                        'distance': float(distance),
                        'rank': i + 1,
                        'retrieval_method': 'faiss_dense'
                    })
                    results.append(doc_metadata)
            
            # Update cache
            if len(self.query_cache) >= self.max_cache_size:
                # Remove oldest entry
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
            
            logger.debug(f"Dense retrieval completed in {retrieval_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def batch_retrieve(self, queries: List[str], k: int = 10) -> List[List[Dict[str, Any]]]:
        """Batch retrieval for multiple queries"""
        if not self.index or not self.embedding_model:
            return [[] for _ in queries]
        
        try:
            # Generate batch embeddings
            query_embeddings = self.embedding_model.encode(queries, convert_to_numpy=True)
            query_embeddings = query_embeddings.astype('float32')
            
            # Batch search
            distances, indices = self.index.search(query_embeddings, k)
            
            # Process results for each query
            batch_results = []
            for query_idx, (query_distances, query_indices) in enumerate(zip(distances, indices)):
                query_results = []
                for rank, (distance, idx) in enumerate(zip(query_distances, query_indices)):
                    if idx == -1 or idx >= len(self.document_metadata):
                        continue
                    
                    similarity = 1.0 / (1.0 + distance)
                    doc_metadata = self.document_metadata[idx].copy()
                    doc_metadata.update({
                        'retriever_type': 'dense',
                        'similarity_score': float(similarity),
                        'distance': float(distance),
                        'rank': rank + 1,
                        'query_index': query_idx
                    })
                    query_results.append(doc_metadata)
                
                batch_results.append(query_results)
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch dense retrieval failed: {e}")
            return [[] for _ in queries]
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        for doc in self.document_metadata:
            if doc.get('id') == doc_id:
                return doc
        return None
    
    def get_similar_documents(self, doc_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        # Find document index
        doc_idx = None
        for i, doc in enumerate(self.document_metadata):
            if doc.get('id') == doc_id:
                doc_idx = i
                break
        
        if doc_idx is None or not self.document_embeddings:
            return []
        
        try:
            # Get document embedding
            doc_embedding = self.document_embeddings[doc_idx:doc_idx+1].astype('float32')
            
            # Search for similar documents
            distances, indices = self.index.search(doc_embedding, k + 1)  # +1 to exclude self
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == doc_idx:  # Skip the document itself
                    continue
                
                if idx < len(self.document_metadata):
                    similarity = 1.0 / (1.0 + distance)
                    doc_metadata = self.document_metadata[idx].copy()
                    doc_metadata.update({
                        'similarity_score': float(similarity),
                        'distance': float(distance)
                    })
                    results.append(doc_metadata)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Similar document search failed: {e}")
            return []
    
    def update_document(self, doc_id: str, new_content: str) -> bool:
        """Update a document in the index"""
        # Find document index
        doc_idx = None
        for i, doc in enumerate(self.document_metadata):
            if doc.get('id') == doc_id:
                doc_idx = i
                break
        
        if doc_idx is None or not self.embedding_model:
            return False
        
        try:
            # Generate new embedding
            new_embedding = self.embedding_model.encode([new_content], convert_to_numpy=True)
            
            # Update embeddings array
            if self.document_embeddings is not None:
                self.document_embeddings[doc_idx] = new_embedding[0]
            
            # Update metadata
            self.document_metadata[doc_idx]['content'] = new_content
            
            # Rebuild index (for simplicity - could be optimized)
            if self.use_quantized:
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFPQ(
                    quantizer, self.embedding_dim, 
                    min(100, len(self.document_metadata) // 10), 8, 8
                )
                self.index.train(self.document_embeddings.astype('float32'))
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            self.index.add(self.document_embeddings.astype('float32'))
            
            # Clear cache
            self.query_cache.clear()
            
            logger.info(f"Updated document {doc_id} in dense index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        stats = self.retrieval_stats.copy()
        stats.update({
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'index_size': len(self.document_metadata) if self.document_metadata else 0,
            'cache_size': len(self.query_cache),
            'use_quantized': self.use_quantized,
            'index_type': type(self.index).__name__ if self.index else None
        })
        return stats
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Dense retriever cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Check retriever health"""
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        if not self.embedding_model:
            health['status'] = 'unhealthy'
            health['issues'].append('Embedding model not loaded')
        
        if not self.index:
            health['status'] = 'unhealthy'
            health['issues'].append('FAISS index not built')
        
        if not self.document_metadata:
            health['status'] = 'unhealthy'
            health['issues'].append('No documents indexed')
        
        return health