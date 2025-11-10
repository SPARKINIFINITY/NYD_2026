import numpy as np
import faiss
import json
import pickle
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import os

class EmbeddingStorage:
    """Creates embeddings for chunks and stores them in FAISS index"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Using random embeddings as fallback")
            self.embedding_model = None
        
        self.index = None
        self.chunk_metadata = []
        
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Create embeddings for text chunks"""
        texts = []
        
        for chunk in chunks:
            # Extract text content from chunk
            content = chunk.get('content', '')
            
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, (list, dict)):
                # Convert structured data to text
                texts.append(json.dumps(content, ensure_ascii=False))
            else:
                texts.append(str(content))
        
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
                return embeddings
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                return self._create_random_embeddings(len(texts))
        else:
            return self._create_random_embeddings(len(texts))
    
    def _create_random_embeddings(self, num_texts: int) -> np.ndarray:
        """Create random embeddings as fallback"""
        return np.random.rand(num_texts, self.dimension).astype('float32')
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
        """Create FAISS index from embeddings"""
        embeddings = embeddings.astype('float32')
        
        if index_type == "flat":
            # Simple flat index for exact search
            index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(100, max(1, embeddings.shape[0] // 10))  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train the index
            if embeddings.shape[0] >= nlist:
                index.train(embeddings)
            else:
                # Fallback to flat index if not enough data
                index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add embeddings to index
        index.add(embeddings)
        
        return index
    
    def store_embeddings_and_chunks(self, chunks: List[Dict[str, Any]], 
                                  output_dir: str = "embeddings_storage") -> Dict[str, str]:
        """Create embeddings, build FAISS index, and store everything"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index
        print("Building FAISS index...")
        index_type = "ivf" if len(chunks) > 100 else "flat"
        self.index = self.create_faiss_index(embeddings, index_type)
        
        # Store chunk metadata
        self.chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'chunk_id': i,
                'chunk_type': chunk.get('chunk_type', 'unknown'),
                'token_count': chunk.get('token_count', 0),
                'metadata': chunk.get('metadata', {}),
                'content_preview': self._create_content_preview(chunk.get('content', ''))
            }
            self.chunk_metadata.append(metadata)
        
        # Save files
        file_paths = {}
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        file_paths['faiss_index'] = index_path
        
        # Save embeddings
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        file_paths['embeddings'] = embeddings_path
        
        # Save chunk metadata
        metadata_path = os.path.join(output_dir, "chunk_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)
        file_paths['metadata'] = metadata_path
        
        # Save chunks data
        chunks_path = os.path.join(output_dir, "chunks_data.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        file_paths['chunks'] = chunks_path
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'num_chunks': len(chunks),
            'index_type': index_type,
            'total_embeddings': embeddings.shape[0]
        }
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        file_paths['config'] = config_path
        
        print(f"Embeddings and index stored in: {output_dir}")
        return file_paths
    
    def load_embeddings_and_index(self, storage_dir: str) -> bool:
        """Load previously stored embeddings and FAISS index"""
        try:
            # Load configuration
            config_path = os.path.join(storage_dir, "config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.model_name = config['model_name']
            self.dimension = config['dimension']
            
            # Load FAISS index
            index_path = os.path.join(storage_dir, "faiss_index.bin")
            self.index = faiss.read_index(index_path)
            
            # Load chunk metadata
            metadata_path = os.path.join(storage_dir, "chunk_metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
            
            print(f"Successfully loaded embeddings from: {storage_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using the query"""
        if not self.index or not self.embedding_model:
            print("Index or embedding model not available")
            return []
        
        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunk_metadata):
                    result = {
                        'rank': i + 1,
                        'chunk_id': idx,
                        'similarity_score': 1 / (1 + distance),  # Convert distance to similarity
                        'distance': float(distance),
                        'metadata': self.chunk_metadata[idx]
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def _create_content_preview(self, content: Any, max_length: int = 200) -> str:
        """Create a preview of chunk content"""
        if isinstance(content, str):
            preview = content[:max_length]
            if len(content) > max_length:
                preview += "..."
            return preview
        elif isinstance(content, (list, dict)):
            json_str = json.dumps(content, ensure_ascii=False)
            preview = json_str[:max_length]
            if len(json_str) > max_length:
                preview += "..."
            return preview
        else:
            return str(content)[:max_length]
    
    def get_storage_statistics(self, storage_dir: str) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        stats = {}
        
        try:
            # Load configuration
            config_path = os.path.join(storage_dir, "config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            stats['config'] = config
            
            # File sizes
            for filename in ['faiss_index.bin', 'embeddings.npy', 'chunk_metadata.json', 'chunks_data.pkl']:
                filepath = os.path.join(storage_dir, filename)
                if os.path.exists(filepath):
                    stats[f'{filename}_size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
            
            # Metadata statistics
            metadata_path = os.path.join(storage_dir, "chunk_metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            chunk_types = {}
            total_tokens = 0
            
            for chunk_meta in metadata:
                chunk_type = chunk_meta.get('chunk_type', 'unknown')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                total_tokens += chunk_meta.get('token_count', 0)
            
            stats['chunk_type_distribution'] = chunk_types
            stats['total_tokens'] = total_tokens
            stats['average_tokens_per_chunk'] = total_tokens / len(metadata) if metadata else 0
            
        except Exception as e:
            stats['error'] = str(e)
        
        return stats