import json
import pandas as pd
from typing import List, Dict, Any, Union
import re
from transformers import AutoTokenizer
import requests

class LLMChunker:
    """Creates semantic chunks using schema information and LLM guidance"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", max_chunk_size: int = 512):
        self.model_name = model_name
        self.max_chunk_size = max_chunk_size
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            # Fallback to a simple tokenizer if model not available
            self.tokenizer = None
    
    def chunk_json_data(self, data: Union[Dict, List], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk JSON data based on schema and semantic boundaries"""
        chunks = []
        
        if isinstance(data, list):
            # Group related items based on schema
            current_chunk = []
            current_size = 0
            
            for item in data:
                item_text = json.dumps(item, ensure_ascii=False)
                item_size = self._estimate_token_count(item_text)
                
                if current_size + item_size > self.max_chunk_size and current_chunk:
                    # Create chunk from current items
                    chunk_metadata = self._create_chunk_metadata(current_chunk, schema, 'json_array')
                    chunks.append({
                        'content': current_chunk,
                        'metadata': chunk_metadata,
                        'chunk_type': 'json_array',
                        'token_count': current_size
                    })
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(item)
                current_size += item_size
            
            # Add remaining items
            if current_chunk:
                chunk_metadata = self._create_chunk_metadata(current_chunk, schema, 'json_array')
                chunks.append({
                    'content': current_chunk,
                    'metadata': chunk_metadata,
                    'chunk_type': 'json_array',
                    'token_count': current_size
                })
        
        elif isinstance(data, dict):
            # Chunk large dictionaries by grouping related keys
            key_groups = self._group_related_keys(data, schema)
            
            for group_name, keys in key_groups.items():
                chunk_data = {k: data[k] for k in keys if k in data}
                chunk_text = json.dumps(chunk_data, ensure_ascii=False)
                token_count = self._estimate_token_count(chunk_text)
                
                chunk_metadata = self._create_chunk_metadata(chunk_data, schema, 'json_object')
                chunk_metadata['key_group'] = group_name
                
                chunks.append({
                    'content': chunk_data,
                    'metadata': chunk_metadata,
                    'chunk_type': 'json_object',
                    'token_count': token_count
                })
        
        return chunks
    
    def chunk_csv_data(self, df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk CSV data based on semantic relationships and size constraints"""
        chunks = []
        
        try:
            # Determine chunking strategy based on data size and schema
            total_rows = len(df)
            
            if total_rows <= 100:
                # Small dataset - create single chunk
                chunk_metadata = self._create_csv_chunk_metadata(df, schema, 0, total_rows)
                
                # Safe conversion to records
                try:
                    content = df.to_dict('records')
                except:
                    # Fallback: convert to string representation
                    content = df.to_string()
                
                # Safe token count estimation
                try:
                    token_count = self._estimate_token_count(df.to_string())
                except:
                    token_count = len(str(df)) // 4  # Rough estimate
                
                chunks.append({
                    'content': content,
                    'metadata': chunk_metadata,
                    'chunk_type': 'csv_complete',
                    'token_count': token_count
                })
            else:
                # Large dataset - chunk by rows with semantic boundaries
                chunk_size = self._calculate_optimal_chunk_size(df, schema)
                
                for start_idx in range(0, total_rows, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_rows)
                    chunk_df = df.iloc[start_idx:end_idx]
                    
                    chunk_metadata = self._create_csv_chunk_metadata(chunk_df, schema, start_idx, end_idx)
                    
                    # Safe conversion to records
                    try:
                        content = chunk_df.to_dict('records')
                    except:
                        content = chunk_df.to_string()
                    
                    # Safe token count estimation
                    try:
                        token_count = self._estimate_token_count(chunk_df.to_string())
                    except:
                        token_count = len(str(chunk_df)) // 4
                    
                    chunks.append({
                        'content': content,
                        'metadata': chunk_metadata,
                        'chunk_type': 'csv_rows',
                        'token_count': token_count
                    })
        
        except Exception as e:
            # Fallback: create a single chunk with error handling
            print(f"Warning: Error chunking CSV data: {str(e)}")
            try:
                content = df.to_string()
                chunk_metadata = {
                    'schema_type': 'tabular',
                    'chunk_type': 'csv_fallback',
                    'row_count': len(df),
                    'columns': list(df.columns),
                    'error': str(e)
                }
                chunks.append({
                    'content': content,
                    'metadata': chunk_metadata,
                    'chunk_type': 'csv_fallback',
                    'token_count': len(content) // 4
                })
            except Exception as fallback_error:
                print(f"Critical error: Could not process CSV data: {str(fallback_error)}")
                return []
        
        return chunks
    
    def chunk_text_data(self, content: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text data using semantic boundaries and LLM-guided splitting"""
        chunks = []
        
        # First, try to identify natural boundaries
        paragraphs = self._split_into_paragraphs(content)
        
        current_chunk = ""
        current_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph_tokens = self._estimate_token_count(paragraph)
            current_tokens = self._estimate_token_count(current_chunk)
            
            if current_tokens + paragraph_tokens > self.max_chunk_size and current_chunk:
                # Create chunk from current content
                chunk_metadata = self._create_text_chunk_metadata(current_chunk, schema, current_paragraphs)
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': chunk_metadata,
                    'chunk_type': 'text_semantic',
                    'token_count': current_tokens
                })
                current_chunk = ""
                current_paragraphs = []
            
            current_chunk += paragraph + "\n\n"
            current_paragraphs.append(paragraph)
        
        # Add remaining content
        if current_chunk.strip():
            chunk_metadata = self._create_text_chunk_metadata(current_chunk, schema, current_paragraphs)
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': chunk_metadata,
                'chunk_type': 'text_semantic',
                'token_count': self._estimate_token_count(current_chunk)
            })
        
        return chunks
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text"""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except:
                pass
        
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def _group_related_keys(self, data: Dict, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Group related keys based on schema analysis"""
        groups = {'metadata': [], 'content': [], 'identifiers': [], 'other': []}
        
        for key in data.keys():
            key_lower = key.lower()
            
            if any(identifier in key_lower for identifier in ['id', 'uuid', 'key', 'index']):
                groups['identifiers'].append(key)
            elif any(meta in key_lower for meta in ['created', 'updated', 'modified', 'timestamp', 'date']):
                groups['metadata'].append(key)
            elif any(content in key_lower for content in ['text', 'content', 'description', 'body', 'message']):
                groups['content'].append(key)
            else:
                groups['other'].append(key)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _calculate_optimal_chunk_size(self, df: pd.DataFrame, schema: Dict[str, Any]) -> int:
        """Calculate optimal chunk size for CSV data"""
        try:
            # Estimate average row size
            sample_rows = min(10, len(df))
            if sample_rows == 0:
                return 1
            
            sample_df = df.head(sample_rows)
            sample_text = sample_df.to_string()
            avg_row_tokens = self._estimate_token_count(sample_text) / sample_rows
            
            if avg_row_tokens <= 0:
                return 50  # Default chunk size
            
            # Calculate chunk size to stay under token limit
            optimal_rows = max(1, int(self.max_chunk_size / avg_row_tokens * 0.8))  # 80% buffer
            return min(optimal_rows, 1000)  # Cap at 1000 rows per chunk
            
        except Exception as e:
            print(f"Warning: Error calculating chunk size: {str(e)}")
            return 50  # Default fallback
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Split text into semantic paragraphs"""
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Further split very long paragraphs
        result = []
        for paragraph in paragraphs:
            if self._estimate_token_count(paragraph) > self.max_chunk_size:
                # Split long paragraphs by sentences
                sentences = re.split(r'[.!?]+\s+', paragraph)
                current_para = ""
                
                for sentence in sentences:
                    if self._estimate_token_count(current_para + sentence) > self.max_chunk_size and current_para:
                        result.append(current_para.strip())
                        current_para = sentence
                    else:
                        current_para += sentence + ". "
                
                if current_para.strip():
                    result.append(current_para.strip())
            else:
                result.append(paragraph.strip())
        
        return [p for p in result if p.strip()]
    
    def _create_chunk_metadata(self, content: Any, schema: Dict[str, Any], chunk_type: str) -> Dict[str, Any]:
        """Create metadata for a chunk"""
        metadata = {
            'schema_type': schema.get('type', 'unknown'),
            'chunk_type': chunk_type,
            'content_summary': self._summarize_content(content, chunk_type)
        }
        
        if chunk_type == 'json_array' and isinstance(content, list):
            metadata['item_count'] = len(content)
            if content and isinstance(content[0], dict):
                metadata['common_keys'] = list(content[0].keys())
        
        return metadata
    
    def _create_csv_chunk_metadata(self, df: pd.DataFrame, schema: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Create metadata for CSV chunk"""
        return {
            'schema_type': 'tabular',
            'chunk_type': 'csv_rows',
            'row_range': [start_idx, end_idx],
            'row_count': len(df),
            'columns': list(df.columns),
            'column_count': len(df.columns)
        }
    
    def _create_text_chunk_metadata(self, content: str, schema: Dict[str, Any], paragraphs: List[str]) -> Dict[str, Any]:
        """Create metadata for text chunk"""
        return {
            'schema_type': 'text',
            'chunk_type': 'text_semantic',
            'paragraph_count': len(paragraphs),
            'character_count': len(content),
            'word_count': len(content.split()),
            'has_structure': schema.get('properties', {}).get('structure_analysis', {})
        }
    
    def _summarize_content(self, content: Any, chunk_type: str) -> str:
        """Create a brief summary of chunk content"""
        if chunk_type == 'json_array' and isinstance(content, list):
            return f"Array of {len(content)} items"
        elif chunk_type == 'json_object' and isinstance(content, dict):
            return f"Object with keys: {', '.join(list(content.keys())[:5])}"
        elif chunk_type.startswith('text'):
            words = str(content).split()[:10]
            return f"Text starting with: {' '.join(words)}..."
        else:
            return f"Content of type {type(content).__name__}"
    
    def chunk_data(self, data: Any, schema: Dict[str, Any], file_type: str) -> List[Dict[str, Any]]:
        """Main method to chunk data based on type and schema"""
        if file_type == '.json':
            return self.chunk_json_data(data, schema)
        elif file_type == '.csv':
            return self.chunk_csv_data(data, schema)
        elif file_type == '.txt':
            return self.chunk_text_data(data, schema)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")