import pandas as pd
import json
from typing import Dict, List, Any, Union
from collections import Counter
import re

class SchemaExtractor:
    """Extracts schema information from loaded data"""
    
    def __init__(self):
        pass
    
    def extract_json_schema(self, data: Union[Dict, List]) -> Dict[str, Any]:
        """Extract schema from JSON data"""
        schema = {
            'type': 'object',
            'properties': {},
            'data_structure': type(data).__name__
        }
        
        if isinstance(data, list) and len(data) > 0:
            # Analyze first few items to infer schema
            sample_size = min(10, len(data))
            sample_data = data[:sample_size]
            
            # Collect all keys from sample
            all_keys = set()
            for item in sample_data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            
            # Analyze each key
            for key in all_keys:
                values = []
                for item in sample_data:
                    if isinstance(item, dict) and key in item:
                        values.append(item[key])
                
                schema['properties'][key] = self._analyze_field_type(values)
        
        elif isinstance(data, dict):
            for key, value in data.items():
                schema['properties'][key] = self._analyze_field_type([value])
        
        return schema
    
    def extract_csv_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract schema from CSV DataFrame"""
        schema = {
            'type': 'tabular',
            'properties': {},
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        for column in df.columns:
            try:
                column_data = df[column].dropna()
                
                # Safe conversion of sample values
                sample_values = []
                if len(column_data) > 0:
                    try:
                        sample_values = column_data.head(5).tolist()
                    except:
                        # Fallback for complex data types
                        sample_values = [str(val) for val in column_data.head(5)]
                
                schema['properties'][column] = {
                    'dtype': str(df[column].dtype),
                    'null_count': int(df[column].isnull().sum()),
                    'unique_count': int(df[column].nunique()),
                    'sample_values': sample_values
                }
                
                # Additional analysis for different data types
                if df[column].dtype == 'object':
                    # Text analysis with safe operations
                    try:
                        avg_length = column_data.astype(str).str.len().mean()
                        schema['properties'][column]['avg_text_length'] = float(avg_length) if pd.notna(avg_length) else 0.0
                        
                        # Convert to list for text type inference
                        text_values = column_data.astype(str).tolist()
                        schema['properties'][column]['inferred_type'] = self._infer_text_type(text_values)
                    except Exception as e:
                        schema['properties'][column]['avg_text_length'] = 0.0
                        schema['properties'][column]['inferred_type'] = 'text'
                        
                elif pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        min_val = df[column].min()
                        max_val = df[column].max()
                        mean_val = df[column].mean()
                        
                        schema['properties'][column]['min_value'] = float(min_val) if pd.notna(min_val) else None
                        schema['properties'][column]['max_value'] = float(max_val) if pd.notna(max_val) else None
                        schema['properties'][column]['mean_value'] = float(mean_val) if pd.notna(mean_val) else None
                    except Exception as e:
                        schema['properties'][column]['min_value'] = None
                        schema['properties'][column]['max_value'] = None
                        schema['properties'][column]['mean_value'] = None
                        
            except Exception as e:
                # Fallback for problematic columns
                schema['properties'][column] = {
                    'dtype': str(df[column].dtype),
                    'null_count': 0,
                    'unique_count': 0,
                    'sample_values': [],
                    'error': str(e)
                }
        
        return schema
    
    def extract_txt_schema(self, content: str) -> Dict[str, Any]:
        """Extract schema from text content"""
        lines = content.split('\n')
        words = content.split()
        
        schema = {
            'type': 'text',
            'properties': {
                'total_characters': len(content),
                'total_words': len(words),
                'total_lines': len(lines),
                'avg_words_per_line': len(words) / len(lines) if lines else 0,
                'avg_chars_per_word': len(content) / len(words) if words else 0
            }
        }
        
        # Text structure analysis
        schema['properties']['structure_analysis'] = self._analyze_text_structure(content)
        
        return schema
    
    def _analyze_field_type(self, values: List[Any]) -> Dict[str, Any]:
        """Analyze the type and characteristics of field values"""
        if not values:
            return {'type': 'unknown', 'null_count': 0}
        
        non_null_values = [v for v in values if v is not None]
        type_counter = Counter(type(v).__name__ for v in non_null_values)
        
        analysis = {
            'type': type_counter.most_common(1)[0][0] if type_counter else 'null',
            'null_count': len(values) - len(non_null_values),
            'type_distribution': dict(type_counter),
            'sample_values': non_null_values[:3]
        }
        
        # Additional analysis for strings
        if analysis['type'] == 'str':
            str_values = [v for v in non_null_values if isinstance(v, str)]
            if str_values:
                analysis['avg_length'] = sum(len(s) for s in str_values) / len(str_values)
                analysis['inferred_subtype'] = self._infer_text_type(str_values)
        
        return analysis
    
    def _infer_text_type(self, text_values) -> str:
        """Infer the subtype of text data"""
        # Handle pandas Series or list
        if hasattr(text_values, 'tolist'):
            text_values = text_values.tolist()
        
        if not text_values:
            return 'unknown'
        
        # Convert to strings and take sample
        sample = [str(v) for v in text_values[:10]]  # Analyze first 10 values
        
        # Check for common patterns
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        url_pattern = re.compile(r'^https?://')
        phone_pattern = re.compile(r'^\+?[\d\s\-\(\)]{10,}$')
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}$')
        
        try:
            email_count = sum(1 for v in sample if email_pattern.match(str(v)))
            url_count = sum(1 for v in sample if url_pattern.match(str(v)))
            phone_count = sum(1 for v in sample if phone_pattern.match(str(v)))
            date_count = sum(1 for v in sample if date_pattern.match(str(v)))
            
            total = len(sample)
            
            if total == 0:
                return 'unknown'
            
            if email_count / total > 0.8:
                return 'email'
            elif url_count / total > 0.8:
                return 'url'
            elif phone_count / total > 0.8:
                return 'phone'
            elif date_count / total > 0.8:
                return 'date'
            else:
                avg_length = sum(len(str(v)) for v in sample) / len(sample)
                if avg_length > 100:
                    return 'long_text'
                elif avg_length < 20:
                    return 'short_text'
                else:
                    return 'medium_text'
        except Exception as e:
            return 'text'
    
    def _analyze_text_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of text content"""
        lines = content.split('\n')
        
        # Check for structured formats
        structure = {
            'has_headers': False,
            'has_bullet_points': False,
            'has_numbered_lists': False,
            'paragraph_count': 0,
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
        }
        
        # Simple heuristics for structure detection
        bullet_indicators = ['•', '-', '*', '◦']
        numbered_pattern = re.compile(r'^\s*\d+\.')
        
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(indicator) for indicator in bullet_indicators):
                structure['has_bullet_points'] = True
            if numbered_pattern.match(stripped):
                structure['has_numbered_lists'] = True
        
        # Count paragraphs (simple heuristic: empty lines separate paragraphs)
        in_paragraph = False
        for line in lines:
            if line.strip():
                if not in_paragraph:
                    structure['paragraph_count'] += 1
                    in_paragraph = True
            else:
                in_paragraph = False
        
        return structure
    
    def extract_schema(self, data: Any, file_type: str) -> Dict[str, Any]:
        """Main method to extract schema based on data type"""
        if file_type == '.json':
            return self.extract_json_schema(data)
        elif file_type == '.csv':
            return self.extract_csv_schema(data)
        elif file_type == '.txt':
            return self.extract_txt_schema(data)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")