import json
import csv
import pandas as pd
from typing import Dict, List, Any, Union
import os

class FileLoader:
    """Handles loading of JSON, CSV, and TXT files"""
    
    def __init__(self):
        self.supported_formats = ['.json', '.csv', '.txt']
    
    def load_json(self, file_path: str) -> Union[Dict, List]:
        """Load JSON file and return parsed data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception as e:
            raise Exception(f"Error loading JSON file {file_path}: {str(e)}")
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file and return pandas DataFrame"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV file {file_path}: {str(e)}")
    
    def load_txt(self, file_path: str) -> str:
        """Load TXT file and return content as string"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            raise Exception(f"Error loading TXT file {file_path}: {str(e)}")
    
    def load_file(self, file_path: str) -> Any:
        """Auto-detect file type and load accordingly"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            return self.load_json(file_path)
        elif file_ext == '.csv':
            return self.load_csv(file_path)
        elif file_ext == '.txt':
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic information about the loaded file"""
        data = self.load_file(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        info = {
            'file_path': file_path,
            'file_type': file_ext,
            'file_size': os.path.getsize(file_path)
        }
        
        if file_ext == '.json':
            info['data_type'] = type(data).__name__
            if isinstance(data, list):
                info['record_count'] = len(data)
            elif isinstance(data, dict):
                info['keys'] = list(data.keys())
        elif file_ext == '.csv':
            info['rows'] = len(data)
            info['columns'] = list(data.columns)
        elif file_ext == '.txt':
            info['character_count'] = len(data)
            info['word_count'] = len(data.split())
        
        return info