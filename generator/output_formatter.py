"""
Output Formatter

Ensures structured JSON output format with proper validation,
error handling, and format enforcement for grounded answers.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class OutputFormatter:
    """Formats and validates structured JSON output for grounded answers"""
    
    def __init__(self, 
                 strict_validation: bool = True,
                 max_summary_length: int = 300,
                 max_answer_length: int = 2000,
                 max_references: int = 10):
        
        self.strict_validation = strict_validation
        self.max_summary_length = max_summary_length
        self.max_answer_length = max_answer_length
        self.max_references = max_references
        
        # Output schema definition
        self.output_schema = self._define_output_schema()
        
        # Formatting statistics
        self.formatting_stats = {
            "total_formatted": 0,
            "successful_formats": 0,
            "validation_failures": 0,
            "auto_corrections": 0
        }
        
        logger.info("Initialized output formatter")
    
    def _define_output_schema(self) -> Dict[str, Any]:
        """Define the expected output JSON schema"""
        
        return {
            "type": "object",
            "required": ["summary", "detailed_answer", "references", "confidence"],
            "properties": {
                "summary": {
                    "type": "string",
                    "minLength": 10,
                    "maxLength": self.max_summary_length,
                    "description": "Brief summary of the answer"
                },
                "detailed_answer": {
                    "type": "string",
                    "minLength": 20,
                    "maxLength": self.max_answer_length,
                    "description": "Detailed grounded answer with citations"
                },
                "references": {
                    "type": "array",
                    "maxItems": self.max_references,
                    "items": {
                        "type": "object",
                        "required": ["doc_id", "span", "relevance_score"],
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "pattern": "^[a-zA-Z0-9_-]+$",
                                "description": "Document identifier"
                            },
                            "span": {
                                "type": "string",
                                "minLength": 5,
                                "maxLength": 500,
                                "description": "Relevant text span from document"
                            },
                            "relevance_score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Relevance score for the reference"
                            },
                            "span_start": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Optional start position in document"
                            },
                            "span_end": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Optional end position in document"
                            }
                        }
                    }
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the answer"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata about generation",
                    "properties": {
                        "model_used": {"type": "string"},
                        "generation_time": {"type": "number"},
                        "evidence_count": {"type": "integer"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
    
    def format_output(self, 
                     raw_answer: Dict[str, Any], 
                     evidence_documents: List[Dict[str, Any]] = None,
                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format raw answer into structured JSON output
        
        Args:
            raw_answer: Raw answer dictionary
            evidence_documents: Original evidence documents for validation
            metadata: Additional metadata to include
        
        Returns:
            Formatted and validated JSON output
        """
        
        self.formatting_stats["total_formatted"] += 1
        
        try:
            # Start with raw answer
            formatted_output = raw_answer.copy()
            
            # Apply formatting corrections
            formatted_output = self._apply_formatting_corrections(formatted_output)
            
            # Validate and fix structure
            formatted_output = self._validate_and_fix_structure(formatted_output, evidence_documents)
            
            # Add metadata if provided
            if metadata:
                formatted_output["metadata"] = self._format_metadata(metadata)
            
            # Final validation
            validation_result = self._validate_output(formatted_output)
            
            if validation_result["is_valid"] or not self.strict_validation:
                self.formatting_stats["successful_formats"] += 1
                return formatted_output
            else:
                # Create fallback output
                self.formatting_stats["validation_failures"] += 1
                return self._create_fallback_output(raw_answer, validation_result["errors"])
        
        except Exception as e:
            logger.error(f"Output formatting failed: {str(e)}")
            self.formatting_stats["validation_failures"] += 1
            return self._create_error_output(str(e))
    
    def _apply_formatting_corrections(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """Apply automatic formatting corrections"""
        
        corrected = answer.copy()
        corrections_made = 0
        
        # Fix summary
        if "summary" in corrected:
            original_summary = corrected["summary"]
            corrected["summary"] = self._fix_text_field(
                original_summary, 
                max_length=self.max_summary_length,
                ensure_sentence=True
            )
            if corrected["summary"] != original_summary:
                corrections_made += 1
        
        # Fix detailed answer
        if "detailed_answer" in corrected:
            original_answer = corrected["detailed_answer"]
            corrected["detailed_answer"] = self._fix_text_field(
                original_answer,
                max_length=self.max_answer_length,
                ensure_sentence=False
            )
            if corrected["detailed_answer"] != original_answer:
                corrections_made += 1
        
        # Fix references
        if "references" in corrected and isinstance(corrected["references"], list):
            original_refs = corrected["references"]
            corrected["references"] = self._fix_references(original_refs)
            if corrected["references"] != original_refs:
                corrections_made += 1
        
        # Fix confidence
        if "confidence" in corrected:
            original_confidence = corrected["confidence"]
            corrected["confidence"] = self._fix_confidence_score(original_confidence)
            if corrected["confidence"] != original_confidence:
                corrections_made += 1
        
        if corrections_made > 0:
            self.formatting_stats["auto_corrections"] += corrections_made
            logger.debug(f"Applied {corrections_made} automatic corrections")
        
        return corrected
    
    def _fix_text_field(self, text: str, max_length: int, ensure_sentence: bool = False) -> str:
        """Fix text field formatting"""
        
        if not isinstance(text, str):
            text = str(text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Truncate if too long
        if len(text) > max_length:
            # Try to truncate at sentence boundary
            sentences = re.split(r'[.!?]+', text)
            truncated = ""
            
            for sentence in sentences:
                if len(truncated + sentence) <= max_length - 3:
                    truncated += sentence + ". "
                else:
                    break
            
            if truncated:
                text = truncated.strip()
            else:
                # Hard truncate with ellipsis
                text = text[:max_length-3] + "..."
        
        # Ensure proper sentence ending if required
        if ensure_sentence and text and not text[-1] in '.!?':
            text += "."
        
        return text
    
    def _fix_references(self, references: List[Any]) -> List[Dict[str, Any]]:
        """Fix references array formatting"""
        
        fixed_references = []
        
        for ref in references[:self.max_references]:  # Limit number of references
            if not isinstance(ref, dict):
                continue
            
            fixed_ref = {}
            
            # Fix doc_id
            doc_id = ref.get('doc_id', '')
            if isinstance(doc_id, str) and doc_id:
                # Clean doc_id to match pattern
                fixed_ref['doc_id'] = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
            else:
                fixed_ref['doc_id'] = f'doc_{len(fixed_references) + 1}'
            
            # Fix span
            span = ref.get('span', '')
            if isinstance(span, str) and len(span) >= 5:
                fixed_ref['span'] = self._fix_text_field(span, max_length=500)
            else:
                fixed_ref['span'] = "No relevant span available"
            
            # Fix relevance_score
            relevance_score = ref.get('relevance_score', 0.0)
            fixed_ref['relevance_score'] = self._fix_confidence_score(relevance_score)
            
            # Optional fields
            if 'span_start' in ref and isinstance(ref['span_start'], (int, float)):
                fixed_ref['span_start'] = max(0, int(ref['span_start']))
            
            if 'span_end' in ref and isinstance(ref['span_end'], (int, float)):
                fixed_ref['span_end'] = max(0, int(ref['span_end']))
            
            fixed_references.append(fixed_ref)
        
        return fixed_references
    
    def _fix_confidence_score(self, score: Any) -> float:
        """Fix confidence score to be valid float between 0.0 and 1.0"""
        
        try:
            score = float(score)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5  # Default neutral confidence
    
    def _validate_and_fix_structure(self, 
                                  answer: Dict[str, Any], 
                                  evidence_documents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate and fix the overall structure"""
        
        fixed_answer = {}
        
        # Ensure required fields exist
        required_fields = ["summary", "detailed_answer", "references", "confidence"]
        
        for field in required_fields:
            if field in answer:
                fixed_answer[field] = answer[field]
            else:
                # Create default values for missing fields
                fixed_answer[field] = self._create_default_field_value(field, evidence_documents)
        
        # Copy optional fields
        optional_fields = ["metadata", "error"]
        for field in optional_fields:
            if field in answer:
                fixed_answer[field] = answer[field]
        
        return fixed_answer
    
    def _create_default_field_value(self, field: str, evidence_documents: List[Dict[str, Any]] = None) -> Any:
        """Create default value for missing field"""
        
        if field == "summary":
            return "Answer summary not available"
        
        elif field == "detailed_answer":
            return "Detailed answer not available"
        
        elif field == "references":
            # Create references from evidence documents if available
            if evidence_documents:
                references = []
                for i, doc in enumerate(evidence_documents[:3]):
                    references.append({
                        "doc_id": doc.get('id', f'doc_{i+1}'),
                        "span": doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                        "relevance_score": doc.get('similarity_score', 0.5)
                    })
                return references
            else:
                return []
        
        elif field == "confidence":
            return 0.1  # Low confidence for missing data
        
        else:
            return None
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format metadata section"""
        
        formatted_metadata = {}
        
        # Standard metadata fields
        if "model_used" in metadata:
            formatted_metadata["model_used"] = str(metadata["model_used"])
        
        if "generation_time" in metadata:
            try:
                formatted_metadata["generation_time"] = float(metadata["generation_time"])
            except (ValueError, TypeError):
                pass
        
        if "evidence_count" in metadata:
            try:
                formatted_metadata["evidence_count"] = int(metadata["evidence_count"])
            except (ValueError, TypeError):
                pass
        
        # Add timestamp
        formatted_metadata["timestamp"] = datetime.now().isoformat()
        
        # Copy other metadata fields
        for key, value in metadata.items():
            if key not in formatted_metadata:
                formatted_metadata[key] = value
        
        return formatted_metadata
    
    def _validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output against schema"""
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required fields
            required_fields = self.output_schema["required"]
            for field in required_fields:
                if field not in output:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["is_valid"] = False
            
            # Validate field types and constraints
            properties = self.output_schema["properties"]
            
            for field, constraints in properties.items():
                if field in output:
                    value = output[field]
                    field_validation = self._validate_field(field, value, constraints)
                    
                    validation_result["errors"].extend(field_validation["errors"])
                    validation_result["warnings"].extend(field_validation["warnings"])
                    
                    if field_validation["errors"]:
                        validation_result["is_valid"] = False
            
            # Additional business logic validation
            business_validation = self._validate_business_logic(output)
            validation_result["errors"].extend(business_validation["errors"])
            validation_result["warnings"].extend(business_validation["warnings"])
            
            if business_validation["errors"]:
                validation_result["is_valid"] = False
        
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    def _validate_field(self, field_name: str, value: Any, constraints: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate individual field against constraints"""
        
        result = {"errors": [], "warnings": []}
        
        # Type validation
        expected_type = constraints.get("type")
        if expected_type == "string" and not isinstance(value, str):
            result["errors"].append(f"{field_name} must be a string")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            result["errors"].append(f"{field_name} must be a number")
        elif expected_type == "array" and not isinstance(value, list):
            result["errors"].append(f"{field_name} must be an array")
        elif expected_type == "object" and not isinstance(value, dict):
            result["errors"].append(f"{field_name} must be an object")
        
        # String constraints
        if isinstance(value, str):
            if "minLength" in constraints and len(value) < constraints["minLength"]:
                result["errors"].append(f"{field_name} is too short (min: {constraints['minLength']})")
            
            if "maxLength" in constraints and len(value) > constraints["maxLength"]:
                result["warnings"].append(f"{field_name} is too long (max: {constraints['maxLength']})")
            
            if "pattern" in constraints:
                pattern = constraints["pattern"]
                if not re.match(pattern, value):
                    result["errors"].append(f"{field_name} does not match required pattern")
        
        # Number constraints
        if isinstance(value, (int, float)):
            if "minimum" in constraints and value < constraints["minimum"]:
                result["errors"].append(f"{field_name} is below minimum ({constraints['minimum']})")
            
            if "maximum" in constraints and value > constraints["maximum"]:
                result["errors"].append(f"{field_name} is above maximum ({constraints['maximum']})")
        
        # Array constraints
        if isinstance(value, list):
            if "maxItems" in constraints and len(value) > constraints["maxItems"]:
                result["warnings"].append(f"{field_name} has too many items (max: {constraints['maxItems']})")
            
            # Validate array items
            if "items" in constraints:
                item_constraints = constraints["items"]
                for i, item in enumerate(value):
                    item_validation = self._validate_field(f"{field_name}[{i}]", item, item_constraints)
                    result["errors"].extend(item_validation["errors"])
                    result["warnings"].extend(item_validation["warnings"])
        
        return result
    
    def _validate_business_logic(self, output: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate business logic rules"""
        
        result = {"errors": [], "warnings": []}
        
        # Check citation consistency
        detailed_answer = output.get("detailed_answer", "")
        references = output.get("references", [])
        
        if detailed_answer and references:
            # Extract citations from answer
            citation_pattern = r'\[([a-zA-Z0-9_-]+)\]'
            citations_in_text = set(re.findall(citation_pattern, detailed_answer))
            reference_doc_ids = set(ref.get('doc_id', '') for ref in references)
            
            # Check for orphaned citations
            orphaned_citations = citations_in_text - reference_doc_ids
            if orphaned_citations:
                result["warnings"].append(f"Citations without references: {list(orphaned_citations)}")
            
            # Check for unused references
            unused_references = reference_doc_ids - citations_in_text
            if unused_references:
                result["warnings"].append(f"References not cited in answer: {list(unused_references)}")
        
        # Check confidence reasonableness
        confidence = output.get("confidence", 0.0)
        if confidence > 0.95:
            result["warnings"].append("Very high confidence score may indicate overconfidence")
        elif confidence < 0.1:
            result["warnings"].append("Very low confidence score may indicate poor answer quality")
        
        return result
    
    def _create_fallback_output(self, raw_answer: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Create fallback output when validation fails"""
        
        fallback = {
            "summary": "Output validation failed",
            "detailed_answer": f"The generated answer could not be properly formatted. Errors: {'; '.join(errors)}",
            "references": [],
            "confidence": 0.1,
            "error": {
                "type": "validation_failure",
                "errors": errors,
                "raw_answer": raw_answer
            }
        }
        
        return fallback
    
    def _create_error_output(self, error_message: str) -> Dict[str, Any]:
        """Create error output when formatting completely fails"""
        
        return {
            "summary": "Formatting error occurred",
            "detailed_answer": f"An error occurred during output formatting: {error_message}",
            "references": [],
            "confidence": 0.0,
            "error": {
                "type": "formatting_error",
                "message": error_message
            }
        }
    
    def format_batch_output(self, 
                          raw_answers: List[Dict[str, Any]], 
                          evidence_batches: List[List[Dict[str, Any]]] = None,
                          metadata_batch: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Format batch of raw answers"""
        
        if not evidence_batches:
            evidence_batches = [None] * len(raw_answers)
        
        if not metadata_batch:
            metadata_batch = [None] * len(raw_answers)
        
        formatted_outputs = []
        
        for raw_answer, evidence_docs, metadata in zip(raw_answers, evidence_batches, metadata_batch):
            formatted_output = self.format_output(raw_answer, evidence_docs, metadata)
            formatted_outputs.append(formatted_output)
        
        return formatted_outputs
    
    def validate_json_string(self, json_string: str) -> Dict[str, Any]:
        """Validate JSON string format"""
        
        try:
            parsed_json = json.loads(json_string)
            return self._validate_output(parsed_json)
        except json.JSONDecodeError as e:
            return {
                "is_valid": False,
                "errors": [f"Invalid JSON format: {str(e)}"],
                "warnings": []
            }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the output schema definition"""
        return self.output_schema.copy()
    
    def update_schema_constraints(self, **kwargs):
        """Update schema constraints"""
        
        if "max_summary_length" in kwargs:
            self.max_summary_length = kwargs["max_summary_length"]
            self.output_schema["properties"]["summary"]["maxLength"] = self.max_summary_length
        
        if "max_answer_length" in kwargs:
            self.max_answer_length = kwargs["max_answer_length"]
            self.output_schema["properties"]["detailed_answer"]["maxLength"] = self.max_answer_length
        
        if "max_references" in kwargs:
            self.max_references = kwargs["max_references"]
            self.output_schema["properties"]["references"]["maxItems"] = self.max_references
        
        logger.info(f"Updated schema constraints: {kwargs}")
    
    def get_formatting_stats(self) -> Dict[str, Any]:
        """Get formatting statistics"""
        
        stats = self.formatting_stats.copy()
        
        # Calculate derived metrics
        if stats["total_formatted"] > 0:
            stats["success_rate"] = stats["successful_formats"] / stats["total_formatted"]
            stats["failure_rate"] = stats["validation_failures"] / stats["total_formatted"]
            stats["avg_corrections_per_output"] = stats["auto_corrections"] / stats["total_formatted"]
        
        return stats
    
    def clear_stats(self):
        """Clear formatting statistics"""
        
        self.formatting_stats = {
            "total_formatted": 0,
            "successful_formats": 0,
            "validation_failures": 0,
            "auto_corrections": 0
        }
        
        logger.info("Cleared formatting statistics")