"""
Prompt Template Manager

Manages prompt templates for different query types and answer generation patterns.
Ensures consistent formatting and citation rules across different LLMs.
"""

from typing import Dict, List, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class PromptTemplateManager:
    """Manages prompt templates for different query types and models"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.citation_rules = self._initialize_citation_rules()
        self.format_constraints = self._initialize_format_constraints()
        
        logger.info("Initialized prompt template manager")
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize prompt templates for different query types"""
        
        templates = {
            "fact": {
                "system_prompt": """You are a helpful AI assistant that provides accurate, factual answers based on given evidence. 
Always cite your sources using [doc_id] format and provide confidence scores.""",
                
                "user_template": """Question: {query}

Evidence Documents:
{evidence_context}

Instructions:
1. Provide a brief summary (1-2 sentences)
2. Give a detailed answer based ONLY on the provided evidence
3. Include citations in [doc_id] format after each claim
4. Estimate confidence (0.0-1.0) based on evidence quality
5. Format response as valid JSON

Required JSON format:
{{
    "summary": "Brief factual summary",
    "detailed_answer": "Detailed answer with citations [doc_1]",
    "references": [
        {{
            "doc_id": "doc_1",
            "span": "relevant text from document",
            "relevance_score": 0.95
        }}
    ],
    "confidence": 0.87
}}

Answer:"""
            },
            
            "explain": {
                "system_prompt": """You are an expert educator who explains complex topics clearly using provided evidence. 
Always ground your explanations in the given sources and cite them properly.""",
                
                "user_template": """Question: {query}

Evidence Documents:
{evidence_context}

Instructions:
1. Provide a clear, educational explanation
2. Break down complex concepts step by step
3. Use evidence from the provided documents
4. Include citations [doc_id] for each major point
5. Assess confidence based on evidence completeness

Required JSON format:
{{
    "summary": "Brief explanation overview",
    "detailed_answer": "Step-by-step explanation with citations [doc_1]",
    "references": [
        {{
            "doc_id": "doc_1", 
            "span": "supporting evidence text",
            "relevance_score": 0.90
        }}
    ],
    "confidence": 0.82
}}

Answer:"""
            },
            
            "compare": {
                "system_prompt": """You are an analytical assistant who provides balanced comparisons based on evidence. 
Present multiple perspectives fairly and cite all sources used.""",
                
                "user_template": """Question: {query}

Evidence Documents:
{evidence_context}

Instructions:
1. Identify the items being compared
2. Present balanced comparison using provided evidence
3. Highlight similarities and differences
4. Cite sources for each comparison point [doc_id]
5. Confidence should reflect evidence balance

Required JSON format:
{{
    "summary": "Brief comparison overview",
    "detailed_answer": "Detailed comparison with citations [doc_1] vs [doc_2]",
    "references": [
        {{
            "doc_id": "doc_1",
            "span": "evidence for first item",
            "relevance_score": 0.88
        }},
        {{
            "doc_id": "doc_2", 
            "span": "evidence for second item",
            "relevance_score": 0.85
        }}
    ],
    "confidence": 0.79
}}

Answer:"""
            },
            
            "table": {
                "system_prompt": """You are a data analyst who presents information in structured formats. 
Use evidence to create clear, organized responses with proper citations.""",
                
                "user_template": """Question: {query}

Evidence Documents:
{evidence_context}

Instructions:
1. Organize information from evidence into structured format
2. Present data clearly and systematically
3. Include citations for each data point [doc_id]
4. Confidence based on data completeness and accuracy

Required JSON format:
{{
    "summary": "Brief overview of data/table content",
    "detailed_answer": "Structured presentation with citations [doc_1]",
    "references": [
        {{
            "doc_id": "doc_1",
            "span": "data source text",
            "relevance_score": 0.92
        }}
    ],
    "confidence": 0.85
}}

Answer:"""
            },
            
            "code": {
                "system_prompt": """You are a programming expert who provides code solutions and explanations based on evidence. 
Always explain your code and cite relevant documentation or examples.""",
                
                "user_template": """Question: {query}

Evidence Documents:
{evidence_context}

Instructions:
1. Provide code solution based on evidence
2. Explain the code functionality
3. Include best practices from evidence
4. Cite documentation sources [doc_id]
5. Confidence based on code accuracy and completeness

Required JSON format:
{{
    "summary": "Brief code solution overview",
    "detailed_answer": "Code with explanation and citations [doc_1]",
    "references": [
        {{
            "doc_id": "doc_1",
            "span": "relevant code/documentation",
            "relevance_score": 0.94
        }}
    ],
    "confidence": 0.88
}}

Answer:"""
            },
            
            "multi-hop": {
                "system_prompt": """You are a research assistant who synthesizes information from multiple sources to answer complex questions. 
Connect information across documents and show your reasoning chain.""",
                
                "user_template": """Question: {query}

Evidence Documents:
{evidence_context}

Instructions:
1. Identify the reasoning chain needed
2. Connect information across multiple documents
3. Show step-by-step logical progression
4. Cite each step with appropriate sources [doc_id]
5. Confidence reflects reasoning chain strength

Required JSON format:
{{
    "summary": "Brief synthesis of multi-step reasoning",
    "detailed_answer": "Step-by-step reasoning with citations [doc_1] → [doc_2] → [doc_3]",
    "references": [
        {{
            "doc_id": "doc_1",
            "span": "first piece of evidence",
            "relevance_score": 0.89
        }},
        {{
            "doc_id": "doc_2",
            "span": "connecting evidence", 
            "relevance_score": 0.86
        }}
    ],
    "confidence": 0.75
}}

Answer:"""
            },
            
            "clarify": {
                "system_prompt": """You are a helpful assistant who provides clarifications based on context and evidence. 
Use previous context and new evidence to provide clear, focused answers.""",
                
                "user_template": """Question: {query}

Previous Context: {previous_context}

Evidence Documents:
{evidence_context}

Instructions:
1. Consider the previous conversation context
2. Provide focused clarification using evidence
3. Address the specific clarification needed
4. Cite relevant sources [doc_id]
5. Confidence based on context alignment

Required JSON format:
{{
    "summary": "Brief clarification summary",
    "detailed_answer": "Focused clarification with citations [doc_1]",
    "references": [
        {{
            "doc_id": "doc_1",
            "span": "clarifying evidence",
            "relevance_score": 0.91
        }}
    ],
    "confidence": 0.83
}}

Answer:"""
            },
            
            "irrelevant": {
                "system_prompt": """You are a polite assistant who handles off-topic or irrelevant queries appropriately.""",
                
                "user_template": """Question: {query}

Instructions:
Politely indicate that the question is outside your knowledge domain or not relevant to the available evidence.

Required JSON format:
{{
    "summary": "Query not relevant to available knowledge",
    "detailed_answer": "I don't have relevant information to answer this question based on the available evidence.",
    "references": [],
    "confidence": 0.95
}}

Answer:"""
            }
        }
        
        return templates
    
    def _initialize_citation_rules(self) -> Dict[str, Any]:
        """Initialize citation formatting rules"""
        
        return {
            "citation_format": "[doc_id]",
            "citation_placement": "after_claim",
            "multiple_citations": "[doc_1, doc_2]",
            "citation_validation": {
                "required_fields": ["doc_id", "span", "relevance_score"],
                "doc_id_pattern": r"^[a-zA-Z0-9_-]+$",
                "relevance_score_range": [0.0, 1.0]
            },
            "span_extraction": {
                "max_length": 200,
                "min_length": 10,
                "preserve_context": True
            }
        }
    
    def _initialize_format_constraints(self) -> Dict[str, Any]:
        """Initialize output format constraints"""
        
        return {
            "required_fields": ["summary", "detailed_answer", "references", "confidence"],
            "field_constraints": {
                "summary": {
                    "max_length": 300,
                    "min_length": 20,
                    "format": "sentence"
                },
                "detailed_answer": {
                    "max_length": 2000,
                    "min_length": 50,
                    "must_contain_citations": True
                },
                "references": {
                    "type": "array",
                    "min_items": 0,
                    "max_items": 10
                },
                "confidence": {
                    "type": "float",
                    "min_value": 0.0,
                    "max_value": 1.0
                }
            },
            "json_schema": {
                "type": "object",
                "required": ["summary", "detailed_answer", "references", "confidence"],
                "properties": {
                    "summary": {"type": "string"},
                    "detailed_answer": {"type": "string"},
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["doc_id", "span", "relevance_score"],
                            "properties": {
                                "doc_id": {"type": "string"},
                                "span": {"type": "string"},
                                "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            }
                        }
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            }
        }
    
    def get_prompt(self, 
                   query: str, 
                   intent: str, 
                   evidence_documents: List[Dict[str, Any]], 
                   previous_context: str = None) -> Dict[str, str]:
        """
        Generate prompt for given query and intent
        
        Args:
            query: User query
            intent: Query intent (fact, explain, compare, etc.)
            evidence_documents: List of evidence documents
            previous_context: Previous conversation context (for clarify intent)
        
        Returns:
            Dictionary with system_prompt and user_prompt
        """
        
        # Get template for intent
        template = self.templates.get(intent, self.templates["fact"])
        
        # Format evidence context
        evidence_context = self._format_evidence_context(evidence_documents)
        
        # Create user prompt
        user_prompt = template["user_template"].format(
            query=query,
            evidence_context=evidence_context,
            previous_context=previous_context or "No previous context"
        )
        
        return {
            "system_prompt": template["system_prompt"],
            "user_prompt": user_prompt
        }
    
    def _format_evidence_context(self, evidence_documents: List[Dict[str, Any]]) -> str:
        """Format evidence documents for prompt context"""
        
        if not evidence_documents:
            return "No evidence documents provided."
        
        formatted_evidence = []
        
        for i, doc in enumerate(evidence_documents):
            doc_id = doc.get('id', f'doc_{i+1}')
            content = doc.get('content', '')
            score = doc.get('similarity_score', 0.0)
            
            # Truncate very long content
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted_doc = f"Document ID: {doc_id}\nRelevance Score: {score:.3f}\nContent: {content}\n"
            formatted_evidence.append(formatted_doc)
        
        return "\n" + "="*50 + "\n".join(formatted_evidence) + "="*50
    
    def get_model_specific_prompt(self, 
                                 base_prompt: Dict[str, str], 
                                 model_name: str) -> Dict[str, str]:
        """Adapt prompt for specific model requirements"""
        
        model_adaptations = {
            "microsoft/DialoGPT": {
                "prefix": "Human: ",
                "suffix": "\nAssistant: ",
                "max_length": 1024
            },
            "facebook/blenderbot": {
                "prefix": "",
                "suffix": "",
                "max_length": 512
            },
            "google/flan-t5": {
                "prefix": "Answer the following question: ",
                "suffix": "",
                "max_length": 512
            },
            "EleutherAI/gpt-neo": {
                "prefix": "",
                "suffix": "\n\nAnswer:",
                "max_length": 2048
            },
            "mistralai/Mistral": {
                "prefix": "[INST] ",
                "suffix": " [/INST]",
                "max_length": 4096
            },
            "microsoft/Phi-3": {
                "prefix": "<|user|>\n",
                "suffix": "\n<|assistant|>\n",
                "max_length": 4096
            },
            "google/gemma": {
                "prefix": "<start_of_turn>user\n",
                "suffix": "\n<start_of_turn>model\n",
                "max_length": 8192
            },
            "TinyLlama/TinyLlama": {
                "prefix": "<|user|>\n",
                "suffix": "\n<|assistant|>\n",
                "max_length": 2048
            }
        }
        
        # Find matching adaptation
        adaptation = None
        for model_key, config in model_adaptations.items():
            if model_key in model_name:
                adaptation = config
                break
        
        if not adaptation:
            # Default adaptation
            adaptation = {"prefix": "", "suffix": "", "max_length": 1024}
        
        # Apply adaptation
        adapted_prompt = base_prompt.copy()
        
        # Modify user prompt
        user_prompt = adapted_prompt["user_prompt"]
        
        # Add prefix and suffix
        if adaptation["prefix"]:
            user_prompt = adaptation["prefix"] + user_prompt
        if adaptation["suffix"]:
            user_prompt = user_prompt + adaptation["suffix"]
        
        # Truncate if too long
        max_length = adaptation["max_length"]
        if len(user_prompt) > max_length:
            # Truncate from the middle (preserve beginning and end)
            start_keep = max_length // 3
            end_keep = max_length // 3
            middle_truncated = user_prompt[:start_keep] + "\n...[truncated]...\n" + user_prompt[-end_keep:]
            user_prompt = middle_truncated
        
        adapted_prompt["user_prompt"] = user_prompt
        
        return adapted_prompt
    
    def validate_output_format(self, generated_output: str) -> Dict[str, Any]:
        """Validate generated output against format constraints"""
        
        validation_result = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "parsed_json": None
        }
        
        try:
            # Try to parse JSON
            parsed_json = json.loads(generated_output)
            validation_result["parsed_json"] = parsed_json
            
            # Check required fields
            required_fields = self.format_constraints["required_fields"]
            for field in required_fields:
                if field not in parsed_json:
                    validation_result["errors"].append(f"Missing required field: {field}")
            
            # Validate field constraints
            field_constraints = self.format_constraints["field_constraints"]
            
            for field, constraints in field_constraints.items():
                if field in parsed_json:
                    value = parsed_json[field]
                    
                    # Check string length constraints
                    if isinstance(value, str):
                        if "max_length" in constraints and len(value) > constraints["max_length"]:
                            validation_result["warnings"].append(f"{field} exceeds max length")
                        if "min_length" in constraints and len(value) < constraints["min_length"]:
                            validation_result["errors"].append(f"{field} below min length")
                    
                    # Check numeric constraints
                    elif isinstance(value, (int, float)):
                        if "min_value" in constraints and value < constraints["min_value"]:
                            validation_result["errors"].append(f"{field} below min value")
                        if "max_value" in constraints and value > constraints["max_value"]:
                            validation_result["errors"].append(f"{field} above max value")
                    
                    # Check array constraints
                    elif isinstance(value, list):
                        if "min_items" in constraints and len(value) < constraints["min_items"]:
                            validation_result["warnings"].append(f"{field} has fewer items than recommended")
                        if "max_items" in constraints and len(value) > constraints["max_items"]:
                            validation_result["warnings"].append(f"{field} has more items than recommended")
            
            # Check citations in detailed_answer
            if "detailed_answer" in parsed_json:
                detailed_answer = parsed_json["detailed_answer"]
                if field_constraints["detailed_answer"].get("must_contain_citations", False):
                    if not re.search(r'\[doc_\w+\]', detailed_answer):
                        validation_result["warnings"].append("detailed_answer should contain citations")
            
            # Validate references format
            if "references" in parsed_json:
                references = parsed_json["references"]
                if isinstance(references, list):
                    for i, ref in enumerate(references):
                        if not isinstance(ref, dict):
                            validation_result["errors"].append(f"Reference {i} is not an object")
                            continue
                        
                        required_ref_fields = ["doc_id", "span", "relevance_score"]
                        for ref_field in required_ref_fields:
                            if ref_field not in ref:
                                validation_result["errors"].append(f"Reference {i} missing {ref_field}")
            
            # Set validity
            validation_result["is_valid"] = len(validation_result["errors"]) == 0
            
        except json.JSONDecodeError as e:
            validation_result["errors"].append(f"Invalid JSON format: {str(e)}")
        
        return validation_result
    
    def get_citation_rules(self) -> Dict[str, Any]:
        """Get citation formatting rules"""
        return self.citation_rules.copy()
    
    def get_format_constraints(self) -> Dict[str, Any]:
        """Get output format constraints"""
        return self.format_constraints.copy()
    
    def get_available_intents(self) -> List[str]:
        """Get list of available intent templates"""
        return list(self.templates.keys())
    
    def add_custom_template(self, intent: str, system_prompt: str, user_template: str):
        """Add custom template for specific intent"""
        
        self.templates[intent] = {
            "system_prompt": system_prompt,
            "user_template": user_template
        }
        
        logger.info(f"Added custom template for intent: {intent}")
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about available templates"""
        
        stats = {
            "total_templates": len(self.templates),
            "available_intents": list(self.templates.keys()),
            "avg_system_prompt_length": 0,
            "avg_user_template_length": 0
        }
        
        if self.templates:
            system_lengths = [len(t["system_prompt"]) for t in self.templates.values()]
            user_lengths = [len(t["user_template"]) for t in self.templates.values()]
            
            stats["avg_system_prompt_length"] = sum(system_lengths) / len(system_lengths)
            stats["avg_user_template_length"] = sum(user_lengths) / len(user_lengths)
        
        return stats