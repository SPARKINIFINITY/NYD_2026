"""
Citation Manager

Handles reference extraction, validation, and formatting for grounded answers.
Ensures proper citation format and evidence span extraction.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CitationManager:
    """Manages citations and reference extraction"""
    
    def __init__(self, 
                 citation_format: str = "[doc_id]",
                 max_span_length: int = 200,
                 min_span_length: int = 10):
        
        self.citation_format = citation_format
        self.max_span_length = max_span_length
        self.min_span_length = min_span_length
        
        # Citation patterns for different formats
        self.citation_patterns = {
            "[doc_id]": r'\[([a-zA-Z0-9_-]+)\]',
            "(doc_id)": r'\(([a-zA-Z0-9_-]+)\)',
            "doc_id:": r'([a-zA-Z0-9_-]+):',
            "^doc_id": r'\^([a-zA-Z0-9_-]+)'
        }
        
        # Statistics
        self.citation_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "invalid_citations": 0,
            "avg_citations_per_answer": 0.0
        }
        
        logger.info(f"Initialized citation manager with format: {citation_format}")
    
    def extract_citations_from_answer(self, answer_text: str) -> List[str]:
        """Extract citation IDs from answer text"""
        
        pattern = self.citation_patterns.get(self.citation_format, self.citation_patterns["[doc_id]"])
        citations = re.findall(pattern, answer_text)
        
        # Remove duplicates while preserving order
        unique_citations = []
        seen = set()
        for citation in citations:
            if citation not in seen:
                unique_citations.append(citation)
                seen.add(citation)
        
        return unique_citations
    
    def extract_evidence_spans(self, 
                             answer_text: str, 
                             evidence_documents: List[Dict[str, Any]],
                             query: str) -> List[Dict[str, Any]]:
        """
        Extract relevant spans from evidence documents that support the answer
        
        Args:
            answer_text: Generated answer text
            evidence_documents: List of evidence documents
            query: Original query for relevance scoring
        
        Returns:
            List of reference objects with doc_id, span, and relevance_score
        """
        
        references = []
        
        # Extract citations from answer
        cited_doc_ids = self.extract_citations_from_answer(answer_text)
        
        # Create document lookup
        doc_lookup = {}
        for doc in evidence_documents:
            doc_id = doc.get('id', '')
            if doc_id:
                doc_lookup[doc_id] = doc
        
        # Extract spans for each cited document
        for doc_id in cited_doc_ids:
            if doc_id in doc_lookup:
                doc = doc_lookup[doc_id]
                spans = self._extract_relevant_spans(answer_text, doc, query)
                
                for span_info in spans:
                    references.append({
                        "doc_id": doc_id,
                        "span": span_info["span"],
                        "relevance_score": span_info["relevance_score"],
                        "span_start": span_info.get("start_pos", 0),
                        "span_end": span_info.get("end_pos", len(span_info["span"]))
                    })
        
        # If no citations found, extract spans from highest scoring documents
        if not references and evidence_documents:
            # Use top 3 documents
            top_docs = sorted(evidence_documents, 
                            key=lambda x: x.get('similarity_score', 0), 
                            reverse=True)[:3]
            
            for doc in top_docs:
                doc_id = doc.get('id', f'doc_{len(references)}')
                spans = self._extract_relevant_spans(answer_text, doc, query)
                
                for span_info in spans[:1]:  # Take best span per document
                    references.append({
                        "doc_id": doc_id,
                        "span": span_info["span"],
                        "relevance_score": span_info["relevance_score"] * 0.8,  # Penalty for no citation
                        "auto_extracted": True
                    })
        
        # Sort by relevance score
        references.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update statistics
        self.citation_stats["total_extractions"] += 1
        if references:
            self.citation_stats["successful_extractions"] += 1
        
        return references[:10]  # Limit to top 10 references
    
    def _extract_relevant_spans(self, 
                              answer_text: str, 
                              document: Dict[str, Any], 
                              query: str) -> List[Dict[str, Any]]:
        """Extract relevant spans from a document"""
        
        doc_content = document.get('content', '')
        if not doc_content:
            return []
        
        # Split document into sentences
        sentences = self._split_into_sentences(doc_content)
        
        # Score each sentence for relevance
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            relevance_score = self._calculate_span_relevance(sentence, answer_text, query)
            
            if relevance_score > 0.3:  # Minimum relevance threshold
                sentence_scores.append({
                    "sentence": sentence,
                    "relevance_score": relevance_score,
                    "position": i
                })
        
        # Sort by relevance and take top spans
        sentence_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Extract spans (combine adjacent high-scoring sentences)
        spans = []
        used_positions = set()
        
        for sentence_info in sentence_scores[:5]:  # Top 5 sentences
            if sentence_info["position"] in used_positions:
                continue
            
            # Try to create span by combining adjacent sentences
            span_sentences = [sentence_info["sentence"]]
            span_positions = [sentence_info["position"]]
            
            # Check adjacent sentences
            for adj_offset in [-1, 1]:
                adj_pos = sentence_info["position"] + adj_offset
                if (0 <= adj_pos < len(sentences) and 
                    adj_pos not in used_positions):
                    
                    adj_sentence = sentences[adj_pos]
                    adj_relevance = self._calculate_span_relevance(adj_sentence, answer_text, query)
                    
                    if adj_relevance > 0.4:  # High relevance for adjacent
                        if adj_offset == -1:
                            span_sentences.insert(0, adj_sentence)
                            span_positions.insert(0, adj_pos)
                        else:
                            span_sentences.append(adj_sentence)
                            span_positions.append(adj_pos)
            
            # Create span
            span_text = " ".join(span_sentences)
            
            # Validate span length
            if self.min_span_length <= len(span_text) <= self.max_span_length:
                spans.append({
                    "span": span_text,
                    "relevance_score": sentence_info["relevance_score"],
                    "start_pos": min(span_positions),
                    "end_pos": max(span_positions)
                })
                
                used_positions.update(span_positions)
        
        return spans
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_span_relevance(self, span: str, answer_text: str, query: str) -> float:
        """Calculate relevance score for a text span"""
        
        span_lower = span.lower()
        answer_lower = answer_text.lower()
        query_lower = query.lower()
        
        # Word overlap with answer
        span_words = set(span_lower.split())
        answer_words = set(answer_lower.split())
        query_words = set(query_lower.split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        span_words -= stop_words
        answer_words -= stop_words
        query_words -= stop_words
        
        # Calculate overlaps
        answer_overlap = len(span_words.intersection(answer_words))
        query_overlap = len(span_words.intersection(query_words))
        
        # Calculate relevance score
        relevance_score = 0.0
        
        if span_words:
            # Answer overlap (primary factor)
            answer_relevance = answer_overlap / len(span_words)
            relevance_score += answer_relevance * 0.6
            
            # Query overlap (secondary factor)
            query_relevance = query_overlap / len(span_words)
            relevance_score += query_relevance * 0.4
        
        return min(relevance_score, 1.0)
    
    def validate_citations(self, 
                          answer_text: str, 
                          references: List[Dict[str, Any]], 
                          evidence_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that citations in answer match provided references"""
        
        validation = {
            "valid_citations": [],
            "invalid_citations": [],
            "missing_citations": [],
            "orphaned_references": [],
            "validation_score": 0.0
        }
        
        # Extract citations from answer
        cited_doc_ids = self.extract_citations_from_answer(answer_text)
        
        # Get reference doc IDs
        reference_doc_ids = [ref.get('doc_id', '') for ref in references]
        
        # Get available document IDs
        available_doc_ids = [doc.get('id', '') for doc in evidence_documents]
        
        # Validate each citation
        for doc_id in cited_doc_ids:
            if doc_id in reference_doc_ids:
                validation["valid_citations"].append(doc_id)
            elif doc_id in available_doc_ids:
                validation["missing_citations"].append(doc_id)
            else:
                validation["invalid_citations"].append(doc_id)
        
        # Check for orphaned references
        for ref in references:
            ref_doc_id = ref.get('doc_id', '')
            if ref_doc_id not in cited_doc_ids:
                validation["orphaned_references"].append(ref_doc_id)
        
        # Calculate validation score
        total_citations = len(cited_doc_ids)
        valid_citations = len(validation["valid_citations"])
        
        if total_citations > 0:
            validation["validation_score"] = valid_citations / total_citations
        else:
            validation["validation_score"] = 1.0 if not references else 0.0
        
        # Update statistics
        if validation["invalid_citations"]:
            self.citation_stats["invalid_citations"] += len(validation["invalid_citations"])
        
        return validation
    
    def format_references_for_output(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format references for final JSON output"""
        
        formatted_references = []
        
        for ref in references:
            # Ensure required fields
            formatted_ref = {
                "doc_id": ref.get('doc_id', 'unknown'),
                "span": ref.get('span', ''),
                "relevance_score": min(max(ref.get('relevance_score', 0.0), 0.0), 1.0)
            }
            
            # Add optional fields if available
            if 'span_start' in ref:
                formatted_ref['span_start'] = ref['span_start']
            if 'span_end' in ref:
                formatted_ref['span_end'] = ref['span_end']
            if 'auto_extracted' in ref:
                formatted_ref['auto_extracted'] = ref['auto_extracted']
            
            formatted_references.append(formatted_ref)
        
        return formatted_references
    
    def get_stats(self) -> Dict[str, Any]:
        """Get citation management statistics"""
        
        stats = self.citation_stats.copy()
        
        # Calculate derived metrics
        if stats["total_extractions"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_extractions"]
            stats["avg_citations_per_answer"] = stats["successful_extractions"] / stats["total_extractions"]
        
        return stats