"""
Enhanced Response Formatter & Feedback Collector

Provides comprehensive output formatting with multiple output formats:
- Structured JSON for APIs
- User-friendly text for display
- Proper citations and references
- Immediate feedback collection (thumbs up/down, corrections)
- Trace storage for learning and improvement
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from .output_formatter import OutputFormatter

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    NOT_CORRECT = "not_correct"
    PARTIALLY_CORRECT = "partially_correct"
    VERY_HELPFUL = "very_helpful"
    NEEDS_IMPROVEMENT = "needs_improvement"

class OutputFormat(Enum):
    """Output format types"""
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"

@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    feedback_id: str
    response_id: str
    feedback_type: FeedbackType
    rating: Optional[int]  # 1-5 scale
    comment: Optional[str]
    correction: Optional[str]
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ResponseTrace:
    """Complete trace of response generation and feedback"""
    response_id: str
    query: str
    intent: str
    generated_response: Dict[str, Any]
    evidence_documents: List[Dict[str, Any]]
    generation_metadata: Dict[str, Any]
    feedback_entries: List[FeedbackEntry]
    timestamp: datetime
    processing_time: float
    validation_results: Optional[Dict[str, Any]] = None

class EnhancedResponseFormatter(OutputFormatter):
    """Enhanced response formatter with feedback collection"""
    
    def __init__(self,
                 strict_validation: bool = True,
                 max_summary_length: int = 300,
                 max_answer_length: int = 2000,
                 max_references: int = 10,
                 enable_feedback_collection: bool = True,
                 feedback_storage_path: str = "feedback_data",
                 trace_storage_path: str = "response_traces"):
        
        # Initialize base formatter
        super().__init__(strict_validation, max_summary_length, max_answer_length, max_references)
        
        # Feedback collection configuration
        self.enable_feedback_collection = enable_feedback_collection
        self.feedback_storage_path = feedback_storage_path
        self.trace_storage_path = trace_storage_path
        
        # In-memory storage for feedback and traces
        self.feedback_entries: Dict[str, List[FeedbackEntry]] = {}
        self.response_traces: Dict[str, ResponseTrace] = {}
        
        # Feedback statistics
        self.feedback_stats = {
            "total_responses": 0,
            "responses_with_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "corrections_received": 0,
            "avg_rating": 0.0,
            "feedback_rate": 0.0
        }
        
        # Output format templates
        self.format_templates = self._initialize_format_templates()
        
        logger.info(f"Enhanced Response Formatter initialized with feedback collection: {enable_feedback_collection}")
    
    def _initialize_format_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize output format templates"""
        
        return {
            "text": {
                "template": """
{summary}

{detailed_answer}

Sources:
{references}

Confidence: {confidence:.1%}
                """.strip(),
                "reference_format": "• {doc_id}: {span}"
            },
            "markdown": {
                "template": """
## Summary
{summary}

## Detailed Answer
{detailed_answer}

## Sources
{references}

**Confidence:** {confidence:.1%}
                """.strip(),
                "reference_format": "- **{doc_id}**: {span}"
            },
            "html": {
                "template": """
<div class="rag-response">
    <div class="summary">
        <h3>Summary</h3>
        <p>{summary}</p>
    </div>
    
    <div class="detailed-answer">
        <h3>Detailed Answer</h3>
        <p>{detailed_answer}</p>
    </div>
    
    <div class="references">
        <h3>Sources</h3>
        <ul>
            {references}
        </ul>
    </div>
    
    <div class="confidence">
        <strong>Confidence:</strong> {confidence:.1%}
    </div>
    
    <div class="feedback-section">
        {feedback_controls}
    </div>
</div>
                """.strip(),
                "reference_format": '<li><strong>{doc_id}:</strong> {span}</li>',
                "feedback_controls": """
<div class="feedback-controls">
    <button onclick="submitFeedback('{response_id}', 'thumbs_up')">👍</button>
    <button onclick="submitFeedback('{response_id}', 'thumbs_down')">👎</button>
    <button onclick="reportIncorrect('{response_id}')">Not Correct</button>
</div>
                """.strip()
            }
        }
    
    def format_enhanced_response(self,
                               raw_answer: Dict[str, Any],
                               evidence_documents: List[Dict[str, Any]] = None,
                               metadata: Dict[str, Any] = None,
                               query: str = "",
                               intent: str = "fact",
                               output_formats: List[OutputFormat] = None,
                               enable_feedback: bool = True) -> Dict[str, Any]:
        """
        Format response with multiple output formats and feedback collection
        
        Args:
            raw_answer: Raw answer dictionary
            evidence_documents: Original evidence documents
            metadata: Additional metadata
            query: Original user query
            intent: Query intent
            output_formats: List of desired output formats
            enable_feedback: Whether to enable feedback collection
            
        Returns:
            Enhanced formatted response with multiple formats
        """
        
        start_time = time.time()
        
        # Generate unique response ID
        response_id = str(uuid.uuid4())
        
        # Format using base formatter
        structured_output = self.format_output(raw_answer, evidence_documents, metadata)
        
        # Default output formats
        if output_formats is None:
            output_formats = [OutputFormat.JSON, OutputFormat.TEXT]
        
        # Create enhanced response
        enhanced_response = {
            "response_id": response_id,
            "query": query,
            "intent": intent,
            "formats": {},
            "feedback_enabled": enable_feedback and self.enable_feedback_collection,
            "generation_timestamp": datetime.now().isoformat(),
            "processing_time": time.time() - start_time
        }
        
        # Generate different output formats
        for format_type in output_formats:
            if format_type == OutputFormat.JSON:
                enhanced_response["formats"]["json"] = structured_output
            elif format_type == OutputFormat.TEXT:
                enhanced_response["formats"]["text"] = self._format_as_text(structured_output)
            elif format_type == OutputFormat.MARKDOWN:
                enhanced_response["formats"]["markdown"] = self._format_as_markdown(structured_output)
            elif format_type == OutputFormat.HTML:
                enhanced_response["formats"]["html"] = self._format_as_html(structured_output, response_id, enable_feedback)
        
        # Store response trace if feedback is enabled
        if enable_feedback and self.enable_feedback_collection:
            self._store_response_trace(
                response_id=response_id,
                query=query,
                intent=intent,
                generated_response=structured_output,
                evidence_documents=evidence_documents or [],
                generation_metadata=metadata or {},
                processing_time=time.time() - start_time,
                validation_results=structured_output.get("metadata", {}).get("validation")
            )
        
        # Update statistics
        self.feedback_stats["total_responses"] += 1
        
        return enhanced_response
    
    def _format_as_text(self, structured_output: Dict[str, Any]) -> str:
        """Format response as user-friendly text"""
        
        template = self.format_templates["text"]["template"]
        ref_format = self.format_templates["text"]["reference_format"]
        
        # Format references
        references = structured_output.get("references", [])
        formatted_refs = []
        for ref in references:
            formatted_ref = ref_format.format(
                doc_id=ref.get("doc_id", "Unknown"),
                span=ref.get("span", "No content available")[:100] + ("..." if len(ref.get("span", "")) > 100 else "")
            )
            formatted_refs.append(formatted_ref)
        
        references_text = "\n".join(formatted_refs) if formatted_refs else "No sources available"
        
        # Format complete text
        formatted_text = template.format(
            summary=structured_output.get("summary", "No summary available"),
            detailed_answer=structured_output.get("detailed_answer", "No detailed answer available"),
            references=references_text,
            confidence=structured_output.get("confidence", 0.0)
        )
        
        return formatted_text
    
    def _format_as_markdown(self, structured_output: Dict[str, Any]) -> str:
        """Format response as Markdown"""
        
        template = self.format_templates["markdown"]["template"]
        ref_format = self.format_templates["markdown"]["reference_format"]
        
        # Format references
        references = structured_output.get("references", [])
        formatted_refs = []
        for ref in references:
            formatted_ref = ref_format.format(
                doc_id=ref.get("doc_id", "Unknown"),
                span=ref.get("span", "No content available")[:150] + ("..." if len(ref.get("span", "")) > 150 else "")
            )
            formatted_refs.append(formatted_ref)
        
        references_text = "\n".join(formatted_refs) if formatted_refs else "- No sources available"
        
        # Format complete markdown
        formatted_markdown = template.format(
            summary=structured_output.get("summary", "No summary available"),
            detailed_answer=structured_output.get("detailed_answer", "No detailed answer available"),
            references=references_text,
            confidence=structured_output.get("confidence", 0.0)
        )
        
        return formatted_markdown
    
    def _format_as_html(self, structured_output: Dict[str, Any], response_id: str, enable_feedback: bool) -> str:
        """Format response as HTML with feedback controls"""
        
        template = self.format_templates["html"]["template"]
        ref_format = self.format_templates["html"]["reference_format"]
        
        # Format references
        references = structured_output.get("references", [])
        formatted_refs = []
        for ref in references:
            formatted_ref = ref_format.format(
                doc_id=ref.get("doc_id", "Unknown"),
                span=ref.get("span", "No content available")[:200] + ("..." if len(ref.get("span", "")) > 200 else "")
            )
            formatted_refs.append(formatted_ref)
        
        references_html = "\n".join(formatted_refs) if formatted_refs else "<li>No sources available</li>"
        
        # Add feedback controls if enabled
        feedback_controls = ""
        if enable_feedback:
            feedback_controls = self.format_templates["html"]["feedback_controls"].format(
                response_id=response_id
            )
        
        # Format complete HTML
        formatted_html = template.format(
            summary=structured_output.get("summary", "No summary available"),
            detailed_answer=structured_output.get("detailed_answer", "No detailed answer available"),
            references=references_html,
            confidence=structured_output.get("confidence", 0.0),
            feedback_controls=feedback_controls
        )
        
        return formatted_html    

    def collect_feedback(self,
                        response_id: str,
                        feedback_type: FeedbackType,
                        rating: Optional[int] = None,
                        comment: Optional[str] = None,
                        correction: Optional[str] = None,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect user feedback for a response
        
        Args:
            response_id: ID of the response being rated
            feedback_type: Type of feedback
            rating: Optional 1-5 rating
            comment: Optional text comment
            correction: Optional correction text
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata
            
        Returns:
            Feedback entry ID
        """
        
        if not self.enable_feedback_collection:
            logger.warning("Feedback collection is disabled")
            return ""
        
        # Create feedback entry
        feedback_id = str(uuid.uuid4())
        feedback_entry = FeedbackEntry(
            feedback_id=feedback_id,
            response_id=response_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            correction=correction,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        # Store feedback
        if response_id not in self.feedback_entries:
            self.feedback_entries[response_id] = []
        
        self.feedback_entries[response_id].append(feedback_entry)
        
        # Update statistics
        self._update_feedback_stats(feedback_entry)
        
        # Update response trace
        if response_id in self.response_traces:
            self.response_traces[response_id].feedback_entries.append(feedback_entry)
        
        logger.info(f"Collected feedback: {feedback_type.value} for response {response_id}")
        
        return feedback_id
    
    def get_feedback_summary(self, response_id: str) -> Dict[str, Any]:
        """Get feedback summary for a specific response"""
        
        if response_id not in self.feedback_entries:
            return {"error": "No feedback found for response"}
        
        feedback_list = self.feedback_entries[response_id]
        
        # Aggregate feedback
        feedback_counts = {}
        ratings = []
        corrections = []
        comments = []
        
        for feedback in feedback_list:
            feedback_type = feedback.feedback_type.value
            feedback_counts[feedback_type] = feedback_counts.get(feedback_type, 0) + 1
            
            if feedback.rating:
                ratings.append(feedback.rating)
            
            if feedback.correction:
                corrections.append(feedback.correction)
            
            if feedback.comment:
                comments.append(feedback.comment)
        
        summary = {
            "response_id": response_id,
            "total_feedback": len(feedback_list),
            "feedback_breakdown": feedback_counts,
            "average_rating": sum(ratings) / len(ratings) if ratings else None,
            "corrections_count": len(corrections),
            "comments_count": len(comments),
            "latest_feedback": feedback_list[-1].timestamp.isoformat() if feedback_list else None
        }
        
        return summary
    
    def get_learning_insights(self, min_feedback_count: int = 5) -> Dict[str, Any]:
        """
        Extract learning insights from collected feedback
        
        Args:
            min_feedback_count: Minimum feedback entries required for insights
            
        Returns:
            Learning insights and patterns
        """
        
        insights = {
            "total_responses_analyzed": 0,
            "responses_with_sufficient_feedback": 0,
            "common_issues": [],
            "improvement_suggestions": [],
            "positive_patterns": [],
            "correction_patterns": [],
            "rating_analysis": {}
        }
        
        responses_with_feedback = []
        all_corrections = []
        all_ratings = []
        issue_patterns = {}
        
        # Analyze each response with feedback
        for response_id, feedback_list in self.feedback_entries.items():
            insights["total_responses_analyzed"] += 1
            
            if len(feedback_list) >= min_feedback_count:
                insights["responses_with_sufficient_feedback"] += 1
                responses_with_feedback.append(response_id)
                
                # Analyze feedback patterns
                negative_feedback = [f for f in feedback_list if f.feedback_type in [FeedbackType.THUMBS_DOWN, FeedbackType.NOT_CORRECT]]
                positive_feedback = [f for f in feedback_list if f.feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.VERY_HELPFUL]]
                
                # Collect corrections and ratings
                for feedback in feedback_list:
                    if feedback.correction:
                        all_corrections.append(feedback.correction)
                    if feedback.rating:
                        all_ratings.append(feedback.rating)
                
                # Identify issue patterns
                if len(negative_feedback) > len(positive_feedback):
                    if response_id in self.response_traces:
                        trace = self.response_traces[response_id]
                        intent = trace.intent
                        confidence = trace.generated_response.get("confidence", 0.0)
                        
                        issue_key = f"{intent}_low_confidence" if confidence < 0.5 else f"{intent}_general"
                        issue_patterns[issue_key] = issue_patterns.get(issue_key, 0) + 1
        
        # Generate insights
        if all_ratings:
            insights["rating_analysis"] = {
                "average_rating": sum(all_ratings) / len(all_ratings),
                "rating_distribution": {i: all_ratings.count(i) for i in range(1, 6)},
                "low_ratings_count": len([r for r in all_ratings if r <= 2]),
                "high_ratings_count": len([r for r in all_ratings if r >= 4])
            }
        
        # Common issues
        insights["common_issues"] = [
            {"pattern": pattern, "count": count} 
            for pattern, count in sorted(issue_patterns.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Correction patterns
        if all_corrections:
            # Simple keyword analysis of corrections
            correction_keywords = {}
            for correction in all_corrections:
                words = correction.lower().split()
                for word in words:
                    if len(word) > 3:  # Filter short words
                        correction_keywords[word] = correction_keywords.get(word, 0) + 1
            
            insights["correction_patterns"] = [
                {"keyword": word, "frequency": count}
                for word, count in sorted(correction_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        
        # Generate improvement suggestions
        insights["improvement_suggestions"] = self._generate_improvement_suggestions(insights)
        
        return insights
    
    def _generate_improvement_suggestions(self, insights: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on feedback analysis"""
        
        suggestions = []
        
        # Rating-based suggestions
        rating_analysis = insights.get("rating_analysis", {})
        if rating_analysis:
            avg_rating = rating_analysis.get("average_rating", 0)
            if avg_rating < 3.0:
                suggestions.append("Overall response quality needs improvement - consider enhancing answer generation")
            
            low_ratings = rating_analysis.get("low_ratings_count", 0)
            total_ratings = sum(rating_analysis.get("rating_distribution", {}).values())
            if total_ratings > 0 and low_ratings / total_ratings > 0.3:
                suggestions.append("High percentage of low ratings - review answer validation and quality checks")
        
        # Issue pattern suggestions
        common_issues = insights.get("common_issues", [])
        for issue in common_issues[:3]:  # Top 3 issues
            pattern = issue["pattern"]
            if "low_confidence" in pattern:
                suggestions.append(f"Improve confidence estimation for {pattern.split('_')[0]} queries")
            elif "general" in pattern:
                suggestions.append(f"Review answer generation strategy for {pattern.split('_')[0]} queries")
        
        # Correction pattern suggestions
        correction_patterns = insights.get("correction_patterns", [])
        if correction_patterns:
            top_keywords = [cp["keyword"] for cp in correction_patterns[:3]]
            suggestions.append(f"Common correction topics: {', '.join(top_keywords)} - consider improving knowledge in these areas")
        
        return suggestions
    
    def _store_response_trace(self,
                            response_id: str,
                            query: str,
                            intent: str,
                            generated_response: Dict[str, Any],
                            evidence_documents: List[Dict[str, Any]],
                            generation_metadata: Dict[str, Any],
                            processing_time: float,
                            validation_results: Optional[Dict[str, Any]] = None):
        """Store complete response trace for learning"""
        
        trace = ResponseTrace(
            response_id=response_id,
            query=query,
            intent=intent,
            generated_response=generated_response,
            evidence_documents=evidence_documents,
            generation_metadata=generation_metadata,
            feedback_entries=[],
            timestamp=datetime.now(),
            processing_time=processing_time,
            validation_results=validation_results
        )
        
        self.response_traces[response_id] = trace
    
    def _update_feedback_stats(self, feedback_entry: FeedbackEntry):
        """Update feedback statistics"""
        
        # Count responses with feedback
        response_id = feedback_entry.response_id
        if response_id not in [f.response_id for f_list in self.feedback_entries.values() for f in f_list[:-1]]:
            self.feedback_stats["responses_with_feedback"] += 1
        
        # Count feedback types
        if feedback_entry.feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.VERY_HELPFUL]:
            self.feedback_stats["positive_feedback"] += 1
        elif feedback_entry.feedback_type in [FeedbackType.THUMBS_DOWN, FeedbackType.NOT_CORRECT, FeedbackType.NEEDS_IMPROVEMENT]:
            self.feedback_stats["negative_feedback"] += 1
        
        # Count corrections
        if feedback_entry.correction:
            self.feedback_stats["corrections_received"] += 1
        
        # Update feedback rate
        if self.feedback_stats["total_responses"] > 0:
            self.feedback_stats["feedback_rate"] = (
                self.feedback_stats["responses_with_feedback"] / self.feedback_stats["total_responses"]
            )
        
        # Update average rating
        all_ratings = []
        for feedback_list in self.feedback_entries.values():
            for feedback in feedback_list:
                if feedback.rating:
                    all_ratings.append(feedback.rating)
        
        if all_ratings:
            self.feedback_stats["avg_rating"] = sum(all_ratings) / len(all_ratings)
    
    def export_feedback_data(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export feedback data for analysis
        
        Args:
            format_type: Export format ("json", "csv", "dict")
            
        Returns:
            Exported feedback data
        """
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "statistics": self.feedback_stats,
            "responses": [],
            "feedback_entries": []
        }
        
        # Export response traces
        for trace in self.response_traces.values():
            export_data["responses"].append({
                "response_id": trace.response_id,
                "query": trace.query,
                "intent": trace.intent,
                "timestamp": trace.timestamp.isoformat(),
                "processing_time": trace.processing_time,
                "confidence": trace.generated_response.get("confidence", 0.0),
                "validation_status": trace.validation_results.get("status") if trace.validation_results else None,
                "feedback_count": len(trace.feedback_entries)
            })
        
        # Export feedback entries
        for response_id, feedback_list in self.feedback_entries.items():
            for feedback in feedback_list:
                export_data["feedback_entries"].append({
                    "feedback_id": feedback.feedback_id,
                    "response_id": feedback.response_id,
                    "feedback_type": feedback.feedback_type.value,
                    "rating": feedback.rating,
                    "comment": feedback.comment,
                    "correction": feedback.correction,
                    "timestamp": feedback.timestamp.isoformat(),
                    "user_id": feedback.user_id,
                    "session_id": feedback.session_id
                })
        
        if format_type == "json":
            return json.dumps(export_data, indent=2)
        elif format_type == "dict":
            return export_data
        else:
            return export_data
    
    def get_response_trace(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Get complete trace for a specific response"""
        
        if response_id not in self.response_traces:
            return None
        
        trace = self.response_traces[response_id]
        
        return {
            "response_id": trace.response_id,
            "query": trace.query,
            "intent": trace.intent,
            "generated_response": trace.generated_response,
            "evidence_count": len(trace.evidence_documents),
            "generation_metadata": trace.generation_metadata,
            "feedback_entries": [asdict(f) for f in trace.feedback_entries],
            "timestamp": trace.timestamp.isoformat(),
            "processing_time": trace.processing_time,
            "validation_results": trace.validation_results
        }
    
    def create_feedback_api_endpoints(self) -> Dict[str, Callable]:
        """
        Create API endpoint functions for feedback collection
        
        Returns:
            Dictionary of endpoint functions
        """
        
        def submit_feedback_endpoint(request_data: Dict[str, Any]) -> Dict[str, Any]:
            """API endpoint for submitting feedback"""
            try:
                feedback_id = self.collect_feedback(
                    response_id=request_data.get("response_id"),
                    feedback_type=FeedbackType(request_data.get("feedback_type")),
                    rating=request_data.get("rating"),
                    comment=request_data.get("comment"),
                    correction=request_data.get("correction"),
                    user_id=request_data.get("user_id"),
                    session_id=request_data.get("session_id"),
                    metadata=request_data.get("metadata")
                )
                
                return {
                    "success": True,
                    "feedback_id": feedback_id,
                    "message": "Feedback collected successfully"
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to collect feedback"
                }
        
        def get_feedback_endpoint(response_id: str) -> Dict[str, Any]:
            """API endpoint for getting feedback summary"""
            try:
                summary = self.get_feedback_summary(response_id)
                return {
                    "success": True,
                    "data": summary
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to get feedback summary"
                }
        
        def get_insights_endpoint() -> Dict[str, Any]:
            """API endpoint for getting learning insights"""
            try:
                insights = self.get_learning_insights()
                return {
                    "success": True,
                    "data": insights
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to get learning insights"
                }
        
        return {
            "submit_feedback": submit_feedback_endpoint,
            "get_feedback": get_feedback_endpoint,
            "get_insights": get_insights_endpoint
        }
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including feedback data"""
        
        base_stats = self.get_formatting_stats()
        
        enhanced_stats = {
            "formatting": base_stats,
            "feedback": self.feedback_stats,
            "traces": {
                "total_traces": len(self.response_traces),
                "traces_with_feedback": len([t for t in self.response_traces.values() if t.feedback_entries]),
                "avg_processing_time": sum(t.processing_time for t in self.response_traces.values()) / len(self.response_traces) if self.response_traces else 0
            }
        }
        
        return enhanced_stats
    
    def clear_all_data(self):
        """Clear all feedback and trace data"""
        
        self.feedback_entries.clear()
        self.response_traces.clear()
        
        self.feedback_stats = {
            "total_responses": 0,
            "responses_with_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "corrections_received": 0,
            "avg_rating": 0.0,
            "feedback_rate": 0.0
        }
        
        self.clear_stats()  # Clear base formatting stats
        
        logger.info("Cleared all feedback and trace data")