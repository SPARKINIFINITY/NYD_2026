"""
Session Memory Manager

Manages conversation context and session state for intelligent planning.
Tracks query history, user preferences, and conversation flow.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json

class SessionMemoryManager:
    """Manages session memory and conversation context"""
    
    def __init__(self, max_history: int = 50, session_timeout: int = 3600):
        self.max_history = max_history
        self.session_timeout = session_timeout  # seconds
        
        # Session state
        self.session_id = None
        self.session_start = None
        self.last_activity = None
        
        # Conversation history
        self.query_history = deque(maxlen=max_history)
        self.intent_history = deque(maxlen=max_history)
        self.entity_history = deque(maxlen=max_history)
        self.plan_history = deque(maxlen=max_history)
        
        # Context tracking
        self.current_topic = None
        self.topic_entities = set()
        self.conversation_flow = []
        
        # User preferences (learned over time)
        self.user_preferences = {
            "preferred_retrievers": defaultdict(int),
            "preferred_k_values": defaultdict(list),
            "successful_strategies": defaultdict(int),
            "query_patterns": defaultdict(int)
        }
        
        # Session statistics
        self.session_stats = {
            "total_queries": 0,
            "intent_distribution": defaultdict(int),
            "avg_entities_per_query": 0,
            "multihop_queries": 0,
            "clarification_requests": 0
        }
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new session or resume existing one"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.session_start = datetime.now()
        self.last_activity = datetime.now()
        
        return session_id
    
    def is_session_active(self) -> bool:
        """Check if current session is still active"""
        if not self.last_activity:
            return False
        
        time_since_activity = (datetime.now() - self.last_activity).total_seconds()
        return time_since_activity < self.session_timeout
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def add_query_context(self, query: str, intent: str, entities: List[str], 
                         execution_plan: Dict[str, Any] = None):
        """Add new query context to session memory"""
        self.update_activity()
        
        timestamp = datetime.now()
        
        # Add to history
        self.query_history.append({
            "query": query,
            "timestamp": timestamp,
            "intent": intent,
            "entities": entities
        })
        
        self.intent_history.append(intent)
        self.entity_history.append(entities)
        
        if execution_plan:
            self.plan_history.append({
                "plan": execution_plan,
                "timestamp": timestamp,
                "query": query
            })
        
        # Update topic tracking
        self._update_topic_context(entities, intent)
        
        # Update conversation flow
        self.conversation_flow.append({
            "type": "query",
            "intent": intent,
            "timestamp": timestamp,
            "entities": entities
        })
        
        # Update statistics
        self._update_session_stats(intent, entities, execution_plan)
    
    def _update_topic_context(self, entities: List[str], intent: str):
        """Update current topic and related entities"""
        # Add entities to topic context
        for entity in entities:
            self.topic_entities.add(entity)
        
        # Determine current topic based on entities and intent
        if entities:
            # Simple topic detection based on entity overlap
            entity_set = set(entities)
            
            # Check if this continues the current topic
            if self.current_topic and len(entity_set.intersection(self.topic_entities)) > 0:
                # Continue current topic
                pass
            else:
                # New topic detected
                self.current_topic = self._infer_topic(entities, intent)
                # Keep only recent entities for topic context
                if len(self.topic_entities) > 20:
                    # Keep most recent entities
                    recent_entities = []
                    for query_context in list(self.query_history)[-5:]:
                        recent_entities.extend(query_context["entities"])
                    self.topic_entities = set(recent_entities)
    
    def _infer_topic(self, entities: List[str], intent: str) -> str:
        """Infer topic from entities and intent"""
        # Simple topic inference based on entity types and patterns
        entity_lower = [e.lower() for e in entities]
        
        # Religious/Mythological
        mythological_terms = {'rama', 'krishna', 'hanuman', 'sita', 'ravana', 'ramayana', 'mahabharata'}
        if any(term in entity_lower for term in mythological_terms):
            return "mythology_religion"
        
        # Technical/Programming
        tech_terms = {'python', 'javascript', 'api', 'database', 'algorithm', 'programming'}
        if any(term in entity_lower for term in tech_terms):
            return "technology_programming"
        
        # Science/Academic
        science_terms = {'dna', 'physics', 'chemistry', 'biology', 'theory', 'research'}
        if any(term in entity_lower for term in science_terms):
            return "science_academic"
        
        # History/Geography
        geo_terms = {'country', 'city', 'river', 'mountain', 'empire', 'war'}
        if any(term in entity_lower for term in geo_terms):
            return "history_geography"
        
        return "general"
    
    def _update_session_stats(self, intent: str, entities: List[str], execution_plan: Dict[str, Any]):
        """Update session statistics"""
        self.session_stats["total_queries"] += 1
        self.session_stats["intent_distribution"][intent] += 1
        
        # Update average entities per query
        total_entities = sum(len(qc["entities"]) for qc in self.query_history)
        self.session_stats["avg_entities_per_query"] = total_entities / len(self.query_history)
        
        # Track multihop queries
        if execution_plan and execution_plan.get("multihop", 0) > 0:
            self.session_stats["multihop_queries"] += 1
        
        # Track clarification requests
        if intent == "clarify":
            self.session_stats["clarification_requests"] += 1
    
    def get_recent_context(self, n: int = 5) -> Dict[str, Any]:
        """Get recent conversation context"""
        recent_queries = list(self.query_history)[-n:] if self.query_history else []
        recent_intents = list(self.intent_history)[-n:] if self.intent_history else []
        recent_entities = []
        
        for query_context in recent_queries:
            recent_entities.extend(query_context["entities"])
        
        return {
            "recent_queries": [qc["query"] for qc in recent_queries],
            "recent_intents": recent_intents,
            "recent_entities": list(set(recent_entities)),
            "current_topic": self.current_topic,
            "topic_entities": list(self.topic_entities),
            "conversation_length": len(self.query_history)
        }
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get learned user preferences"""
        # Calculate preferred retrievers
        preferred_retrievers = []
        if self.user_preferences["preferred_retrievers"]:
            sorted_retrievers = sorted(
                self.user_preferences["preferred_retrievers"].items(),
                key=lambda x: x[1], reverse=True
            )
            preferred_retrievers = [r[0] for r in sorted_retrievers[:3]]
        
        # Calculate average preferred k values
        avg_k_values = {}
        for retriever, k_values in self.user_preferences["preferred_k_values"].items():
            if k_values:
                avg_k_values[retriever] = sum(k_values) / len(k_values)
        
        return {
            "preferred_retrievers": preferred_retrievers,
            "avg_k_values": avg_k_values,
            "successful_strategies": dict(self.user_preferences["successful_strategies"]),
            "query_patterns": dict(self.user_preferences["query_patterns"])
        }
    
    def learn_from_feedback(self, execution_plan: Dict[str, Any], success: bool, 
                           feedback_score: float = None):
        """Learn from execution feedback to improve future planning"""
        if success:
            # Update successful strategies
            strategy = execution_plan.get("strategy", "default")
            self.user_preferences["successful_strategies"][strategy] += 1
            
            # Update preferred retrievers
            for retriever in execution_plan.get("retrievers", []):
                self.user_preferences["preferred_retrievers"][retriever] += 1
            
            # Update preferred k values
            for key, value in execution_plan.items():
                if key.endswith("_k") and isinstance(value, int):
                    retriever = key.replace("_k", "")
                    self.user_preferences["preferred_k_values"][retriever].append(value)
                    # Keep only recent values
                    if len(self.user_preferences["preferred_k_values"][retriever]) > 10:
                        self.user_preferences["preferred_k_values"][retriever].pop(0)
    
    def detect_conversation_patterns(self) -> Dict[str, Any]:
        """Detect patterns in conversation flow"""
        if len(self.conversation_flow) < 3:
            return {"patterns": [], "recommendations": []}
        
        patterns = []
        recommendations = []
        
        # Check for clarification patterns
        recent_intents = [item["intent"] for item in self.conversation_flow[-5:]]
        if recent_intents.count("clarify") >= 2:
            patterns.append("frequent_clarifications")
            recommendations.append("increase_context_in_responses")
        
        # Check for topic switching
        recent_topics = []
        for item in self.conversation_flow[-5:]:
            topic = self._infer_topic(item["entities"], item["intent"])
            recent_topics.append(topic)
        
        unique_topics = len(set(recent_topics))
        if unique_topics >= 3:
            patterns.append("topic_switching")
            recommendations.append("provide_topic_transitions")
        
        # Check for complex query patterns
        complex_intents = ["multi-hop", "compare", "explain"]
        complex_count = sum(1 for intent in recent_intents if intent in complex_intents)
        if complex_count >= 3:
            patterns.append("complex_reasoning_session")
            recommendations.append("enable_advanced_retrieval")
        
        return {
            "patterns": patterns,
            "recommendations": recommendations,
            "session_complexity": complex_count / len(recent_intents) if recent_intents else 0
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        duration = None
        if self.session_start:
            duration = (datetime.now() - self.session_start).total_seconds()
        
        context = self.get_recent_context()
        preferences = self.get_user_preferences()
        patterns = self.detect_conversation_patterns()
        
        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "is_active": self.is_session_active(),
            "statistics": self.session_stats,
            "current_context": context,
            "user_preferences": preferences,
            "conversation_patterns": patterns,
            "total_interactions": len(self.query_history)
        }
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export session data for persistence"""
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "query_history": [
                {
                    "query": qc["query"],
                    "timestamp": qc["timestamp"].isoformat(),
                    "intent": qc["intent"],
                    "entities": qc["entities"]
                }
                for qc in self.query_history
            ],
            "user_preferences": {
                "preferred_retrievers": dict(self.user_preferences["preferred_retrievers"]),
                "preferred_k_values": {k: list(v) for k, v in self.user_preferences["preferred_k_values"].items()},
                "successful_strategies": dict(self.user_preferences["successful_strategies"]),
                "query_patterns": dict(self.user_preferences["query_patterns"])
            },
            "session_stats": self.session_stats,
            "current_topic": self.current_topic,
            "topic_entities": list(self.topic_entities)
        }
    
    def load_session_data(self, session_data: Dict[str, Any]):
        """Load session data from persistence"""
        self.session_id = session_data.get("session_id")
        
        if session_data.get("session_start"):
            self.session_start = datetime.fromisoformat(session_data["session_start"])
        if session_data.get("last_activity"):
            self.last_activity = datetime.fromisoformat(session_data["last_activity"])
        
        # Load query history
        for qc in session_data.get("query_history", []):
            self.query_history.append({
                "query": qc["query"],
                "timestamp": datetime.fromisoformat(qc["timestamp"]),
                "intent": qc["intent"],
                "entities": qc["entities"]
            })
            self.intent_history.append(qc["intent"])
            self.entity_history.append(qc["entities"])
        
        # Load preferences
        prefs = session_data.get("user_preferences", {})
        self.user_preferences["preferred_retrievers"] = defaultdict(int, prefs.get("preferred_retrievers", {}))
        self.user_preferences["preferred_k_values"] = defaultdict(list, prefs.get("preferred_k_values", {}))
        self.user_preferences["successful_strategies"] = defaultdict(int, prefs.get("successful_strategies", {}))
        self.user_preferences["query_patterns"] = defaultdict(int, prefs.get("query_patterns", {}))
        
        # Load other data
        self.session_stats = session_data.get("session_stats", self.session_stats)
        self.current_topic = session_data.get("current_topic")
        self.topic_entities = set(session_data.get("topic_entities", []))