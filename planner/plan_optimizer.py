"""
Execution Plan Optimizer

Optimizes retrieval parameters based on query characteristics,
resource constraints, and performance requirements.
"""

from typing import Dict, List, Any, Optional, Tuple
import math
from collections import defaultdict

class ExecutionPlanOptimizer:
    """Optimizes execution plans for performance and resource efficiency"""
    
    def __init__(self, max_total_retrievals: int = 500, max_multihop_depth: int = 3):
        self.max_total_retrievals = max_total_retrievals
        self.max_multihop_depth = max_multihop_depth
        
        # Performance profiles for different retrievers
        self.retriever_profiles = {
            "dense": {"speed": 0.7, "accuracy": 0.9, "resource_cost": 0.8},
            "bm25": {"speed": 0.9, "accuracy": 0.7, "resource_cost": 0.3},
            "graph": {"speed": 0.5, "accuracy": 0.8, "resource_cost": 0.9},
            "hybrid": {"speed": 0.6, "accuracy": 0.95, "resource_cost": 0.7}
        }
        
        # Optimization history for learning
        self.optimization_history = []
        self.performance_feedback = defaultdict(list)
    
    def optimize_plan(self, plan: Dict[str, Any], constraints: Dict[str, Any] = None,
                     optimization_goal: str = "balanced") -> Dict[str, Any]:
        """
        Optimize execution plan based on constraints and goals
        
        Args:
            plan: Original execution plan
            constraints: Resource/performance constraints
            optimization_goal: "speed", "accuracy", "resource_efficient", "balanced"
        """
        if constraints is None:
            constraints = {}
        
        optimized_plan = plan.copy()
        
        # Apply different optimization strategies
        if optimization_goal == "speed":
            optimized_plan = self._optimize_for_speed(optimized_plan, constraints)
        elif optimization_goal == "accuracy":
            optimized_plan = self._optimize_for_accuracy(optimized_plan, constraints)
        elif optimization_goal == "resource_efficient":
            optimized_plan = self._optimize_for_resources(optimized_plan, constraints)
        else:  # balanced
            optimized_plan = self._optimize_balanced(optimized_plan, constraints)
        
        # Apply global constraints
        optimized_plan = self._apply_constraints(optimized_plan, constraints)
        
        # Add optimization metadata
        optimized_plan["optimization_applied"] = True
        optimized_plan["optimization_goal"] = optimization_goal
        optimized_plan["original_plan"] = plan
        
        # Store for learning
        self.optimization_history.append({
            "original": plan,
            "optimized": optimized_plan,
            "goal": optimization_goal,
            "constraints": constraints
        })
        
        return optimized_plan
    
    def _optimize_for_speed(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize plan for maximum speed"""
        optimized = plan.copy()
        
        # Prefer faster retrievers
        retrievers = optimized.get("retrievers", [])
        speed_sorted_retrievers = sorted(
            retrievers, 
            key=lambda r: self.retriever_profiles.get(r, {}).get("speed", 0.5),
            reverse=True
        )
        
        # Limit to top 2 fastest retrievers if more than 2
        if len(speed_sorted_retrievers) > 2:
            optimized["retrievers"] = speed_sorted_retrievers[:2]
        
        # Reduce k values for speed
        for retriever in optimized["retrievers"]:
            k_key = f"{retriever}_k"
            if k_key in optimized:
                optimized[k_key] = min(optimized[k_key], 75)  # Cap at 75 for speed
        
        # Reduce multihop for speed
        if optimized.get("multihop", 0) > 1:
            optimized["multihop"] = 1
        
        # Disable reranking if not critical
        if not constraints.get("require_rerank", False):
            optimized["rerank"] = False
        
        optimized["optimization_reasoning"] = "Optimized for speed: reduced k values, limited retrievers, minimal multihop"
        
        return optimized
    
    def _optimize_for_accuracy(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize plan for maximum accuracy"""
        optimized = plan.copy()
        
        # Prefer more accurate retrievers
        retrievers = optimized.get("retrievers", [])
        
        # Add hybrid retriever if not present and beneficial
        if "hybrid" not in retrievers and len(retrievers) >= 2:
            optimized["retrievers"] = ["hybrid"]
            optimized["hybrid_k"] = sum(optimized.get(f"{r}_k", 50) for r in retrievers[:2])
            # Remove individual retriever k values
            for r in retrievers:
                k_key = f"{r}_k"
                if k_key in optimized:
                    del optimized[k_key]
        else:
            # Increase k values for better coverage
            for retriever in retrievers:
                k_key = f"{retriever}_k"
                if k_key in optimized:
                    optimized[k_key] = min(optimized[k_key] * 1.5, 200)
        
        # Enable reranking for accuracy
        optimized["rerank"] = True
        
        # Allow higher multihop if beneficial
        if optimized.get("multihop", 0) > 0:
            optimized["multihop"] = min(optimized["multihop"] + 1, self.max_multihop_depth)
        
        optimized["optimization_reasoning"] = "Optimized for accuracy: increased k values, enabled reranking, enhanced multihop"
        
        return optimized
    
    def _optimize_for_resources(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize plan for resource efficiency"""
        optimized = plan.copy()
        
        # Prefer resource-efficient retrievers
        retrievers = optimized.get("retrievers", [])
        resource_sorted_retrievers = sorted(
            retrievers,
            key=lambda r: self.retriever_profiles.get(r, {}).get("resource_cost", 0.5)
        )
        
        # Remove most resource-intensive retrievers if multiple
        if len(resource_sorted_retrievers) > 2:
            optimized["retrievers"] = resource_sorted_retrievers[:2]
        
        # Reduce k values to save resources
        total_k = 0
        for retriever in optimized["retrievers"]:
            k_key = f"{retriever}_k"
            if k_key in optimized:
                total_k += optimized[k_key]
        
        # If total k exceeds threshold, scale down proportionally
        if total_k > 200:
            scale_factor = 200 / total_k
            for retriever in optimized["retrievers"]:
                k_key = f"{retriever}_k"
                if k_key in optimized:
                    optimized[k_key] = max(10, int(optimized[k_key] * scale_factor))
        
        # Limit multihop to save resources
        optimized["multihop"] = min(optimized.get("multihop", 0), 1)
        
        # Conditional reranking
        if total_k > 100:
            optimized["rerank"] = False
        
        optimized["optimization_reasoning"] = "Optimized for resources: reduced k values, limited retrievers, minimal processing"
        
        return optimized
    
    def _optimize_balanced(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize plan for balanced performance"""
        optimized = plan.copy()
        
        # Calculate balance scores for retrievers
        retrievers = optimized.get("retrievers", [])
        balanced_retrievers = []
        
        for retriever in retrievers:
            profile = self.retriever_profiles.get(retriever, {})
            balance_score = (
                profile.get("speed", 0.5) * 0.3 +
                profile.get("accuracy", 0.5) * 0.5 +
                (1 - profile.get("resource_cost", 0.5)) * 0.2
            )
            balanced_retrievers.append((retriever, balance_score))
        
        # Sort by balance score and keep top retrievers
        balanced_retrievers.sort(key=lambda x: x[1], reverse=True)
        optimized["retrievers"] = [r[0] for r in balanced_retrievers[:3]]
        
        # Balance k values
        total_budget = 300  # Total k budget
        num_retrievers = len(optimized["retrievers"])
        
        if num_retrievers > 0:
            base_k = total_budget // num_retrievers
            for retriever in optimized["retrievers"]:
                k_key = f"{retriever}_k"
                current_k = optimized.get(k_key, 50)
                # Balance between current plan and budget allocation
                optimized[k_key] = int((current_k + base_k) / 2)
        
        # Moderate multihop
        if optimized.get("multihop", 0) > 2:
            optimized["multihop"] = 2
        
        # Enable reranking for quality
        optimized["rerank"] = True
        
        optimized["optimization_reasoning"] = "Balanced optimization: moderate k values, quality retrievers, reasonable multihop"
        
        return optimized
    
    def _apply_constraints(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hard constraints to the plan"""
        constrained = plan.copy()
        
        # Apply maximum total retrievals constraint
        total_retrievals = sum(
            constrained.get(f"{r}_k", 0) 
            for r in constrained.get("retrievers", [])
        )
        
        if total_retrievals > self.max_total_retrievals:
            scale_factor = self.max_total_retrievals / total_retrievals
            for retriever in constrained.get("retrievers", []):
                k_key = f"{retriever}_k"
                if k_key in constrained:
                    constrained[k_key] = max(1, int(constrained[k_key] * scale_factor))
        
        # Apply multihop constraint
        if constrained.get("multihop", 0) > self.max_multihop_depth:
            constrained["multihop"] = self.max_multihop_depth
        
        # Apply user-specified constraints
        max_k = constraints.get("max_k_per_retriever")
        if max_k:
            for retriever in constrained.get("retrievers", []):
                k_key = f"{retriever}_k"
                if k_key in constrained:
                    constrained[k_key] = min(constrained[k_key], max_k)
        
        # Force specific retrievers if required
        required_retrievers = constraints.get("required_retrievers", [])
        if required_retrievers:
            current_retrievers = set(constrained.get("retrievers", []))
            for req_retriever in required_retrievers:
                if req_retriever not in current_retrievers:
                    constrained["retrievers"].append(req_retriever)
                    constrained[f"{req_retriever}_k"] = constraints.get("default_k", 50)
        
        # Apply timeout constraints
        max_timeout = constraints.get("max_timeout_seconds")
        if max_timeout:
            # Estimate execution time and adjust if needed
            estimated_time = self._estimate_execution_time(constrained)
            if estimated_time > max_timeout:
                # Reduce complexity to meet timeout
                constrained = self._reduce_complexity_for_timeout(constrained, max_timeout)
        
        return constrained
    
    def _estimate_execution_time(self, plan: Dict[str, Any]) -> float:
        """Estimate execution time for a plan (in seconds)"""
        base_time = 0.0
        
        # Time for each retriever
        for retriever in plan.get("retrievers", []):
            k_value = plan.get(f"{retriever}_k", 50)
            retriever_speed = self.retriever_profiles.get(retriever, {}).get("speed", 0.5)
            
            # Base time inversely proportional to speed, proportional to k
            retriever_time = (k_value / 100) * (2.0 / retriever_speed)
            base_time += retriever_time
        
        # Time for reranking
        if plan.get("rerank", False):
            total_k = sum(plan.get(f"{r}_k", 0) for r in plan.get("retrievers", []))
            rerank_time = total_k / 200  # Assume 200 docs per second reranking
            base_time += rerank_time
        
        # Time for multihop
        multihop_steps = plan.get("multihop", 0)
        if multihop_steps > 0:
            # Each multihop step adds complexity
            base_time *= (1 + multihop_steps * 0.5)
        
        return base_time
    
    def _reduce_complexity_for_timeout(self, plan: Dict[str, Any], max_timeout: float) -> Dict[str, Any]:
        """Reduce plan complexity to meet timeout constraint"""
        reduced = plan.copy()
        
        while self._estimate_execution_time(reduced) > max_timeout:
            # Try different reduction strategies
            
            # 1. Reduce k values
            for retriever in reduced.get("retrievers", []):
                k_key = f"{retriever}_k"
                if k_key in reduced and reduced[k_key] > 10:
                    reduced[k_key] = max(10, int(reduced[k_key] * 0.8))
            
            # 2. Remove slowest retriever if multiple
            if len(reduced.get("retrievers", [])) > 1:
                retrievers = reduced["retrievers"]
                slowest = min(retrievers, key=lambda r: self.retriever_profiles.get(r, {}).get("speed", 0.5))
                reduced["retrievers"].remove(slowest)
                k_key = f"{slowest}_k"
                if k_key in reduced:
                    del reduced[k_key]
            
            # 3. Reduce multihop
            if reduced.get("multihop", 0) > 0:
                reduced["multihop"] -= 1
            
            # 4. Disable reranking as last resort
            if reduced.get("rerank", False):
                reduced["rerank"] = False
            
            # Prevent infinite loop
            if not reduced.get("retrievers") or all(reduced.get(f"{r}_k", 0) <= 10 for r in reduced.get("retrievers", [])):
                break
        
        reduced["timeout_optimization_applied"] = True
        return reduced
    
    def learn_from_feedback(self, plan: Dict[str, Any], performance_metrics: Dict[str, Any]):
        """Learn from execution feedback to improve future optimizations"""
        strategy = plan.get("strategy", "unknown")
        optimization_goal = plan.get("optimization_goal", "balanced")
        
        # Store performance feedback
        feedback_key = f"{strategy}_{optimization_goal}"
        self.performance_feedback[feedback_key].append({
            "plan": plan,
            "metrics": performance_metrics,
            "timestamp": performance_metrics.get("timestamp")
        })
        
        # Keep only recent feedback (last 100 entries per key)
        if len(self.performance_feedback[feedback_key]) > 100:
            self.performance_feedback[feedback_key] = self.performance_feedback[feedback_key][-100:]
    
    def get_optimization_recommendations(self, query_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization recommendations based on query characteristics and learned patterns"""
        
        recommendations = {
            "suggested_goal": "balanced",
            "suggested_constraints": {},
            "reasoning": []
        }
        
        # Analyze query characteristics
        query_length = query_characteristics.get("query_length", 0)
        entity_count = query_characteristics.get("entity_count", 0)
        intent = query_characteristics.get("intent", "unknown")
        
        # Recommend based on query complexity
        complexity_score = (query_length / 10) + (entity_count / 3)
        
        if complexity_score > 3:
            recommendations["suggested_goal"] = "accuracy"
            recommendations["reasoning"].append("High complexity query benefits from accuracy optimization")
        elif complexity_score < 1:
            recommendations["suggested_goal"] = "speed"
            recommendations["reasoning"].append("Simple query can be optimized for speed")
        
        # Recommend based on intent
        if intent in ["multi-hop", "compare"]:
            recommendations["suggested_goal"] = "accuracy"
            recommendations["suggested_constraints"]["min_multihop"] = 1
            recommendations["reasoning"].append(f"Intent '{intent}' benefits from comprehensive retrieval")
        elif intent in ["fact", "clarify"]:
            recommendations["suggested_goal"] = "speed"
            recommendations["reasoning"].append(f"Intent '{intent}' can be handled with fast retrieval")
        
        # Add learned recommendations from feedback
        if self.performance_feedback:
            # Find similar successful optimizations
            similar_feedback = self._find_similar_feedback(query_characteristics)
            if similar_feedback:
                avg_performance = self._calculate_average_performance(similar_feedback)
                if avg_performance.get("success_rate", 0) > 0.8:
                    best_goal = max(
                        set(fb["plan"].get("optimization_goal", "balanced") for fb in similar_feedback),
                        key=lambda goal: sum(1 for fb in similar_feedback 
                                           if fb["plan"].get("optimization_goal") == goal and 
                                           fb["metrics"].get("success", False))
                    )
                    recommendations["suggested_goal"] = best_goal
                    recommendations["reasoning"].append(f"Historical data suggests '{best_goal}' works well for similar queries")
        
        return recommendations
    
    def _find_similar_feedback(self, query_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar feedback entries based on query characteristics"""
        similar = []
        target_intent = query_characteristics.get("intent", "unknown")
        target_complexity = (query_characteristics.get("query_length", 0) / 10 + 
                           query_characteristics.get("entity_count", 0) / 3)
        
        for feedback_list in self.performance_feedback.values():
            for feedback in feedback_list:
                plan = feedback["plan"]
                plan_intent = plan.get("base_strategy", plan.get("strategy", "unknown"))
                plan_complexity = (plan.get("query_length", 0) / 10 + 
                                 plan.get("entity_count", 0) / 3)
                
                # Check similarity
                intent_match = plan_intent == target_intent
                complexity_similar = abs(plan_complexity - target_complexity) < 1.0
                
                if intent_match and complexity_similar:
                    similar.append(feedback)
        
        return similar
    
    def _calculate_average_performance(self, feedback_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate average performance metrics from feedback list"""
        if not feedback_list:
            return {}
        
        metrics_sum = defaultdict(float)
        metrics_count = defaultdict(int)
        
        for feedback in feedback_list:
            metrics = feedback["metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_sum[key] += value
                    metrics_count[key] += 1
                elif isinstance(value, bool):
                    metrics_sum[key] += 1 if value else 0
                    metrics_count[key] += 1
        
        avg_metrics = {}
        for key in metrics_sum:
            if metrics_count[key] > 0:
                avg_metrics[key] = metrics_sum[key] / metrics_count[key]
        
        return avg_metrics
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization history and performance"""
        stats = {
            "total_optimizations": len(self.optimization_history),
            "optimization_goals": defaultdict(int),
            "average_improvements": defaultdict(list),
            "feedback_entries": sum(len(fb_list) for fb_list in self.performance_feedback.values())
        }
        
        # Analyze optimization history
        for opt in self.optimization_history:
            goal = opt["goal"]
            stats["optimization_goals"][goal] += 1
        
        # Analyze performance feedback
        for feedback_list in self.performance_feedback.values():
            for feedback in feedback_list:
                metrics = feedback["metrics"]
                if "execution_time" in metrics:
                    stats["average_improvements"]["execution_time"].append(metrics["execution_time"])
                if "success" in metrics:
                    stats["average_improvements"]["success_rate"].append(1 if metrics["success"] else 0)
        
        # Calculate averages
        for key, values in stats["average_improvements"].items():
            if values:
                stats["average_improvements"][key] = sum(values) / len(values)
        
        return dict(stats)