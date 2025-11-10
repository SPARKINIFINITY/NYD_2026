"""
Graph Retriever - Knowledge Graph Traversal

Implements graph-based retrieval using knowledge graph structure
for entity-relation based search and neighbor expansion.
"""

import json
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import time
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class GraphRetriever:
    """Graph-based retrieval using knowledge graph traversal"""
    
    def __init__(self, 
                 graph_path: str = None,
                 max_hops: int = 3,
                 min_edge_weight: float = 0.1):
        
        self.max_hops = max_hops
        self.min_edge_weight = min_edge_weight
        
        # Initialize graph
        self.graph = nx.MultiDiGraph()
        self.entity_to_docs = defaultdict(set)  # Entity -> document IDs
        self.doc_to_entities = defaultdict(set)  # Document ID -> entities
        self.document_metadata = {}
        
        # Relation types and weights
        self.relation_weights = {
            'IS_A': 1.0,
            'PART_OF': 0.9,
            'LOCATED_IN': 0.8,
            'RELATED_TO': 0.7,
            'FATHER_OF': 0.9,
            'MOTHER_OF': 0.9,
            'SON_OF': 0.9,
            'DAUGHTER_OF': 0.9,
            'RULED': 0.8,
            'BORN_IN': 0.8,
            'DEFEATED': 0.7,
            'MARRIED_TO': 0.8,
            'WORKS_FOR': 0.7,
            'FOUNDED_BY': 0.8
        }
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "cache_hits": 0,
            "avg_hops": 0.0
        }
        
        # Query cache
        self.query_cache = {}
        self.max_cache_size = 500
        
        # Load graph if path provided
        if graph_path:
            self.load_graph(graph_path)
        
        logger.info(f"Initialized graph retriever with max_hops: {max_hops}")
    
    def build_graph(self, knowledge_graph: Dict[str, Any], documents: List[Dict[str, Any]] = None) -> bool:
        """
        Build graph from knowledge graph data
        
        Args:
            knowledge_graph: Graph data with nodes and edges
            documents: Optional document metadata
        """
        try:
            logger.info("Building knowledge graph...")
            start_time = time.time()
            
            # Add nodes
            nodes = knowledge_graph.get('nodes', [])
            for node in nodes:
                node_id = node.get('id')
                if node_id:
                    self.graph.add_node(
                        node_id,
                        label=node.get('label', node_id),
                        type=node.get('type', 'UNKNOWN'),
                        properties=node.get('properties', {})
                    )
            
            # Add edges
            edges = knowledge_graph.get('edges', [])
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                relation_type = edge.get('relation_type', 'RELATED_TO')
                
                if source and target:
                    weight = self.relation_weights.get(relation_type, 0.5)
                    confidence = edge.get('properties', {}).get('confidence', 0.5)
                    final_weight = weight * confidence
                    
                    if final_weight >= self.min_edge_weight:
                        self.graph.add_edge(
                            source, target,
                            relation=relation_type,
                            weight=final_weight,
                            properties=edge.get('properties', {})
                        )
            
            # Load metadata from knowledge graph if available
            if 'metadata' in knowledge_graph:
                metadata = knowledge_graph['metadata']
                
                # Load entity_to_docs
                if 'entity_to_docs' in metadata:
                    self.entity_to_docs = defaultdict(set)
                    for entity, docs in metadata['entity_to_docs'].items():
                        self.entity_to_docs[entity] = set(docs)
                
                # Load doc_to_entities
                if 'doc_to_entities' in metadata:
                    self.doc_to_entities = defaultdict(set)
                    for doc, entities in metadata['doc_to_entities'].items():
                        self.doc_to_entities[doc] = set(entities)
                
                # Load document_metadata
                if 'document_metadata' in metadata:
                    self.document_metadata = metadata['document_metadata']
                
                logger.info(f"Loaded metadata: {len(self.entity_to_docs)} entities, {len(self.doc_to_entities)} documents")
            
            # Build entity-document mappings if documents provided
            if documents:
                self._build_entity_document_mappings(documents)
            
            build_time = time.time() - start_time
            logger.info(f"Graph built in {build_time:.2f}s: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            return False
    
    def _build_entity_document_mappings(self, documents: List[Dict[str, Any]]):
        """Build mappings between entities and documents"""
        for doc in documents:
            doc_id = doc.get('id', str(len(self.document_metadata)))
            self.document_metadata[doc_id] = doc
            
            # Extract entities from document
            entities = doc.get('entities', [])
            if isinstance(entities, str):
                entities = [entities]
            
            for entity in entities:
                entity_clean = entity.strip().lower()
                # Find matching graph nodes
                for node_id in self.graph.nodes():
                    node_label = self.graph.nodes[node_id].get('label', '').lower()
                    if entity_clean == node_label or entity_clean in node_label:
                        self.entity_to_docs[node_id].add(doc_id)
                        self.doc_to_entities[doc_id].add(node_id)
    
    def save_graph(self, file_path: str):
        """Save graph to file"""
        try:
            graph_data = {
                'nodes': [
                    {
                        'id': node_id,
                        'label': data.get('label', node_id),
                        'type': data.get('type', 'UNKNOWN'),
                        'properties': data.get('properties', {})
                    }
                    for node_id, data in self.graph.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        'relation_type': data.get('relation', 'RELATED_TO'),
                        'weight': data.get('weight', 0.5),
                        'properties': data.get('properties', {})
                    }
                    for source, target, data in self.graph.edges(data=True)
                ],
                'metadata': {
                    'entity_to_docs': {k: list(v) for k, v in self.entity_to_docs.items()},
                    'doc_to_entities': {k: list(v) for k, v in self.doc_to_entities.items()},
                    'document_metadata': self.document_metadata
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Graph saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
    
    def load_graph(self, file_path: str) -> bool:
        """Load graph from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Load nodes
            for node in graph_data.get('nodes', []):
                self.graph.add_node(
                    node['id'],
                    label=node.get('label', node['id']),
                    type=node.get('type', 'UNKNOWN'),
                    properties=node.get('properties', {})
                )
            
            # Load edges
            for edge in graph_data.get('edges', []):
                self.graph.add_edge(
                    edge['source'],
                    edge['target'],
                    relation=edge.get('relation_type', 'RELATED_TO'),
                    weight=edge.get('weight', 0.5),
                    properties=edge.get('properties', {})
                )
            
            # Load metadata
            metadata = graph_data.get('metadata', {})
            self.entity_to_docs = defaultdict(set)
            for entity, docs in metadata.get('entity_to_docs', {}).items():
                self.entity_to_docs[entity] = set(docs)
            
            self.doc_to_entities = defaultdict(set)
            for doc, entities in metadata.get('doc_to_entities', {}).items():
                self.doc_to_entities[doc] = set(entities)
            
            self.document_metadata = metadata.get('document_metadata', {})
            
            logger.info(f"Graph loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return False
    
    def retrieve(self, query_entities: List[str], k: int = 10, 
                max_hops: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using graph traversal
        
        Args:
            query_entities: List of entities to start traversal from
            k: Number of results to return
            max_hops: Maximum hops for traversal (overrides default)
            threshold: Optional similarity threshold
        
        Returns:
            List of retrieved documents with scores
        """
        if not self.graph.nodes():
            logger.error("Graph retriever not properly initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{sorted(query_entities)}_{k}_{max_hops}_{threshold}"
            if cache_key in self.query_cache:
                self.retrieval_stats["cache_hits"] += 1
                return self.query_cache[cache_key]
            
            max_hops = max_hops or self.max_hops
            
            # Find matching entities in graph
            matched_entities = self._find_matching_entities(query_entities)
            
            if not matched_entities:
                logger.debug(f"No exact matching entities found for: {query_entities}")
                # Fallback: try fuzzy matching with all nodes
                matched_entities = self._fuzzy_match_entities(query_entities, top_k=5)
                
            if not matched_entities:
                logger.debug(f"No matching entities found even with fuzzy matching")
                return []
            
            # Perform graph traversal
            relevant_entities, traversal_info = self._graph_traversal(matched_entities, max_hops)
            
            # Score and rank documents
            document_scores = self._score_documents(relevant_entities, matched_entities, traversal_info)
            
            # Get top-k results
            sorted_docs = sorted(document_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:k]
            
            # Process results
            results = []
            for rank, (doc_id, score_info) in enumerate(sorted_docs):
                score = score_info['score']
                
                # Apply threshold if specified
                if threshold and score < threshold:
                    continue
                
                # Get document metadata
                doc_metadata = self.document_metadata.get(doc_id, {'id': doc_id})
                doc_metadata = doc_metadata.copy()
                doc_metadata.update({
                    'retriever_type': 'graph',
                    'similarity_score': float(score),
                    'rank': rank + 1,
                    'retrieval_method': 'graph_traversal',
                    'matched_entities': score_info['matched_entities'],
                    'traversal_hops': score_info['avg_hops'],
                    'relation_paths': score_info['relation_paths'][:3]  # Top 3 paths
                })
                results.append(doc_metadata)
            
            # Update cache
            if len(self.query_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            
            self.query_cache[cache_key] = results
            
            # Update stats
            retrieval_time = time.time() - start_time
            self.retrieval_stats["total_queries"] += 1
            self.retrieval_stats["total_time"] += retrieval_time
            self.retrieval_stats["average_time"] = (
                self.retrieval_stats["total_time"] / self.retrieval_stats["total_queries"]
            )
            
            if traversal_info:
                avg_hops = sum(traversal_info.values()) / len(traversal_info)
                self.retrieval_stats["avg_hops"] = (
                    (self.retrieval_stats["avg_hops"] * (self.retrieval_stats["total_queries"] - 1) + avg_hops) /
                    self.retrieval_stats["total_queries"]
                )
            
            logger.debug(f"Graph retrieval completed in {retrieval_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []
    
    def _find_matching_entities(self, query_entities: List[str]) -> List[str]:
        """Find entities in graph that match query entities"""
        matched = []
        
        for query_entity in query_entities:
            query_lower = query_entity.lower().strip()
            
            # Exact match first
            for node_id in self.graph.nodes():
                node_label = self.graph.nodes[node_id].get('label', '').lower()
                if query_lower == node_label:
                    matched.append(node_id)
                    break
            else:
                # Partial match
                for node_id in self.graph.nodes():
                    node_label = self.graph.nodes[node_id].get('label', '').lower()
                    if query_lower in node_label or node_label in query_lower:
                        matched.append(node_id)
                        break
        
        return list(set(matched))  # Remove duplicates
    
    def _fuzzy_match_entities(self, query_entities: List[str], top_k: int = 5) -> List[str]:
        """Fuzzy match entities when exact match fails"""
        matched = []
        scores = []
        
        for query_entity in query_entities:
            query_lower = query_entity.lower().strip()
            query_words = set(query_lower.split())
            
            for node_id in self.graph.nodes():
                node_label = self.graph.nodes[node_id].get('label', '').lower()
                node_words = set(node_label.split())
                
                # Calculate word overlap score
                if query_words and node_words:
                    overlap = len(query_words & node_words)
                    if overlap > 0:
                        score = overlap / max(len(query_words), len(node_words))
                        matched.append(node_id)
                        scores.append(score)
        
        # Return top-k matches by score
        if matched:
            sorted_matches = sorted(zip(matched, scores), key=lambda x: x[1], reverse=True)
            return [m[0] for m in sorted_matches[:top_k]]
        
        return []
    
    def _graph_traversal(self, start_entities: List[str], max_hops: int) -> Tuple[Set[str], Dict[str, int]]:
        """Perform graph traversal from start entities"""
        visited = set()
        relevant_entities = set(start_entities)
        traversal_info = {}  # entity -> hops from start
        
        # BFS traversal
        queue = deque([(entity, 0) for entity in start_entities])
        
        for start_entity in start_entities:
            traversal_info[start_entity] = 0
        
        while queue:
            current_entity, hops = queue.popleft()
            
            if current_entity in visited or hops >= max_hops:
                continue
            
            visited.add(current_entity)
            
            # Get neighbors
            neighbors = list(self.graph.neighbors(current_entity))
            predecessors = list(self.graph.predecessors(current_entity))
            all_connected = set(neighbors + predecessors)
            
            for neighbor in all_connected:
                if neighbor not in visited and hops + 1 <= max_hops:
                    relevant_entities.add(neighbor)
                    traversal_info[neighbor] = min(traversal_info.get(neighbor, float('inf')), hops + 1)
                    queue.append((neighbor, hops + 1))
        
        return relevant_entities, traversal_info
    
    def _score_documents(self, relevant_entities: Set[str], query_entities: List[str], 
                        traversal_info: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
        """Score documents based on relevant entities and graph structure"""
        document_scores = defaultdict(lambda: {
            'score': 0.0,
            'matched_entities': [],
            'avg_hops': 0.0,
            'relation_paths': []
        })
        
        for entity in relevant_entities:
            # Get documents containing this entity
            docs = self.entity_to_docs.get(entity, set())
            
            # Calculate entity score based on hops from query
            hops = traversal_info.get(entity, self.max_hops)
            entity_score = 1.0 / (1.0 + hops)  # Closer entities get higher scores
            
            # Bonus for exact query matches
            if entity in query_entities:
                entity_score *= 2.0
            
            for doc_id in docs:
                document_scores[doc_id]['score'] += entity_score
                document_scores[doc_id]['matched_entities'].append(entity)
                
                # Update average hops
                current_hops = document_scores[doc_id]['avg_hops']
                entity_count = len(document_scores[doc_id]['matched_entities'])
                document_scores[doc_id]['avg_hops'] = (
                    (current_hops * (entity_count - 1) + hops) / entity_count
                )
        
        # Add relation path information
        for doc_id in document_scores:
            doc_entities = document_scores[doc_id]['matched_entities']
            paths = self._find_relation_paths(query_entities, doc_entities)
            document_scores[doc_id]['relation_paths'] = paths
        
        return dict(document_scores)
    
    def _find_relation_paths(self, query_entities: List[str], doc_entities: List[str]) -> List[Dict[str, Any]]:
        """Find relation paths between query entities and document entities"""
        paths = []
        
        for query_entity in query_entities:
            for doc_entity in doc_entities:
                if query_entity == doc_entity:
                    continue
                
                try:
                    # Find shortest path
                    if nx.has_path(self.graph, query_entity, doc_entity):
                        path = nx.shortest_path(self.graph, query_entity, doc_entity)
                        
                        # Get relations along path
                        relations = []
                        for i in range(len(path) - 1):
                            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                            if edge_data:
                                # Get the first edge if multiple edges exist
                                first_edge = list(edge_data.values())[0]
                                relations.append(first_edge.get('relation', 'RELATED_TO'))
                        
                        paths.append({
                            'from': query_entity,
                            'to': doc_entity,
                            'path': path,
                            'relations': relations,
                            'length': len(path) - 1
                        })
                        
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by path length (shorter paths first)
        paths.sort(key=lambda x: x['length'])
        return paths
    
    def expand_entities(self, entities: List[str], hops: int = 1) -> Dict[str, List[str]]:
        """Expand entities by finding their neighbors"""
        expansion = {}
        
        for entity in entities:
            if entity not in self.graph:
                expansion[entity] = []
                continue
            
            neighbors = set()
            queue = deque([(entity, 0)])
            visited = set()
            
            while queue:
                current, current_hops = queue.popleft()
                
                if current in visited or current_hops >= hops:
                    continue
                
                visited.add(current)
                
                # Get direct neighbors
                direct_neighbors = list(self.graph.neighbors(current)) + list(self.graph.predecessors(current))
                
                for neighbor in direct_neighbors:
                    if neighbor != entity:  # Don't include the original entity
                        neighbors.add(neighbor)
                        if current_hops + 1 < hops:
                            queue.append((neighbor, current_hops + 1))
            
            expansion[entity] = list(neighbors)
        
        return expansion
    
    def get_entity_relations(self, entity: str) -> List[Dict[str, Any]]:
        """Get all relations for an entity"""
        relations = []
        
        if entity not in self.graph:
            return relations
        
        # Outgoing relations
        for target in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, target)
            if edge_data:
                for edge in edge_data.values():
                    relations.append({
                        'type': 'outgoing',
                        'source': entity,
                        'target': target,
                        'relation': edge.get('relation', 'RELATED_TO'),
                        'weight': edge.get('weight', 0.5)
                    })
        
        # Incoming relations
        for source in self.graph.predecessors(entity):
            edge_data = self.graph.get_edge_data(source, entity)
            if edge_data:
                for edge in edge_data.values():
                    relations.append({
                        'type': 'incoming',
                        'source': source,
                        'target': entity,
                        'relation': edge.get('relation', 'RELATED_TO'),
                        'weight': edge.get('weight', 0.5)
                    })
        
        return relations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        stats = self.retrieval_stats.copy()
        stats.update({
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'max_hops': self.max_hops,
            'min_edge_weight': self.min_edge_weight,
            'entities_with_docs': len(self.entity_to_docs),
            'total_documents': len(self.document_metadata),
            'cache_size': len(self.query_cache)
        })
        
        # Graph connectivity stats
        if self.graph.nodes():
            stats['avg_degree'] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
            stats['is_connected'] = nx.is_weakly_connected(self.graph)
        
        return stats
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Graph retriever cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Check retriever health"""
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        if self.graph.number_of_nodes() == 0:
            health['status'] = 'unhealthy'
            health['issues'].append('No nodes in graph')
        
        if self.graph.number_of_edges() == 0:
            health['status'] = 'unhealthy'
            health['issues'].append('No edges in graph')
        
        if not self.entity_to_docs:
            health['status'] = 'warning'
            health['issues'].append('No entity-document mappings')
        
        return health