import spacy
import json
import requests
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict

class EntityRelationExtractor:
    """Extracts entities using spaCy and relations using LLM prompts"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm", llm_endpoint: str = None):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model '{spacy_model}' not found. Please install it with: python -m spacy download {spacy_model}")
            self.nlp = None
        
        self.llm_endpoint = llm_endpoint
        self.entity_types = ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'DATE', 'MONEY', 'QUANTITY']
        
    def extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores by default
                }
                entities.append(entity)
        
        return entities
    
    def extract_relations_llm_prompt(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using LLM with prompt-based approach"""
        if not entities or len(entities) < 2:
            return []
        
        # Create entity pairs for relation extraction
        entity_pairs = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity_pairs.append((entities[i], entities[j]))
        
        relations = []
        
        # Use rule-based approach if LLM endpoint not available
        if not self.llm_endpoint:
            relations = self._extract_relations_rule_based(text, entity_pairs)
        else:
            relations = self._extract_relations_llm(text, entity_pairs)
        
        return relations
    
    def _extract_relations_rule_based(self, text: str, entity_pairs: List[Tuple]) -> List[Dict[str, Any]]:
        """Extract relations using rule-based patterns"""
        relations = []
        
        # Define relation patterns
        relation_patterns = {
            'WORKS_FOR': [
                r'{entity1}.*(?:works for|employed by|employee of).*{entity2}',
                r'{entity2}.*(?:employs|hires).*{entity1}'
            ],
            'LOCATED_IN': [
                r'{entity1}.*(?:in|located in|based in).*{entity2}',
                r'{entity2}.*(?:contains|includes).*{entity1}'
            ],
            'FOUNDED_BY': [
                r'{entity1}.*(?:founded by|created by|established by).*{entity2}',
                r'{entity2}.*(?:founded|created|established).*{entity1}'
            ],
            'PART_OF': [
                r'{entity1}.*(?:part of|division of|subsidiary of).*{entity2}',
                r'{entity2}.*(?:includes|contains|owns).*{entity1}'
            ],
            'RELATED_TO': [
                r'{entity1}.*(?:related to|associated with|connected to).*{entity2}',
                r'{entity2}.*(?:related to|associated with|connected to).*{entity1}'
            ]
        }
        
        for entity1, entity2 in entity_pairs:
            # Skip if entities are too close (likely same entity)
            if abs(entity1['start'] - entity2['start']) < 10:
                continue
            
            # Check for relation patterns
            for relation_type, patterns in relation_patterns.items():
                for pattern in patterns:
                    # Create regex pattern with entity placeholders
                    regex_pattern = pattern.format(
                        entity1=re.escape(entity1['text']),
                        entity2=re.escape(entity2['text'])
                    )
                    
                    if re.search(regex_pattern, text, re.IGNORECASE):
                        relation = {
                            'source_entity': entity1,
                            'target_entity': entity2,
                            'relation_type': relation_type,
                            'confidence': 0.8,
                            'extraction_method': 'rule_based'
                        }
                        relations.append(relation)
                        break
        
        return relations
    
    def _extract_relations_llm(self, text: str, entity_pairs: List[Tuple]) -> List[Dict[str, Any]]:
        """Extract relations using LLM API (placeholder for actual implementation)"""
        relations = []
        
        # This is a placeholder for LLM API integration
        # You would replace this with actual API calls to your chosen LLM
        
        for entity1, entity2 in entity_pairs:
            prompt = self._create_relation_extraction_prompt(text, entity1, entity2)
            
            # Placeholder for LLM API call
            # response = self._call_llm_api(prompt)
            # relation_type = self._parse_llm_response(response)
            
            # For now, use rule-based as fallback
            rule_based_relations = self._extract_relations_rule_based(text, [(entity1, entity2)])
            relations.extend(rule_based_relations)
        
        return relations
    
    def _create_relation_extraction_prompt(self, text: str, entity1: Dict, entity2: Dict) -> str:
        """Create a prompt for LLM relation extraction"""
        prompt = f"""
        Given the following text and two entities, determine if there is a semantic relationship between them.
        
        Text: "{text}"
        
        Entity 1: "{entity1['text']}" (Type: {entity1['label']})
        Entity 2: "{entity2['text']}" (Type: {entity2['label']})
        
        Possible relationship types:
        - WORKS_FOR: Entity 1 works for Entity 2
        - LOCATED_IN: Entity 1 is located in Entity 2
        - FOUNDED_BY: Entity 1 was founded by Entity 2
        - PART_OF: Entity 1 is part of Entity 2
        - RELATED_TO: General relationship between entities
        - NONE: No clear relationship
        
        Please respond with only the relationship type or "NONE" if no relationship exists.
        """
        return prompt
    
    def create_knowledge_graph(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a knowledge graph from extracted entities and relations"""
        graph = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_chunks_processed': len(chunks),
                'extraction_timestamp': None
            }
        }
        
        # Track unique entities and relations
        entity_map = {}  # text -> node_id
        node_id_counter = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_content = chunk.get('content', '')
            
            # Convert content to string if needed
            if isinstance(chunk_content, (list, dict)):
                chunk_content = json.dumps(chunk_content, ensure_ascii=False)
            
            # Extract entities
            entities = self.extract_entities_spacy(str(chunk_content))
            
            # Add entities as nodes
            chunk_entities = []
            for entity in entities:
                entity_key = f"{entity['text'].lower()}_{entity['label']}"
                
                if entity_key not in entity_map:
                    node_id = f"node_{node_id_counter}"
                    entity_map[entity_key] = node_id
                    node_id_counter += 1
                    
                    graph['nodes'].append({
                        'id': node_id,
                        'label': entity['text'],
                        'type': entity['label'],
                        'properties': {
                            'original_text': entity['text'],
                            'entity_type': entity['label'],
                            'first_seen_chunk': chunk_idx,
                            'occurrences': 1
                        }
                    })
                else:
                    # Update occurrence count
                    node_id = entity_map[entity_key]
                    for node in graph['nodes']:
                        if node['id'] == node_id:
                            node['properties']['occurrences'] += 1
                            break
                
                chunk_entities.append({
                    'entity': entity,
                    'node_id': entity_map[entity_key]
                })
            
            # Extract relations
            relations = self.extract_relations_llm_prompt(str(chunk_content), entities)
            
            # Add relations as edges
            for relation in relations:
                source_key = f"{relation['source_entity']['text'].lower()}_{relation['source_entity']['label']}"
                target_key = f"{relation['target_entity']['text'].lower()}_{relation['target_entity']['label']}"
                
                if source_key in entity_map and target_key in entity_map:
                    edge = {
                        'source': entity_map[source_key],
                        'target': entity_map[target_key],
                        'relation_type': relation['relation_type'],
                        'properties': {
                            'confidence': relation.get('confidence', 0.5),
                            'extraction_method': relation.get('extraction_method', 'unknown'),
                            'source_chunk': chunk_idx
                        }
                    }
                    graph['edges'].append(edge)
        
        # Add graph statistics
        graph['metadata'].update({
            'total_nodes': len(graph['nodes']),
            'total_edges': len(graph['edges']),
            'node_types': list(set(node['type'] for node in graph['nodes'])),
            'relation_types': list(set(edge['relation_type'] for edge in graph['edges']))
        })
        
        return graph
    
    def save_graph_to_json(self, graph: Dict[str, Any], output_path: str):
        """Save the knowledge graph to a JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, indent=2, ensure_ascii=False)
            print(f"Knowledge graph saved to {output_path}")
        except Exception as e:
            print(f"Error saving graph: {str(e)}")
    
    def get_graph_statistics(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        stats = {
            'nodes': len(graph['nodes']),
            'edges': len(graph['edges']),
            'node_types': {},
            'relation_types': {},
            'most_connected_entities': []
        }
        
        # Count node types
        for node in graph['nodes']:
            node_type = node['type']
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count relation types
        for edge in graph['edges']:
            rel_type = edge['relation_type']
            stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1
        
        # Find most connected entities
        connection_count = defaultdict(int)
        for edge in graph['edges']:
            connection_count[edge['source']] += 1
            connection_count[edge['target']] += 1
        
        # Get top 5 most connected entities
        top_connected = sorted(connection_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for node_id, count in top_connected:
            # Find node label
            node_label = next((node['label'] for node in graph['nodes'] if node['id'] == node_id), node_id)
            stats['most_connected_entities'].append({
                'entity': node_label,
                'connections': count
            })
        
        return stats