# Enhanced Entity and Relation Extractor

import spacy
import re
from typing import List, Dict, Tuple, Set, Any, Optional
from transformers import pipeline
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Load models with error handling
# -----------------------------
try:
    nlp = spacy.load("en_core_web_sm")
    # Add custom entity ruler for domain-specific entities
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        # Add patterns for common entities that spaCy might miss
        patterns = [
            {"label": "DEITY", "pattern": [{"LOWER": {"IN": ["rama", "krishna", "shiva", "vishnu", "hanuman", "ganesha", "durga", "lakshmi", "saraswati"]}}]},
            {"label": "EPIC", "pattern": [{"LOWER": {"IN": ["ramayana", "mahabharata", "bhagavad", "gita", "puranas"]}}]},
            {"label": "PLACE", "pattern": [{"LOWER": {"IN": ["ayodhya", "lanka", "mathura", "vrindavan", "kurukshetra", "hastinapura"]}}]},
            {"label": "CONCEPT", "pattern": [{"LOWER": {"IN": ["dharma", "karma", "moksha", "samsara", "ahimsa", "yoga"]}}]}
        ]
        ruler.add_patterns(patterns)
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
    nlp = None

try:
    relation_extractor = pipeline("text2text-generation", model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
, device=-1)  # Use CPU
except Exception as e:
    logger.error(f"Error loading relation extraction model: {e}")
    relation_extractor = None


# -----------------------------
# Enhanced Helper Functions
# -----------------------------
def get_conjuncts(token) -> List:
    """
    Recursively get all conjuncts (like 'Rama and Lakshmana')
    """
    tokens = [token]
    for child in token.children:
        if child.dep_ == "conj":
            tokens.extend(get_conjuncts(child))
    return tokens

def get_compound_entities(doc) -> List[str]:
    """
    Extract compound entities (multi-word entities) that spaCy might miss
    """
    compounds = []
    for token in doc:
        if token.dep_ == "compound" and token.head.pos_ in ["NOUN", "PROPN"]:
            # Get the full compound phrase
            compound_tokens = [token]
            head = token.head
            
            # Collect all compound modifiers
            for child in head.children:
                if child.dep_ == "compound" and child != token:
                    compound_tokens.append(child)
            
            compound_tokens.append(head)
            compound_tokens.sort(key=lambda x: x.i)  # Sort by position
            compound_phrase = " ".join([t.text for t in compound_tokens])
            compounds.append(compound_phrase)
    
    return list(set(compounds))

def resolve_coreferences(doc) -> str:
    """
    Enhanced pronoun/coreference resolution using context and entity tracking
    """
    resolved_text = []
    entity_stack = []  # Stack to track recent entities
    
    for token in doc:
        if token.ent_type_ or token.pos_ == "PROPN":
            entity_stack.append(token.text)
            if len(entity_stack) > 3:  # Keep only recent 3 entities
                entity_stack.pop(0)
            resolved_text.append(token.text)
        elif token.lower_ in ["he", "she", "they", "who", "him", "her", "his", "hers", "their"]:
            # Use most recent appropriate entity
            if entity_stack:
                resolved_text.append(entity_stack[-1])
            else:
                resolved_text.append(token.text)
        else:
            resolved_text.append(token.text)
    
    return " ".join(resolved_text)

def clean_entity_text(text: str) -> str:
    """
    Clean and normalize entity text
    """
    # Remove extra whitespace and punctuation
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text)  # Remove leading/trailing non-word chars
    return text

def filter_valid_entities(entities: List[str]) -> List[str]:
    """
    Filter out invalid or low-quality entities
    """
    filtered = []
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    
    for entity in entities:
        entity_clean = clean_entity_text(entity)
        
        # Skip if empty, too short, or just stop words
        if (len(entity_clean) < 2 or 
            entity_clean.lower() in stop_words or
            len(entity_clean.split()) > 5 or  # Skip very long phrases
            entity_clean.isdigit()):  # Skip pure numbers
            continue
            
        filtered.append(entity_clean)
    
    return list(set(filtered))  # Remove duplicates


# -----------------------------
# Enhanced Entity Extraction
# -----------------------------
def extract_entities_comprehensive(query: str) -> Dict[str, Any]:
    """
    Comprehensive entity extraction with multiple strategies
    """
    if not nlp:
        return {"entities": [], "entity_types": {}, "confidence_scores": {}}
    
    doc = nlp(query)
    entities_info = {
        "entities": [],
        "entity_types": {},
        "confidence_scores": {},
        "entity_positions": {}
    }
    
    # Strategy 1: spaCy NER entities
    spacy_entities = []
    for ent in doc.ents:
        entity_text = clean_entity_text(ent.text)
        if entity_text:
            spacy_entities.append(entity_text)
            entities_info["entity_types"][entity_text] = ent.label_
            entities_info["confidence_scores"][entity_text] = 0.9  # High confidence for spaCy NER
            entities_info["entity_positions"][entity_text] = (ent.start_char, ent.end_char)
    
    # Strategy 2: Proper nouns not caught by NER
    proper_nouns = []
    for token in doc:
        if token.pos_ == "PROPN" and not any(token.text in ent for ent in spacy_entities):
            entity_text = clean_entity_text(token.text)
            if entity_text:
                proper_nouns.append(entity_text)
                entities_info["entity_types"][entity_text] = "PROPN"
                entities_info["confidence_scores"][entity_text] = 0.7
                entities_info["entity_positions"][entity_text] = (token.idx, token.idx + len(token.text))
    
    # Strategy 3: Compound entities
    compound_entities = get_compound_entities(doc)
    for comp_ent in compound_entities:
        entity_text = clean_entity_text(comp_ent)
        if entity_text and entity_text not in spacy_entities:
            entities_info["entity_types"][entity_text] = "COMPOUND"
            entities_info["confidence_scores"][entity_text] = 0.6
    
    # Strategy 4: Pattern-based extraction for specific domains
    pattern_entities = extract_pattern_entities(query)
    for pattern_ent, ent_type in pattern_entities.items():
        if pattern_ent not in entities_info["entity_types"]:
            entities_info["entity_types"][pattern_ent] = ent_type
            entities_info["confidence_scores"][pattern_ent] = 0.8
    
    # Combine all entities
    all_entities = list(set(spacy_entities + proper_nouns + compound_entities + list(pattern_entities.keys())))
    entities_info["entities"] = filter_valid_entities(all_entities)
    
    return entities_info

def extract_pattern_entities(query: str) -> Dict[str, str]:
    """
    Extract entities using regex patterns for specific domains
    """
    patterns = {
        # Dates
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b': 'DATE',
        r'\b\d{4}\b': 'YEAR',
        
        # Numbers and quantities
        r'\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|crore|lakh)\b': 'QUANTITY',
        r'\b\d+(?:\.\d+)?\s*(?:kg|km|meter|feet|inch|pound|gram)\b': 'MEASUREMENT',
        
        # Technical terms
        r'\b(?:API|HTTP|JSON|XML|SQL|HTML|CSS|JavaScript|Python|Java|C\+\+)\b': 'TECHNOLOGY',
        
        # Religious/mythological terms
        r'\b(?:Ramayana|Mahabharata|Bhagavad Gita|Vedas|Upanishads|Puranas)\b': 'SCRIPTURE',
        r'\b(?:dharma|karma|moksha|samsara|ahimsa|yoga|meditation)\b': 'CONCEPT',
        
        # Geographic patterns
        r'\b(?:Mount|Mt\.)\s+[A-Z][a-z]+\b': 'MOUNTAIN',
        r'\b(?:River|Lake)\s+[A-Z][a-z]+\b': 'WATER_BODY',
    }
    
    found_entities = {}
    query_lower = query.lower()
    
    for pattern, entity_type in patterns.items():
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            entity_text = clean_entity_text(match.group())
            if entity_text:
                found_entities[entity_text] = entity_type
    
    return found_entities

def extract_entities(query: str) -> List[str]:
    """
    Simplified entity extraction function for backward compatibility
    """
    entities_info = extract_entities_comprehensive(query)
    return entities_info["entities"]


# -----------------------------
# Enhanced Relation Extraction
# -----------------------------
def extract_syntactic_relations(doc) -> List[Tuple[str, str, str]]:
    """
    Extract syntactic relations with improved accuracy
    """
    relations = []
    
    for token in doc:
        if token.pos_ in ["VERB", "AUX"]:
            subjects = []
            objects = []
            
            # Get subjects
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    subjects.extend([clean_entity_text(t.text) for t in get_conjuncts(child)])
                elif child.dep_ in ["dobj", "iobj", "pobj"]:
                    objects.extend([clean_entity_text(t.text) for t in get_conjuncts(child)])
                elif child.dep_ == "prep":
                    # Handle prepositional relations
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":
                            objects.extend([clean_entity_text(t.text) for t in get_conjuncts(prep_child)])
            
            # Create relations
            verb = clean_entity_text(token.lemma_)
            for subj in subjects:
                for obj in objects:
                    if subj and obj and verb:
                        relations.append((subj, verb, obj))
    
    return relations

def extract_semantic_relations_patterns(query: str, entities: List[str]) -> List[Dict[str, Any]]:
    """
    Extract semantic relations using predefined patterns
    """
    relations = []
    query_lower = query.lower()
    
    # Define relation patterns
    relation_patterns = {
        'IS_A': [r'{entity1}\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*{entity2}',
                 r'{entity1}\s+(?:is|was|are|were)\s+(?:known as|called)\s+{entity2}'],
        'PART_OF': [r'{entity1}\s+(?:is|was)\s+(?:part of|in|within)\s+{entity2}',
                    r'{entity2}\s+(?:contains|includes|has)\s+{entity1}'],
        'LOCATED_IN': [r'{entity1}\s+(?:is|was)\s+(?:in|at|located in|situated in)\s+{entity2}',
                       r'{entity1}\s+(?:of|from)\s+{entity2}'],
        'RELATED_TO': [r'{entity1}\s+(?:and|with|related to|associated with)\s+{entity2}'],
        'FATHER_OF': [r'{entity1}\s+(?:is|was)\s+(?:the\s+)?(?:father|dad)\s+of\s+{entity2}',
                      r'{entity2}\s+(?:is|was)\s+(?:the\s+)?(?:son|child)\s+of\s+{entity1}'],
        'MOTHER_OF': [r'{entity1}\s+(?:is|was)\s+(?:the\s+)?(?:mother|mom)\s+of\s+{entity2}',
                      r'{entity2}\s+(?:is|was)\s+(?:the\s+)?(?:daughter|child)\s+of\s+{entity1}'],
        'RULED': [r'{entity1}\s+(?:ruled|governed|was king of|was queen of)\s+{entity2}',
                  r'{entity2}\s+(?:was ruled by|was governed by)\s+{entity1}'],
        'BORN_IN': [r'{entity1}\s+(?:was born in|born in|from)\s+{entity2}'],
        'DEFEATED': [r'{entity1}\s+(?:defeated|killed|destroyed|conquered)\s+{entity2}'],
        'MARRIED_TO': [r'{entity1}\s+(?:married|wed)\s+{entity2}',
                       r'{entity1}\s+(?:and|with)\s+{entity2}\s+(?:married|wed)']
    }
    
    # Check each entity pair against patterns
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j:  # Don't relate entity to itself
                for relation_type, patterns in relation_patterns.items():
                    for pattern in patterns:
                        # Create regex with entity placeholders
                        regex_pattern = pattern.format(
                            entity1=re.escape(entity1.lower()),
                            entity2=re.escape(entity2.lower())
                        )
                        
                        if re.search(regex_pattern, query_lower):
                            relations.append({
                                'subject': entity1,
                                'relation': relation_type,
                                'object': entity2,
                                'confidence': 0.8,
                                'method': 'pattern_based'
                            })
                            break
    
    return relations

def extract_relations_llm_enhanced(query: str, entities: List[str]) -> str:
    """
    Enhanced LLM-based relation extraction with better prompting
    """
    if not relation_extractor or not entities:
        return ""
    
    try:
        entities_str = ", ".join(entities[:10])  # Limit to avoid token overflow
        
        prompt = f"""
        Given this sentence and the identified entities, extract the relationships between entities.
        
        Sentence: "{query}"
        Entities: {entities_str}
        
        Extract relationships in this format:
        - Entity1 RELATION Entity2
        - Entity1 RELATION Entity2
        
        Focus on meaningful relationships like: father_of, son_of, ruled, born_in, located_in, defeated, married_to, part_of.
        
        Relationships:
        """
        
        response = relation_extractor(prompt, max_new_tokens=100, temperature=0.3)[0]['generated_text']
        return response.strip()
        
    except Exception as e:
        logger.error(f"LLM relation extraction failed: {e}")
        return ""

def extract_relations_comprehensive(query: str, entities: List[str]) -> Dict[str, Any]:
    """
    Comprehensive relation extraction combining multiple approaches
    """
    if not nlp:
        return {"syntactic_relations": [], "semantic_relations": [], "llm_relations": ""}
    
    doc = nlp(query)
    query_resolved = resolve_coreferences(doc)
    resolved_doc = nlp(query_resolved)
    
    # Extract different types of relations
    syntactic_relations = extract_syntactic_relations(resolved_doc)
    semantic_relations = extract_semantic_relations_patterns(query_resolved, entities)
    llm_relations = extract_relations_llm_enhanced(query_resolved, entities)
    
    return {
        "syntactic_relations": syntactic_relations,
        "semantic_relations": semantic_relations,
        "llm_relations": llm_relations,
        "resolved_query": query_resolved
    }

def extract_relations_dynamic(query: str):
    """
    Backward compatibility function
    """
    entities = extract_entities(query)
    relations_info = extract_relations_comprehensive(query, entities)
    
    return relations_info["syntactic_relations"], relations_info["llm_relations"]


# -----------------------------
# Enhanced Unified Functions
# -----------------------------
def extract_entities_and_relations_comprehensive(query: str) -> Dict[str, Any]:
    """
    Comprehensive extraction of entities and relations with detailed information
    """
    # Extract entities with detailed information
    entities_info = extract_entities_comprehensive(query)
    entities = entities_info["entities"]
    
    # Extract relations with multiple approaches
    relations_info = extract_relations_comprehensive(query, entities)
    
    return {
        "entities": entities,
        "entity_details": entities_info,
        "relations": relations_info,
        "summary": {
            "entity_count": len(entities),
            "syntactic_relation_count": len(relations_info["syntactic_relations"]),
            "semantic_relation_count": len(relations_info["semantic_relations"]),
            "has_llm_relations": bool(relations_info["llm_relations"])
        }
    }

def extract_entities_and_relations(query: str):
    """
    Backward compatible function for existing code
    Returns:
        entities: list of entity strings
        base_relations: list of (subject, verb, object)
        llm_relations: string output from LLM
    """
    comprehensive_result = extract_entities_and_relations_comprehensive(query)
    
    entities = comprehensive_result["entities"]
    base_relations = comprehensive_result["relations"]["syntactic_relations"]
    llm_relations = comprehensive_result["relations"]["llm_relations"]
    
    return entities, base_relations, llm_relations

def get_entity_statistics(query: str) -> Dict[str, Any]:
    """
    Get detailed statistics about entities in the query
    """
    entities_info = extract_entities_comprehensive(query)
    
    # Count entities by type
    entity_type_counts = Counter(entities_info["entity_types"].values())
    
    # Calculate confidence statistics
    confidences = list(entities_info["confidence_scores"].values())
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        "total_entities": len(entities_info["entities"]),
        "entity_types": dict(entity_type_counts),
        "average_confidence": avg_confidence,
        "high_confidence_entities": [
            entity for entity, conf in entities_info["confidence_scores"].items() 
            if conf >= 0.8
        ],
        "entities_with_positions": entities_info["entity_positions"]
    }

# -----------------------------
# Enhanced Test and Demo
# -----------------------------
def demo_entity_extraction():
    """
    Demonstration of enhanced entity extraction capabilities
    """
    test_queries = [
        "Rama is the son of King Dasharatha who ruled Ayodhya, the capital of Kosala.",
        "Krishna was born in Mathura and later moved to Vrindavan with his foster parents.",
        "The Ramayana tells the story of how Rama defeated Ravana in Lanka.",
        "Python is a programming language used for machine learning and web development.",
        "Mount Everest is located in the Himalayas between Nepal and Tibet.",
        "The GDP of India in 2023 was approximately 3.7 trillion dollars."
    ]
    
    print("=" * 80)
    print("ENHANCED ENTITY EXTRACTOR DEMONSTRATION")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-" * 60)
        
        # Get comprehensive results
        result = extract_entities_and_relations_comprehensive(query)
        
        print(f"Entities ({result['summary']['entity_count']}): {result['entities']}")
        
        if result['entity_details']['entity_types']:
            print("Entity Types:")
            for entity, etype in result['entity_details']['entity_types'].items():
                conf = result['entity_details']['confidence_scores'].get(entity, 0)
                print(f"  - {entity}: {etype} (confidence: {conf:.2f})")
        
        if result['relations']['syntactic_relations']:
            print(f"Syntactic Relations ({result['summary']['syntactic_relation_count']}):")
            for rel in result['relations']['syntactic_relations']:
                print(f"  - {rel[0]} → {rel[1]} → {rel[2]}")
        
        if result['relations']['semantic_relations']:
            print(f"Semantic Relations ({result['summary']['semantic_relation_count']}):")
            for rel in result['relations']['semantic_relations']:
                print(f"  - {rel['subject']} → {rel['relation']} → {rel['object']} (conf: {rel['confidence']:.2f})")
        
        if result['relations']['llm_relations']:
            print(f"LLM Relations: {result['relations']['llm_relations']}")

if __name__ == "__main__":
    demo_entity_extraction()
