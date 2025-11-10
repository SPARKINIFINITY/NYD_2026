# =========================================
# 🌍 Universal Intent Detection Module (Hybrid with Planner Mapping)
# =========================================

from sentence_transformers import SentenceTransformer
import numpy as np
import re

# ----------------------------
# Utility: Normalize queries
# ----------------------------
def normalize_query(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# Intent definitions 
# ----------------------------
INTENTS = [
    {"intent_name": "fact", "description": "Questions asking for facts, dates, names, or specific factual information."},
    {"intent_name": "explain", "description": "Requests for explanations, definitions, meanings, or how things work."},
    {"intent_name": "compare", "description": "Requests to compare two or more things, entities, or concepts."},
    {"intent_name": "table", "description": "Requests for tabular data, structured information, or data in table format."},
    {"intent_name": "code", "description": "Requests for code examples, programming help, or technical implementation."},
    {"intent_name": "multi-hop", "description": "Complex questions requiring multi-step reasoning, deep analysis, or connecting multiple pieces of information."},
    {"intent_name": "clarify", "description": "Requests for clarification, follow-up questions, or seeking more details about previous responses."},
    {"intent_name": "irrelevant", "description": "Small talk, casual conversation, greetings, abusive language, or queries not related to informational content."}
]

# ----------------------------
# Retrieval config mapping
# ----------------------------
RETRIEVAL_CONFIGS = {
    "fact": {"retrievers": ["dense","bm25"], "dense_k":50, "bm25_k":50, "rerank":True, "multihop":0},
    "explain": {"retrievers": ["dense","bm25"], "dense_k":50, "bm25_k":50, "rerank":True, "multihop":0},
    "compare": {"retrievers": ["dense","bm25"], "dense_k":100, "bm25_k":100, "rerank":True, "multihop":0},
    "table": {"retrievers": ["dense","bm25"], "dense_k":75, "bm25_k":75, "rerank":True, "multihop":0},
    "code": {"retrievers": ["dense"], "dense_k":50, "rerank":True, "multihop":0},
    "multi-hop": {"retrievers": ["dense","graph"], "dense_k":100, "graph_k":50, "rerank":True, "multihop":2},
    "clarify": {"retrievers": ["dense"], "dense_k":30, "rerank":True, "multihop":0},
    "irrelevant": {"retrievers": [], "dense_k":0, "rerank":False, "multihop":0}
}

# =========================================
# Universal Intent Detector Class
# =========================================
class UniversalIntentDetector:
    _model_instance = None
    _intent_embeddings = None

    def __init__(self, intents, retrieval_configs, model_name="all-mpnet-base-v2", threshold=0.4):
        self.INTENTS = intents
        self.retrieval_configs = retrieval_configs
        self.threshold = threshold

        # Load model once
        if UniversalIntentDetector._model_instance is None:
            UniversalIntentDetector._model_instance = SentenceTransformer(model_name, device="cpu")
        self.model = UniversalIntentDetector._model_instance

        # Embed intents once
        if UniversalIntentDetector._intent_embeddings is None:
            intent_texts = [
                f"{i['intent_name']} | {i['description']}" for i in self.INTENTS
            ]
            UniversalIntentDetector._intent_embeddings = self.model.encode(intent_texts, normalize_embeddings=True)
        self.intent_embeddings = UniversalIntentDetector._intent_embeddings

    def detect_intent(self, query, top_k=1):
        query_norm = normalize_query(query)
        query_emb = self.model.encode(query_norm, normalize_embeddings=True)
        sims = np.dot(self.intent_embeddings, query_emb)

        # Normalize to 0-1
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idxs:
            intent_name = self.INTENTS[i]['intent_name']
            conf = float(sims[i])
            results.append({
                "intent": intent_name,
                "confidence": conf,
                "retrieval_config": self.retrieval_configs.get(intent_name, {}),
                "needs_clarification": conf < self.threshold
            })
        return results

# =========================================
# 🔹 Test
# =========================================
if __name__ == "__main__":
    detector = UniversalIntentDetector(INTENTS, RETRIEVAL_CONFIGS)

    test_queries = [
        "Why did Rama go to exile?",
        "Who is Hanuman?",
        "Where was Krishna born?",
        "Compare karma and destiny.",
        "Tell me the story of Sita’s abduction.",
        "What is quantum entanglement?",
        "Why does gravity exist?",
        "How to calculate kinetic energy?",
        "When was the theory of relativity published?",
        "Compare mitosis and meiosis.",
        "Explain the meaning of inflation.",
        "How can I invest in mutual funds?",
        "Hey, how are you?",
        "Thanks for your help!",
        "I want to understand myself better."
    ]

    for q in test_queries:
        intents = detector.detect_intent(q, top_k=2)
        print(f"\nOriginal: {q}")
        for r in intents:
            print(f"Detected Intent: {r['intent']} | Confidence: {r['confidence']:.3f} | Needs Clarification: {r['needs_clarification']}")
        print("-" * 80)
