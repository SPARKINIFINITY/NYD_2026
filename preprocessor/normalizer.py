from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DynamicQueryNormalizer:
    """
    Dynamically rewrites incomplete or shorthand user queries into
    clean, standardized English questions using LLM inference.
    """

    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def normalize(self, query: str, context: str = None) -> str:
        """
        Dynamically rewrites a query into a clear, grammatically correct,
        contextually complete English question.
        """

        prompt = f"""
        Rewrite the following user query into a complete, normalized, and explicit English question.
        Do NOT change its meaning. Add missing context, grammar, and structure if needed.

        Example:
        - User query: "ipl t20 25 score?"
          Normalized: "What is the score of the IPL T20 match in 2025?"

        - User query: "ramayana king father"
          Normalized: "Who is the father of the king mentioned in the Ramayana?"

        User query: "{query}"
        """

        if context:
            prompt += f"\nAdditional context: {context}"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        normalized = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return normalized


# ------------------------------
# ✅ Example Usage / Test
# ------------------------------
if __name__ == "__main__":
    normalizer = DynamicQueryNormalizer()

    test_queries = [
        "ipl t20 25 score?",
        "ramayana king father",
        "weather delhi tomorrow",
        "gdp india vs china 2023",
        "ai use in medicine"
    ]

    for q in test_queries:
        norm = normalizer.normalize(q)
        print(f"\n🔹 User Query: {q}")
        print(f"🔸 Normalized: {norm}")
