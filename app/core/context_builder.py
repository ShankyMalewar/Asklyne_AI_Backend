from app.config import Config
class ContextBuilder:
    def __init__(self, tier: str):
        self.tier = tier.lower()
        self.token_limit = Config.TOKEN_LIMITS[self.tier]
        self.safety_margin = 0.9  # Use 90% of token budget for context

    def build(self, chunks: list[dict]) -> str:
        """
        Merges top chunks into a token-safe string for LLM input.
        """
        max_tokens = int(self.token_limit * self.safety_margin)
        total_tokens = 0
        selected_chunks = []

        # Sort chunks by score if available (desc)
        chunks_sorted = sorted(chunks, key=lambda x: -x.get("score", 0))

        for chunk in chunks_sorted:
            text = chunk["text"].strip()
            token_est = self.estimate_tokens(text)
            if total_tokens + token_est > max_tokens:
                break
            selected_chunks.append(text)
            total_tokens += token_est

        return "\n---\n".join(selected_chunks)

    def estimate_tokens(self, text: str) -> int:
        """
        Approximate token count (safe heuristic: 1 token = ~4 chars)
        """
        return max(1, len(text) // 4)
