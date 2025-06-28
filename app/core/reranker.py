from typing import List, Dict
from app.config import Config

class Reranker:
    def __init__(self, tier: str = "free"):
        self.tier = tier.lower()
        if self.tier in ["plus", "pro"]:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            self.model = None

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if self.model is None:
            # Free tier: skip reranking
            return chunks

        pairs = [(query, chunk["text"]) for chunk in chunks]
        if not pairs:
            return [] 
        scores = self.model.predict(pairs)

        # Inject scores back into chunks
        for i, score in enumerate(scores):
            chunks[i]["score"] = float(score)
            chunks[i]["source"] = chunks[i].get("source", "reranked")

        return sorted(chunks, key=lambda x: -x["score"])
