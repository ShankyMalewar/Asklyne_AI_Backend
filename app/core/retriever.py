# app/core/retriever.py

from app.services.qdrant_service import QdrantService
from app.services.typesense_service import TypesenseService
from app.core.embedder import Embedder

class Retriever:
    def __init__(self, tier: str):
        self.tier = tier.lower()
        self.embedder = Embedder(tier=self.tier)
        self.qdrant = QdrantService(tier=self.tier)
        self.typesense = TypesenseService(tier=self.tier)

    def retrieve(self, query: str, session_id: str, mode: str, top_k: int = 5) -> list[dict]:
        # Embed the query
        query_vec = self.embedder.embed_chunks([query])[0]

        # Semantic search
        semantic_hits = self.qdrant.search(
            query_embedding=query_vec,
            top_k=top_k,
            filters={"session_id": session_id, "mode": mode}
        )
        semantic_chunks = [
            {"text": hit.payload["text"], "source": "qdrant", "score": hit.score}
            for hit in semantic_hits
        ]

        # Keyword search
        keyword_hits = self.typesense.search(
            query=query,
            top_k=top_k,
            filters={"session_id": session_id, "mode": mode}
        )
        keyword_chunks = [
            {"text": hit["document"]["text"], "source": "typesense", "score": hit["text_match_score"]}
            for hit in keyword_hits
        ]

        # Merge & deduplicate
        all_chunks = {c["text"]: c for c in semantic_chunks + keyword_chunks}  # dedup by text
        return list(all_chunks.values())
