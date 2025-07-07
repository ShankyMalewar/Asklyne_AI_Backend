from app.services.qdrant_service import QdrantService
from app.services.typesense_service import TypesenseService
from app.core.embedder import Embedder
import asyncio


class Retriever:
    def __init__(self, tier: str, mode: str):
        self.tier = tier.lower()
        self.mode = mode.lower()

        self.embedder = Embedder(tier=self.tier, mode=self.mode)
        self.qdrant = QdrantService(tier=self.tier, mode=self.mode)
        self.typesense = TypesenseService(tier=self.tier)


    async def retrieve_async(self, query: str, session_id: str, mode: str, top_k: int = 5) -> list[dict]:
        if not self.embedder or self.embedder.mode != mode:
            self.embedder = Embedder(tier=self.tier, mode=mode)

        query_vec = self.embedder.embed_chunks([query])[0]

        semantic_task = asyncio.to_thread(
            self.qdrant.search,
            query_embedding=query_vec,
            top_k=top_k,
            filters={"session_id": session_id, "mode": mode}
        )
        keyword_task = asyncio.to_thread(
            self.typesense.search,
            query=query,
            top_k=top_k,
            filters={"session_id": session_id, "mode": mode}
        )

        semantic_hits, keyword_hits = await asyncio.gather(semantic_task, keyword_task)

        semantic_chunks = [
            {"text": hit.payload["text"], "source": "qdrant", "score": hit.score}
            for hit in semantic_hits
        ]

        keyword_chunks = [
            {
                "text": hit["document"].get("text", ""),
                "source": "typesense",
                "score": hit.get("text_match_score", 0.0)
            }
            for hit in keyword_hits
        ]

        all_chunks = {c["text"]: c for c in semantic_chunks + keyword_chunks}
        return list(all_chunks.values())
