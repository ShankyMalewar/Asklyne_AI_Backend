# app/services/qdrant_service.py

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers.util import cos_sim
import uuid
from app.config import Config
from app.core.embedder import Embedder
from typing import List

QDRANT_HOST = Config.QDRANT_HOST
QDRANT_PORT = Config.QDRANT_PORT
QDRANT_API_KEY = Config.QDRANT_API_KEY




class QdrantService:
    def __init__(self, tier: str):
        self.tier = tier.lower()
        self.collection_name = f"asklyne_chunks_{self.tier}"
        self.vector_size = 1024  

        self.client = QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
        )

        self.embedder = Embedder(tier=self.tier)
        self.ensure_collection_exists()

    def ensure_collection_exists(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            print(f"✅ Created new collection: {self.collection_name}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"ℹ️ Collection {self.collection_name} already exists.")
            else:
                raise e

    def upsert_chunks(self, chunks: list[str], embeddings: list[list[float]], session_id: str, mode: str):
        payloads = []
        for i, chunk in enumerate(chunks):
            payloads.append(
                PointStruct(
                    id=i,
                    vector=embeddings[i],
                    payload={
                        "session_id": session_id,
                        "tier": self.tier,
                        "mode": mode,
                        "text": chunk,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=payloads)
        print(f"✅ Upserted {len(payloads)} chunks to {self.collection_name}")

    def search(self, query_embedding, top_k=5, filters: dict = None):
        """
        Perform similarity search with optional metadata filter.
        """
        query_filter = None
        if filters:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=val))
                    for key, val in filters.items()
                ]
            )

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter
        )

        return hits
