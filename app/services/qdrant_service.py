# app/services/qdrant_service.py

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType, Filter, FieldCondition, MatchValue
from sentence_transformers.util import cos_sim
from app.config import Config
from app.core.embedder import Embedder
from typing import List
import uuid

QDRANT_HOST = Config.QDRANT_HOST
QDRANT_API_KEY = Config.QDRANT_API_KEY


class QdrantService:
    def __init__(self, tier: str, mode: str):
        self.tier = tier.lower()
        self.mode = mode.lower()
        self.collection_name = f"asklyne_chunks_{self.tier}_{self.mode}"
        self.vector_size = self.get_vector_size()


        self.client = QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY
        )

        self.embedder = Embedder(tier=self.tier)
        self.ensure_collection_exists()
        
    def get_vector_size(self) -> int:
        # You can customize these as needed
        VECTOR_DIM_MAP = {
            ("free", "text"): 1024,
            ("plus", "text"): 1024,
            ("pro", "text"): 1024,
            ("free", "code"): 768,
            ("plus", "code"): 768,
            ("pro", "code"): 768
        }
        key = (self.tier, self.mode)
        dim = VECTOR_DIM_MAP.get(key)
        if not dim:
            raise ValueError(f"No vector dimension defined for ({self.tier}, {self.mode})")
        return dim

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

            # ✅ Create payload indexes one-by-one (works in all 1.7.x)
            index_fields = {
                "session_id": "keyword",
                "mode": "keyword",
                "tier": "keyword",
                "text": "text"
            }
            for field, schema_type in index_fields.items():
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema_type
                    )
                    print(f"✅ Payload index created for '{field}'")
                except Exception as e:
                    if "already exists" in str(e):
                        print(f"ℹ️ Payload index for '{field}' already exists.")
                    else:
                        raise e


    def upsert_chunks(self, chunks: list[str], embeddings: list[list[float]], session_id: str, mode: str):
        payloads = [
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
            for i, chunk in enumerate(chunks)
        ]

        self.client.upsert(collection_name=self.collection_name, points=payloads)
        print(f"✅ Upserted {len(payloads)} chunks to {self.collection_name}")
        print("✅ Chunk Payload:", payloads[0].payload)


    def search(self, query_embedding, top_k=5, filters: dict = None):
        query_filter = None
        if filters:
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
        print(f"🔍 Qdrant search hits:\n{hits}")
        return hits
