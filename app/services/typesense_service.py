# app/services/typesense_service.py

import typesense
from app.config import Config

class TypesenseService:
    def __init__(self, tier: str):
        self.tier = tier.lower()
        self.collection_name = f"asklyne_chunks_{self.tier}"

        self.client = typesense.Client({
            'nodes': [{
                'host': Config.TYPESENSE_HOST.replace("http://", "").replace("https://", ""),
                'port': 8108,
                'protocol': 'http'
            }],
            'api_key': Config.TYPESENSE_API_KEY,
            'connection_timeout_seconds': 2
        })

        self.ensure_collection_exists()

    def ensure_collection_exists(self):
        try:
            self.client.collections[self.collection_name].retrieve()
            print(f"\u2139\ufe0f Typesense collection '{self.collection_name}' already exists.")
        except Exception:
            schema = {
                "name": self.collection_name,
                "fields": [
                    {"name": "id", "type": "string"},
                    {"name": "text", "type": "string"},
                    {"name": "mode", "type": "string", "facet": True},
                    {"name": "tier", "type": "string", "facet": True},
                    {"name": "session_id", "type": "string", "facet": True}
                ],
                "default_sorting_field": "id"
            }
            self.client.collections.create(schema)
            print(f"✅ Created Typesense collection '{self.collection_name}'")

    def upsert_chunks(self, chunks: list[str], session_id: str, mode: str):
        documents = [
            {
                "id": f"{session_id}_{i}",
                "text": chunk,
                "mode": mode,
                "tier": self.tier,
                "session_id": session_id
            }
            for i, chunk in enumerate(chunks)
        ]
        self.client.collections[self.collection_name].documents.import_(documents, {'action': 'upsert'})
        print(f"✅ Upserted {len(documents)} chunks to Typesense")

    def search(self, query: str, top_k: int = 5, filters: dict = None):
        filter_by = " && ".join([f"{k}:={v}" for k, v in filters.items()]) if filters else ""
        results = self.client.collections[self.collection_name].documents.search({
            'q': query,
            'query_by': 'text',
            'filter_by': filter_by,
            'per_page': top_k
        })
        return results['hits']
