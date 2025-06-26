from typing import List
from sentence_transformers import SentenceTransformer

_loaded_models = {}

TIER_EMBED_MODEL = {
    "free": "BAAI/bge-small-en-v1.5",
    "plus": "BAAI/bge-large-en-v1.5",
    "pro": "intfloat/multilingual-e5-large"
    # Optional: add 'code' support later (e.g. microsoft/codebert-base)
}
class Embedder:
    def __init__(self, tier: str = "free"):
        from sentence_transformers import SentenceTransformer

        model_name = TIER_EMBED_MODEL.get(tier.lower())
        if not model_name:
            raise ValueError(f"Unsupported tier: {tier}")

        if model_name in _loaded_models:
            self.model = _loaded_models[model_name]
        else:
            print(f"[Embedder] Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            _loaded_models[model_name] = model
            self.model = model

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        return self.model.encode(chunks, convert_to_numpy=True).tolist()