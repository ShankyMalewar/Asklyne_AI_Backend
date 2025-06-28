from typing import List
from sentence_transformers import SentenceTransformer

_loaded_models = {}

TIER_MODE_MODEL_MAP = {
    "free": {
        "text": "BAAI/bge-large-en-v1.5",
        "code": "microsoft/codebert-base"
    },
    "plus": {
        "text": "BAAI/bge-large-en-v1.5",
        "code": "Salesforce/codet5-base"
    },
    "pro": {
        "text": "intfloat/multilingual-e5-large",
        "code": "microsoft/graphcodebert-base"
    }
}

class Embedder:
    def __init__(self, tier: str = "free", mode: str = "text"):
        self.tier = tier.lower()
        self.mode = mode.lower()

        model_name = TIER_MODE_MODEL_MAP.get(self.tier, {}).get(self.mode)
        if not model_name:
            raise ValueError(f"No model defined for tier '{self.tier}' and mode '{self.mode}'")

        if model_name in _loaded_models:
            self.model = _loaded_models[model_name]
        else:
            print(f"[Embedder] Loading model for {self.tier}/{self.mode}: {model_name}")
            model = SentenceTransformer(model_name)
            _loaded_models[model_name] = model
            self.model = model

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        return self.model.encode(chunks, convert_to_numpy=True).tolist()
