# app/core/llm_client.py

import os
import httpx
from app.config import Config

# 🔁 Finalized tier-to-model mapping
TIER_MODEL_MAP = {
    "free": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "plus": "mistralai/Mistral-7B-Instruct-v0.3",
    "pro": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

class LLMClient:
    def __init__(self, model_name: str):
        self.api_key = Config.TOGETHER_API_KEY
        self.model_name = model_name
        self.api_url = "https://api.together.xyz/inference"

    @classmethod
    def from_tier(cls, tier: str):
        model_name = TIER_MODEL_MAP.get(tier.lower())
        if not model_name:
            raise ValueError(f"LLM not supported for tier: {tier}")
        return cls(model_name)

    def build_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def build_payload(self, prompt: str):
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        }

    async def query(self, prompt: str) -> str:
        payload = self.build_payload(prompt)
        headers = self.build_headers()

        try:
            timeout = httpx.Timeout(30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                print("LLM response data:", data)

                # Try Together's newer format first
                if "output" in data:
                    return data["output"]["choices"][0]["text"]
                # Fallback to OpenAI-style format
                elif "choices" in data and "message" in data["choices"][0]:
                    return data["choices"][0]["message"]["content"]
                else:
                    return "[LLM returned unexpected format]"

        except Exception as e:
            print("LLM error:")
            import traceback
            traceback.print_exc()
            return "[LLM failed]"
