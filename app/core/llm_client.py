# app/core/llm_client.py

import os
import httpx
from app.config import Config

class LLMClient:
    def __init__(self, model_name: str):
        self.api_key = Config.TOGETHER_API_KEY
        self.model_name = model_name
        self.api_url = "https://api.together.xyz/inference"


    def build_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def build_payload(self, prompt: str):
        return {
            "model": f"mistralai/{self.model_name}",
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
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["output"]["choices"][0]["text"]


        except Exception as e:
            print("LLM error:")
            import traceback
            traceback.print_exc()
            return "[LLM failed]"


