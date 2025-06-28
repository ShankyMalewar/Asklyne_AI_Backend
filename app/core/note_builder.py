import os
import json
from app.core.llm_client import LLMClient

def load_interactions(session_id: str) -> list[dict]:
    log_path = f"sessions/{session_id}/interaction_log.json"
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as f:
        return json.load(f)

def format_qa_log(interactions: list[dict]) -> str:
    return "\n\n".join(
        f"Q{i+1}: {entry['query']}\nA{i+1}: {entry['response']}"
        for i, entry in enumerate(interactions)
    )

def build_prompt(mode: str, qa_log: str, custom_prompt: str | None = None) -> str:
    base = """You are an AI assistant that generates high-quality study notes from a Q&A session.

Format the notes with:
- Headings
- Bullet points
- Clean, grouped topics
- No repetition

Here is the session:
"""
    if custom_prompt:
        base += f"\nUser Instructions:\n{custom_prompt}\n"
    base += f"\n{qa_log}"
    return base

async def generate_notes(session_id: str, tier: str, mode: str, prompt_type: str, custom_prompt: str = "") -> str:
    interactions = load_interactions(session_id)
    if not interactions:
        return "[Error: No conversation found for this session.]"

    if prompt_type == "response_only":
        qa_log = "\n\n".join(f"A{i+1}: {entry['response']}" for i, entry in enumerate(interactions))
    else:
        qa_log = format_qa_log(interactions)

    prompt = build_prompt(mode, qa_log, custom_prompt if prompt_type == "custom" else None)

    llm = LLMClient(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    return await llm.query(prompt)
