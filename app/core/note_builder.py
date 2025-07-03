import os
import json
import markdown2
import pdfkit
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
    base = """You are an intelligent AI assistant trained to generate high-quality notes from user conversations.

    Given the following interaction history between the user and the AI, generate structured notes that help the user **review and understand** the material.

    Your output should:
    - Adapt to the domain (e.g., academic, legal, technical, general)
    - Use meaningful headings
    - Highlight important insights, definitions, rules, examples
    - Organize content logically (bullet points, paragraphs, or mixed)
    - Be concise but clear
    - Avoid repeating raw Q&A format

    If the content lacks context, expand it intelligently. Always optimize for learning and clarity.
    """

    if custom_prompt:
        base += f"\nUser Instructions:\n{custom_prompt}\n"
    base += f"\nSession:\n{qa_log}"
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

    # Use Mistral for now for better speed and context balance
    if tier in ["plus", "pro"]:
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    else:
        model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

    llm = LLMClient(model_name=model_name)

    return await llm.query(prompt)

def save_notes_as_pdf(md_content: str, session_id: str) -> str:
    html = markdown2.markdown(md_content)

    styled_html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                line-height: 1.7;
                padding: 40px;
                background-color: #ffffff;
            }}
            h1, h2, h3 {{
                color: #1a237e;
            }}
            ul {{
                margin-bottom: 1em;
            }}
            li {{
                margin-bottom: 0.4em;
            }}
        </style>
    </head>
    <body>{html}</body>
    </html>
    """

    pdf_path = f"sessions/{session_id}/asklyne_notes.pdf"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    pdfkit.from_string(styled_html, pdf_path)
    return pdf_path
