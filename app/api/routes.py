from fastapi import APIRouter, File, UploadFile, Form , Query
from typing import Optional, Literal
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdfplumber
import io
from PIL import ImageEnhance, ImageFilter
import numpy as np
import os
import traceback
from app.services.qdrant_service import QdrantService
from app.services.typesense_service import TypesenseService
from PIL import Image
import pytesseract
from app.utils.code_parser import extract_code_from_py, extract_code_from_ipynb
import json
from datetime import datetime
from app.core.note_builder import generate_notes as build_notes
from fastapi.responses import FileResponse
from app.core.note_builder import save_notes_as_pdf
from app.core.chunker import Chunker
from app.core.embedder import Embedder
from app.core.retriever import Retriever
from app.core.reranker import Reranker
from app.core.context_builder import ContextBuilder
from app.core.llm_client import LLMClient
import time


router = APIRouter()

@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@router.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    mode: str = Form(...),
):
    tier = os.getenv("ASKLYNE_TIERS", "free").lower()
    # Read file content
    content = await file.read()
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    original_mode = mode

    if original_mode == "notes":
        try:
            def preprocess_ocr_image(image: Image.Image) -> Image.Image:
                image = image.convert("L")  # grayscale
                image = image.resize((1200, int(image.height * (1200 / image.width))))  # upscale
                image = image.filter(ImageFilter.MedianFilter(size=3))  # denoise
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)  # boost contrast
                return image

            image = Image.open(io.BytesIO(content))
            image = preprocess_ocr_image(image)
            text = pytesseract.image_to_string(image, config="--psm 6")
        except Exception as e:
            return JSONResponse(content={"error": f"OCR failed: {str(e)}"}, status_code=400)


    elif mode == "code":
        if file.filename.endswith(".py"):
            text = extract_code_from_py(content)
        elif file.filename.endswith(".ipynb"):
            text = extract_code_from_ipynb(content)
        else:
            return JSONResponse(content={"error": "Unsupported code file type"}, status_code=400)

    elif file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    else:
        text = content.decode("utf-8", errors="ignore")
    
    mode = "text" if original_mode == "notes" else mode

    # Chunking setup
    max_tokens = 480
    overlap = 80
    chunker = Chunker(file_type=mode, max_tokens=max_tokens, overlap=overlap)
    chunks = chunker.chunk(text)

    # Embedding
    embedder = Embedder(tier=tier, mode=mode)
    embeddings = embedder.embed_chunks(chunks)

    # Store in Qdrant
    qdrant = QdrantService(tier=tier, mode=mode)

    qdrant.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        session_id=session_id,
        mode=mode
    )

    # Store in Typesense
    typesense = TypesenseService(tier=tier)
    typesense.upsert_chunks(
        chunks=chunks,
        session_id=session_id,
        mode=mode
    )

    return {
        "session_id": session_id,
        "tier": tier,
        "mode": mode,
        "num_chunks": len(chunks),
        "sample_chunk": chunks[0] if chunks else "[empty]",
        "sample_embedding": embeddings[0][:5] if embeddings else []
    }


class QueryRequest(BaseModel):
    session_id: str
    query: str
    mode: Literal["text", "notes", "code"]

@router.post("/query")
async def handle_query(request: QueryRequest):
    
    start = time.time()

    tier = os.getenv("ASKLYNE_TIERS", "free").lower()
    def save_interaction(session_id: str, query: str, response: str):
        path = f"sessions/{session_id}/interaction_log.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append({
            "timestamp": str(datetime.utcnow()),
            "query": query,
            "response": response
        })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    try:
        mode = "text" if request.mode == "notes" else request.mode

        retriever = Retriever(tier=tier,mode=mode)

        chunks = await retriever.retrieve_async(
            query=request.query,
            session_id=request.session_id,
            mode=mode
        )
        print("⏱️ Retrieval Time:", round(time.time() - start, 2))

        
        reranker = Reranker(tier=tier)
        if not chunks:
            return {"error": "No relevant chunks found for this query."}

        t1 = time.time()
        ranked_chunks = reranker.rerank(request.query, chunks[:5])
        print("⏱️ Rerank Time:", round(time.time() - t1, 2))

        builder = ContextBuilder(tier=tier)
        context = builder.build(ranked_chunks)

        client = LLMClient.from_tier(tier, mode)


        if request.mode == "code":
            full_prompt = f"""You are a helpful AI coding assistant.

            You will be given source code and a user request. Based on the request, you may:
            - Modify the code
            - Add new functionality
            - Explain parts of the code
            - Fix bugs or improve performance

            Code Context:
            {context}

            User Request:
            {request.query}

            Respond with either a modified version of the code or a helpful explanation.
            If you update the code, include the full version of the updated code.
            """
        else:
            SYSTEM_PROMPT = """You are a highly intelligent and helpful AI assistant. 
            You will be given context from a user-uploaded document or notes, along with a user query.
            Session Metadata:
            - Tier: {request.tier.capitalize()}
            - Mode: {request.mode.capitalize()}
            Your job is to:
            - Answer clearly, accurately, and in a user-friendly way
            - Use bullet points, headings, or paragraphs depending on the question
            - Expand and explain if context is short or vague
            - Only use the given context — do NOT make up facts
            - If the context is not sufficient, respond: "Based on the provided context, I cannot answer confidently."
            
            If mode = 'notes' → Expect imperfect OCR and try to infer meaning  
            If mode = 'code'  → Use technical reasoning and full examples where possible
            """

            full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser Question:\n{request.query}\n\nAnswer:"


        t2 = time.time()
        response = await client.query(full_prompt)
        print("⏱️ LLM Time:", round(time.time() - t2, 2))
        print("⚡ Total Time:", round(time.time() - start, 2))
        
        
        save_interaction(request.session_id, request.query, response)

        return {
            "response": response,
            "context_used": context,
            "model_used": client.model_name  # ✅ shows user which LLM generated the answer
        }


    except Exception as e:
        tb = traceback.format_exc()
        print(" Exception Traceback:\n", tb)
        return JSONResponse(content={"error": str(e), "trace": tb}, status_code=500)



@router.post("/generate-notes")
async def generate_notes_route(
    session_id: str = Form(...),
    mode: str = Form(...),
    prompt_type: Literal["full", "response_only", "custom"] = Form(...),
    custom_prompt: Optional[str] = Form(None),
    as_pdf: bool = Form(False)
):
    tier = os.getenv("ASKLYNE_TIERS", "free").lower()
    mode = "text" if mode == "notes" else mode

    notes = await build_notes(
        session_id=session_id,
        tier=tier,
        mode=mode,
        prompt_type=prompt_type,
        custom_prompt=custom_prompt or ""
    )

    if as_pdf:
        pdf_path = save_notes_as_pdf(notes, session_id)
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=f"asklyne_notes_{session_id}.pdf"
        )
    else:
        return {"notes": notes}
