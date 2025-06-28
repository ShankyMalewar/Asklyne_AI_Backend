from fastapi import APIRouter, File, UploadFile, Form , Query
from typing import Literal
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdfplumber
import io
import traceback
from app.services.qdrant_service import QdrantService
from app.services.typesense_service import TypesenseService
from PIL import Image
import pytesseract

router = APIRouter()

@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@router.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    mode: str = Form(...),
    tier: str = Form(...),
):
    from app.core.chunker import Chunker
    from app.core.embedder import Embedder
    from app.services.qdrant_service import QdrantService
    from app.services.typesense_service import TypesenseService

    # Read file content
    content = await file.read()
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if mode == "notes":
        try:
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)
            
        except Exception as e:
            return JSONResponse(content={"error": f"OCR failed: {str(e)}"}, status_code=400)

    elif file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = content.decode("utf-8", errors="ignore")

    # Chunking setup
    max_tokens = 480
    overlap = 80
    chunker = Chunker(file_type=mode, max_tokens=max_tokens, overlap=overlap)
    chunks = chunker.chunk(text)

    # Embedding
    embedder = Embedder(tier=tier)
    embeddings = embedder.embed_chunks(chunks)

    # Store in Qdrant
    qdrant = QdrantService(tier=tier)
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
    tier: Literal["free", "plus", "pro"]

@router.post("/query")
async def handle_query(request: QueryRequest):
    from app.core.retriever import Retriever
    from app.core.reranker import Reranker
    from app.core.context_builder import ContextBuilder
    from app.core.llm_client import LLMClient

    try:
        # Step 1: Retrieve chunks
        retriever = Retriever(tier=request.tier)
        chunks = retriever.retrieve(
            query=request.query,
            session_id=request.session_id,
            mode=request.mode
        )

        # Step 2: Rerank if Plus/Pro
        reranker = Reranker(tier=request.tier)
        if not chunks:
            return {"error": "No relevant chunks found for this query."}

        ranked_chunks = reranker.rerank(request.query, chunks)

        # Step 3: Build context
        builder = ContextBuilder(tier=request.tier)
        context = builder.build(ranked_chunks)

        # Step 4: Query LLM
        client = LLMClient.from_tier(request.tier)
        full_prompt = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"
        response = await client.query(full_prompt)

        return {"response": response, "context_used": context}

    except Exception as e:
        tb = traceback.format_exc()
        print("🔥 Exception Traceback:\n", tb)
        return JSONResponse(content={"error": str(e), "trace": tb}, status_code=500)


@router.post("/test-llm")
async def test_llm(tier: str = Query("free", enum=["free", "plus", "pro"])):
    from app.core.llm_client import LLMClient
    try:
        client = LLMClient.from_tier(tier)
        response = await client.query("Explain gravity in simple words.")
        return {"tier": tier, "response": response}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
