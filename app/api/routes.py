from fastapi import APIRouter, File, UploadFile, Form , Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdfplumber
import io

router = APIRouter()

@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@router.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    mode: str = Form(...),   # "text" or "code"
    tier: str = Form(...),   # Still passed, for logging or future use
):
    from app.core.chunker import Chunker

    # Read file content
    content = await file.read()
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = content.decode("utf-8", errors="ignore")

    max_tokens = 480
    overlap = 80

    # Init chunker and chunk
    chunker = Chunker(file_type=mode, max_tokens=max_tokens, overlap=overlap)
    chunks = chunker.chunk(text)
    
    from app.core.embedder import Embedder
    
    # Init embedder and embed chunks
    embedder = Embedder(tier=tier)
    embeddings = embedder.embed_chunks(chunks)


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
    mode: str
    tier: str

@router.post("/query")
async def handle_query(request: QueryRequest):
    # TODO: retrieve → rerank → build prompt → LLM
    return {"response": f"Mock answer to: {request.query}"}


@router.post("/test-llm")
async def test_llm(tier: str = Query("free", enum=["free", "plus", "pro"])):
    from app.core.llm_client import LLMClient
    try:
        client = LLMClient.from_tier(tier)
        response = await client.query("Explain gravity in simple words.")
        return {"tier": tier, "response": response}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)