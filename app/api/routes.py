from fastapi import APIRouter, File, UploadFile, Form , Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@router.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    mode: str = Form(...),
    tier: str = Form(...)
):
    # TODO: chunk → embed → index file
    return {"message": f"Received file {file.filename} for session {session_id}"}


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