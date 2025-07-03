from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from app.api.routes import router as api_router
from app.core.embedder import Embedder

# Read allowed tiers from environment variable
TIERS_TO_LOAD = os.getenv("ASKLYNE_TIERS", "free").split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Startup] Preloading embedding models for: {TIERS_TO_LOAD}")

    for tier in TIERS_TO_LOAD:
        for mode in ["text", "code"]:
            try:
                print(f"▶ Preloading embedder: {tier}/{mode}")
                Embedder(tier=tier, mode=mode)
                print(f"✅ Done: {tier}/{mode}")
            except Exception as e:
                print(f"⚠️ Failed to preload embedder for {tier}/{mode}: {e}")

    print("[Startup] Embedding model loading complete.")
    yield

app = FastAPI(
    title="Asklyne API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Asklyne API is live 🚀"}
