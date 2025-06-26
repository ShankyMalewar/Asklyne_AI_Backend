from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.core.embedder import Embedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ Preload embedding models at startup
    print("[Startup] Preloading embedding models...")
    for tier in ["free", "plus", "pro"]:
        Embedder(tier=tier)
    print("[Startup] All embedding models preloaded.")
    yield  # The app runs after this line

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
