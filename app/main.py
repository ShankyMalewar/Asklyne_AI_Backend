from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.core.embedder import Embedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Preloading embedding models...")

    tiers = ["free", "plus", "pro"]
    for tier in tiers:
        print(f"▶ Preloading text embedder for {tier}")
        Embedder(tier=tier, mode="text")
        print(f"✅ Done text for {tier}")

        print(f"▶ Preloading code embedder for {tier}")
        Embedder(tier=tier, mode="code")
        print(f"✅ Done code for {tier}")

    print("[Startup] All embedding models preloaded.")
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
