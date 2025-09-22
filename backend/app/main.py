from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

from app.api.v1.endpoints import health, train
from app.core.paths import VIDEOS_DIR, WEIGHTS_DIR
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="Lunar Lander RL API",
    version="1.0.0",
    description="API para entrenar, evaluar y grabar videos de un agente RL en LunarLander-v3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")

app.include_router(health.router, prefix="/api/v1")

# This endpoint should ideally be in `train.py`, but adding it here for simplicity
@app.get("/api/v1/train/weights", response_model=List[str], tags=["train"])
async def list_weights():
    """Lists all available model weight files, sorted by most recent first."""
    if not os.path.exists(WEIGHTS_DIR):
        return []
    files = sorted([f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.npz')], key=lambda f: os.path.getmtime(os.path.join(WEIGHTS_DIR, f)), reverse=True)
    return files

app.include_router(train.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI server iniciado.")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 FastAPI server detenido.")
