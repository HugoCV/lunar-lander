from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import health, train
from app.core.paths import VIDEOS_DIR
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
app.include_router(train.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI server iniciado.")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 FastAPI server detenido.")
