from fastapi import APIRouter
from app.services.rl_service import rl_service
from app.core.paths import VIDEOS_DIR
from pathlib import Path
import os
from typing import List, Dict, Optional
from pydantic import BaseModel

class EvalPayload(BaseModel):
    weights_file: Optional[str] = None


router = APIRouter()

@router.post("/train/start")
def start_training():
    rl_service.start()
    return {"started": True}

@router.post("/train/stop")
def stop_training():
    rl_service.stop()
    return {"stopped": True}

@router.get("/train/status")
def training_status():
    return rl_service.status()

@router.post("/train/evaluate")
def evaluate(payload: EvalPayload):
    ok, mean, scores = rl_service.evaluate(weights_file=payload.weights_file)
    return {"ok": ok, "mean": mean, "scores": scores}

@router.post("/train/video")
def record_video(payload: EvalPayload):
    ok, path = rl_service.record_video(weights_file=payload.weights_file)
    return {"ok": ok, "path": path}

@router.get("/train/videos")
def list_videos():
    items: List[Dict] = []
    for p in sorted(VIDEOS_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = p.stat()
        items.append({
            "name": p.name,
            "url": f"/videos/{p.name}",
            "sizeBytes": stat.st_size,
            "modified": stat.st_mtime,
        })
    return {"videos": items}
