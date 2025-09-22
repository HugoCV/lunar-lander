from fastapi import APIRouter
from app.services import rl_service
from app.core.paths import VIDEOS_DIR
from pathlib import Path
import os
from typing import List, Dict

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
def evaluate():
    ok, mean, scores = rl_service.evaluate()
    return {"ok": ok, "mean": mean, "scores": scores}

@router.post("/train/video")
def record_video():
    ok, path = rl_service.record_video()
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
