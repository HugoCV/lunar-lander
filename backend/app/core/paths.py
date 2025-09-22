from pathlib import Path

# backend/app/core/paths.py  -> VIDEOS_DIR = <repo>/backend/videos
VIDEOS_DIR = Path(__file__).resolve().parents[2] / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)