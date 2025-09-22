from pathlib import Path

# backend/app/core/paths.py  -> VIDEOS_DIR = <repo>/backend/videos
REPO_ROOT = Path(__file__).resolve().parents[2]

VIDEOS_DIR = REPO_ROOT / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_DIR = REPO_ROOT / "data" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)