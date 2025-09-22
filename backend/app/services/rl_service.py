from typing import Optional, List, Dict, Any, Tuple
from threading import Event, Thread, Lock
import time

from app.rl.trainer import run_training
from app.rl.evaluator import evaluate as eval_agent
from app.rl.video import record_video as record_video_fn

_state: Dict[str, Any] = {
    "running": False,
    "hasAgent": False,
    "thread": None,
    "stop": Event(),
    "agent": None,
    "episodes": 0,
    "last_reward": None,
    "epsilon": None,
    "mem": 0,
    "history": [],
}
_lock = Lock()

def _on_step(ep: int, t: int, r: float, eps: float, mem: int):
    with _lock:
        _state["epsilon"] = eps
        _state["mem"] = mem

def _on_episode(ep: int, total: float, eps: float):
    with _lock:
        _state["episodes"] = ep
        _state["last_reward"] = total
        _state["epsilon"] = eps
        _state["history"].append((ep, total))
        if len(_state["history"]) > 5000:
            _state["history"] = _state["history"][-2000:]

def _trainer_loop():
    try:
        agent = run_training(
            on_step=_on_step,
            on_episode=_on_episode,
            stop_event=_state["stop"],
        )
        with _lock:
            _state["agent"] = agent
            _state["hasAgent"] = True
    finally:
        with _lock:
            _state["running"] = False

def start():
    with _lock:
        if _state["running"]:
            return
        _state["stop"].clear()
        _state["running"] = True
        _state["history"].clear()
        _state["episodes"] = 0
        _state["last_reward"] = None
        _state["epsilon"] = None
        _state["mem"] = 0
        th = Thread(target=_trainer_loop, daemon=True)
        _state["thread"] = th
        th.start()

def stop():
    with _lock:
        if not _state["running"]:
            return
        _state["stop"].set()

def status() -> Dict[str, Any]:
    with _lock:
        return {
            "running": _state["running"],
            "hasAgent": _state["hasAgent"],
            "episodes": _state["episodes"],
            "lastReward": _state["last_reward"],
            "epsilon": _state["epsilon"],
            "memorySize": _state["mem"],
            "historyTail": _state["history"][-50:],
        }

def evaluate() -> Tuple[bool, float, List[float]]:
    with _lock:
        agent = _state["agent"]
    if agent is None:
        return False, 0.0, []
    mean, scores = eval_agent(agent, episodes=3)
    return True, mean, scores

def record_video() -> Tuple[bool, Optional[str]]:
    with _lock:
        agent = _state["agent"]
    if agent is None:
        return False, None
    web_path = record_video_fn(agent, folder="./videos", episodes=1)
    return (web_path is not None), web_path
