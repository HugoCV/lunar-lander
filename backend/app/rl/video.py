import os
from typing import Optional
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from app.rl.utils import set_seed, normalize_state
from app.rl import config as C

def record_video(agent, folder: str = "./videos", episodes: int = 1, max_steps: int = None) -> Optional[str]:
    os.makedirs(folder, exist_ok=True)
    max_steps = max_steps or C.MAX_STEPS

    env = RecordVideo(
        gym.make(C.ENV_ID, render_mode="rgb_array"),
        video_folder=folder,
        episode_trigger=lambda ep: True
    )
    set_seed(env, C.SEED + 2)

    prev_eps = getattr(agent, "epsilon", None)
    if prev_eps is not None:
        agent.epsilon = 0.0

    try:
        for _ in range(episodes):
            s, _ = env.reset()
            s = normalize_state(s)
            steps = 0
            done = False
            while not done and steps < max_steps:
                a = agent.choose_action(s, greedy=True) if "greedy" in agent.choose_action.__code__.co_varnames else agent.choose_action(s)
                s, r, term, trunc, _ = env.step(a)
                s = normalize_state(s)
                done = term or trunc
                steps += 1
    finally:
        if prev_eps is not None:
            agent.epsilon = prev_eps
        env.close()

    mp4s = sorted([f for f in os.listdir(folder) if f.lower().endswith(".mp4")])
    if not mp4s:
        return None

    last_name = mp4s[-1]
    return f"/videos/{last_name}"
