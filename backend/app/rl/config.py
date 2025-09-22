# backend/app/rl/config.py
import os

# Hiperparámetros globales
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
MEM_CAPACITY = 100_000
MEM_WARMUP = 5_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100_000

TAU = 0.005
GRAD_CLIP_NORM = 1.0   # <--- este faltaba
MAX_STEPS = 1000       # lo usa trainer/evaluator
ENV_ID = "LunarLander-v3"
SEED = 0
EPISODES = 2000
# --- Persistence ---
from app.core.paths import WEIGHTS_DIR
# Default name for the experiment run
EXPERIMENT_NAME = "lunar_lander_dqn"
