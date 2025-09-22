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
# Path for the single model weights file
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "lunar_lander_dqn.npz")
