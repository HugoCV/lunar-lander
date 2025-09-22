# app/rl/config.py

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
