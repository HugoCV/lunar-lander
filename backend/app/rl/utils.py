# backend/app/rl/utils.py

import numpy as np
import random


def set_seed(env, seed: int = 0):
    """
    Fija la semilla para reproducibilidad.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        # Algunos entornos no soportan seed explícita
        pass


def normalize_state(s: np.ndarray) -> np.ndarray:
    """
    Normalización rápida y robusta de los estados.
    LunarLander ya está acotado [-1, 1] en gran parte,
    pero esto ayuda a estabilizar.
    """
    return np.tanh(s)


def huber_grad(y_pred: np.ndarray, y_true: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Calcula el gradiente de la pérdida Huber respecto a y_pred.
    L(e) = 0.5*e^2        si |e| <= delta
         = delta*(|e|-0.5*delta) si |e| > delta
    Retorna dL/dy_pred
    """
    e = y_pred - y_true
    mask = np.abs(e) <= delta
    grad = np.where(mask, e, delta * np.sign(e))
    return grad
