# backend/app/rl/networks.py

import numpy as np
import os
from app.rl import config as C
from app.rl.utils import huber_grad


class NeuralNetwork:
    """
    Red MLP en NumPy para aproximar Q(s,a).
    Arquitectura: input -> Dense(ReLU, hidden) -> Dense(lineal, output)
    Optimizador: Adam
    Pérdida: Huber sobre Q(s,a) (y_pred vs y_true)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = C.LR):
        # Inicializaciones tipo Kaiming/He para ReLU
        k1 = np.sqrt(2.0 / max(1, input_size))
        k2 = np.sqrt(2.0 / max(1, hidden_size))

        self.W1 = np.random.randn(input_size, hidden_size) * k1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * k2
        self.b2 = np.zeros((1, output_size))

        # Adam
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

        self.m = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }
        self.v = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }

        # Activaciones intermedias (para backward)
        self.z1 = None
        self.a1 = None

    # ---------- Forward ----------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: [B, input_size]
        return: Q-values [B, output_size]
        """
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)  # ReLU
        z2 = self.a1 @ self.W2 + self.b2
        return z2

    # ---------- Backward ----------
    def backward(self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Calcula gradientes con Huber loss y hace un paso de Adam.
        x:      [B, in]
        y_true: [B, out] (targets en la acción elegida)
        y_pred: [B, out] (Q actuales)
        """
        B = x.shape[0]
        # dL/dy_pred
        dy = huber_grad(y_pred, y_true, delta=1.0) / max(1, B)

        # Grad capa 2
        dW2 = self.a1.T @ dy                              # [hidden, out]
        db2 = np.sum(dy, axis=0, keepdims=True)           # [1, out]

        # Backprop a capa oculta
        dh = dy @ self.W2.T                               # [B, hidden]
        dh[self.z1 <= 0.0] = 0.0                          # ReLU'

        dW1 = x.T @ dh                                    # [in, hidden]
        db1 = np.sum(dh, axis=0, keepdims=True)           # [1, hidden]

        # Clipping por norma global (evita explosiones)
        dW1, db1, dW2, db2 = self._clip_norm(dW1, db1, dW2, db2, max_norm=C.GRAD_CLIP_NORM)

        # Paso Adam
        self.t += 1
        self._adam_update("W1", dW1)
        self._adam_update("b1", db1)
        self._adam_update("W2", dW2)
        self._adam_update("b2", db2)

    # ---------- Utilidades de optimización ----------
    def _clip_norm(self, *grads: np.ndarray, max_norm: float):
        """
        Clipping por norma L2 global de todos los gradientes juntos.
        max_norm se pasa como keyword-only (obligatorio).
        """
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(np.sum(g * g))
        norm = np.sqrt(total)
        if norm > max_norm > 0.0:
            scale = max_norm / (norm + 1e-8)
            return [g * scale for g in grads]
        return list(grads)

    def _adam_update(self, name: str, grad: np.ndarray) -> None:
        m = self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
        v = self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)

        m_hat = m / (1.0 - self.beta1 ** self.t)
        v_hat = v / (1.0 - self.beta2 ** self.t)

        param = getattr(self, name)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        setattr(self, name, param)

    # ---------- Target updates ----------
    def soft_update_from(self, other: "NeuralNetwork", tau: float = C.TAU) -> None:
        """
        θ_target ← τ * θ_online + (1 - τ) * θ_target
        """
        for n in ("W1", "b1", "W2", "b2"):
            tgt = getattr(self, n)
            src = getattr(other, n)
            setattr(self, n, tau * src + (1.0 - tau) * tgt)

    def hard_copy_from(self, other: "NeuralNetwork") -> None:
        for n in ("W1", "b1", "W2", "b2"):
            setattr(self, n, np.copy(getattr(other, n)))

    # ---------- Persistence ----------
    def save_weights(self, path: str) -> None:
        """Saves model weights to a .npz file."""
        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        np.savez_compressed(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2
        )

    def load_weights(self, path: str) -> bool:
        """
        Loads model weights from a .npz file.
        Returns True on success, False otherwise.
        """
        if not os.path.exists(path):
            return False
        
        data = np.load(path)
        for param_name in ("W1", "b1", "W2", "b2"):
            setattr(self, param_name, data[param_name])
        
        return True
