import numpy as np
import gymnasium as gym

from app.rl.utils import set_seed, normalize_state
from app.rl import config as C


def evaluate(agent, episodes: int = 3):
    """
    Evalúa el agente en modo greedy (sin exploración) durante N episodios
    y devuelve (mean_reward, lista_de_recompensas_por_ep).

    Params
    ------
    agent      : instancia con .choose_action(state, greedy=True)
    episodes   : número de episodios de evaluación

    Returns
    -------
    (mean_reward: float, scores: list[float])
    """
    env = gym.make(C.ENV_ID, render_mode="rgb_array")
    set_seed(env, C.SEED + 1)

    scores = []
    # Forzar comportamiento greedy durante la evaluación si el agente usa epsilon
    prev_eps = getattr(agent, "epsilon", None)
    if prev_eps is not None:
        agent.epsilon = 0.0

    try:
        for _ in range(episodes):
            s, _ = env.reset()
            s = normalize_state(s)
            total = 0.0

            for _t in range(C.MAX_STEPS):
                # choose_action admite greedy=True en tu implementación,
                # pero si no estuviera, haber seteado epsilon=0.0 ya funciona.
                a = agent.choose_action(s, greedy=True) \
                    if "greedy" in agent.choose_action.__code__.co_varnames else agent.choose_action(s)

                s, r, term, trunc, _ = env.step(a)
                s = normalize_state(s)
                total += r
                if term or trunc:
                    break

            scores.append(float(total))

    finally:
        # Restaurar epsilon (si existía) y cerrar el entorno
        if prev_eps is not None:
            agent.epsilon = prev_eps
        env.close()

    mean_reward = float(np.mean(scores)) if scores else 0.0
    return mean_reward, scores
