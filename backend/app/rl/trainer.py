import numpy as np
import gymnasium as gym
from threading import Event
from app.rl.dqn import DQNAgent
from app.rl.utils import normalize_state, set_seed

# Config
ENV_ID = "LunarLander-v3"
SEED = 0
EPISODES = 500
MAX_STEPS = 1000
TRAIN_EVERY = 4

def run_training(on_step=None, on_episode=None, stop_event: Event = None):
    """
    Trains a DQN agent on LunarLander-v3.
    Can be stopped externally by setting stop_event.
    """
    env = gym.make(ENV_ID, render_mode="rgb_array")
    set_seed(env, SEED)
    training_started = False

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for ep in range(1, EPISODES + 1):
        if stop_event and stop_event.is_set():
            print("Training loop interrupted before new episode.")
            break
        training_started = True

        state, _ = env.reset()
        state = normalize_state(state)
        total_reward = 0.0

        for t in range(MAX_STEPS):
            if stop_event and stop_event.is_set():
                print("Episode interrupted mid-run.")
                break

            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = normalize_state(next_state)
            rc = np.clip(reward, -1.0, 1.0)

            agent.remember(state, action, rc, next_state, done)
            agent.step_update()

            if agent.total_steps % TRAIN_EVERY == 0:
                agent.replay()

            state = next_state
            total_reward += reward

            if on_step:
                on_step(ep, t, reward, agent.epsilon, len(agent.memory))

            if done:
                break

        if on_episode:
            on_episode(ep, total_reward, agent.epsilon)

    try:
        # Save model if training actually ran and was stopped or completed.
        if training_started:
            agent.save()
    finally:
        env.close()

    return agent
