import numpy as np
import random
from collections import deque
from app.rl.networks import NeuralNetwork
from app.rl import config as C  # global (gamma, batch_size, etc.)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=C.MEM_CAPACITY)
        self.gamma = C.GAMMA

        self.epsilon = C.EPS_START
        self.epsilon_min = C.EPS_END
        self.epsilon_decay_steps = C.EPS_DECAY_STEPS
        self.total_steps = 0

        self.batch_size = C.BATCH_SIZE

        # networks
        self.model = NeuralNetwork(state_size, 128, action_size, lr=C.LR)
        self.target = NeuralNetwork(state_size, 128, action_size, lr=C.LR)

        self.load() # Attempt to load a saved model

        self.target.hard_copy_from(self.model)

    def save(self):
        """Saves model weights to the default path."""
        print(f"Saving model to {C.MODEL_PATH}")
        self.model.save_weights(C.MODEL_PATH)

    def load(self):
        """Loads model weights from the default path."""
        if self.model.load_weights(C.MODEL_PATH):
            print(f"Loaded model from {C.MODEL_PATH}")
        else:
            print(f"No model found at {C.MODEL_PATH}. Starting fresh.")

    def _eps_now(self):
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return C.EPS_START + (C.EPS_END - C.EPS_START) * frac

    def choose_action(self, state, greedy=False):
        if (not greedy) and (np.random.rand() < self.epsilon):
            return random.randrange(self.action_size)
        q = self.model.forward(state[np.newaxis, :])
        return int(np.argmax(q[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < max(self.batch_size, C.MEM_WARMUP):
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        q_next_online = self.model.forward(next_states)  
        a_star = np.argmax(q_next_online, axis=1)  
        q_next_target = self.target.forward(next_states) 
        target_vals = q_next_target[np.arange(self.batch_size), a_star]

        targets = rewards + self.gamma * target_vals * (1.0 - dones)

        q_curr = self.model.forward(states)
        y_true = np.array(q_curr, copy=True)
        y_true[np.arange(self.batch_size), actions] = targets

        self.model.backward(states, y_true, q_curr)

        # Soft update target
        self.target.soft_update_from(self.model, tau=C.TAU)

    def step_update(self):
        self.total_steps += 1
        self.epsilon = max(self.epsilon_min, self._eps_now())
