import numpy as np
import random
import os
import time
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
        """Saves weights to a file with a timestamp."""
        timestamp = int(time.time())
        filename = f"{C.EXPERIMENT_NAME}_{timestamp}.npz"
        weights_path = os.path.join(C.WEIGHTS_DIR, filename)
        print(f"Saving weights to {weights_path}")
        self.model.save_weights(weights_path)

    def load(self, weights_file: str = None):
        """
        Loads model weights. If no file is specified, loads the most recent one.
        """
        if weights_file:
            path = os.path.join(C.WEIGHTS_DIR, weights_file)
        else:
            # Find the latest file
            if not os.path.exists(C.WEIGHTS_DIR) or not os.listdir(C.WEIGHTS_DIR):
                print("No previous weights found. Starting fresh.")
                return
            
            files = [f for f in os.listdir(C.WEIGHTS_DIR) if f.endswith('.npz')]
            if not files:
                print("No .npz files found in weights directory. Starting fresh.")
                return
            
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(C.WEIGHTS_DIR, f)))
            path = os.path.join(C.WEIGHTS_DIR, latest_file)

        if self.model.load_weights(path):
            print(f"Loaded weights from {path}")
        else:
            print(f"Warning: Could not load weights from {path}. Starting fresh.")
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
