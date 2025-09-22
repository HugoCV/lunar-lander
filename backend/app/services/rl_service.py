from threading import Thread, Event
from typing import Optional, List, Tuple
import gymnasium as gym

from app.rl.trainer import run_training
from app.rl.evaluator import evaluate as evaluate_agent
from app.rl.video import record_video as record_video_for_agent
from app.rl.dqn import DQNAgent
from app.core.paths import VIDEOS_DIR
from app.rl import config as C

class RLService:
    def __init__(self):
        self._thread: Optional[Thread] = None
        self._stop_event: Optional[Event] = None
        self.history: List[Tuple[int, float]] = []
        self.last_reward: Optional[float] = None
        self.epsilon: Optional[float] = None
        self.memory_size: int = 0

    def _on_episode_end(self, episode: int, reward: float, epsilon: float):
        self.history.append((episode, reward))
        self.last_reward = reward
        self.epsilon = epsilon

    def _on_step(self, ep, t, r, eps, mem_size):
        self.memory_size = mem_size

    def start(self):
        if self.is_running():
            return
        self.history = []
        self.last_reward = None
        self.epsilon = None
        self.memory_size = 0
        self._stop_event = Event()
        self._thread = Thread(target=run_training, kwargs={
            "on_step": self._on_step,
            "on_episode": self._on_episode_end,
            "stop_event": self._stop_event
        })
        self._thread.start()

    def stop(self):
        if self._stop_event:
            self._stop_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        return {
            "running": self.is_running(),
            "hasAgent": len(self.history) > 0,
            "episodes": len(self.history),
            "lastReward": self.last_reward,
            "epsilon": self.epsilon,
            "memorySize": self.memory_size,
            "historyTail": self.history[-100:]
        }

    def _create_agent_with_weights(self, weights_file: Optional[str]) -> DQNAgent:
        """Helper to create an agent and load specific weights."""
        # We need state and action size from the environment
        env = gym.make(C.ENV_ID)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        env.close()
        
        agent = DQNAgent(state_size, action_size)
        agent.load(weights_file=weights_file)
        return agent

    def evaluate(self, weights_file: Optional[str] = None):
        agent = self._create_agent_with_weights(weights_file)
        mean, scores = evaluate_agent(agent)
        return True, mean, scores

    def record_video(self, weights_file: Optional[str] = None):
        agent = self._create_agent_with_weights(weights_file)
        path = record_video_for_agent(agent, folder=str(VIDEOS_DIR))
        return True, path

# Singleton instance
rl_service = RLService()