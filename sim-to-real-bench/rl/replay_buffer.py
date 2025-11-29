# rl/replay_buffer.py

"""
Episode-level replay buffer for recurrent off-policy RL (RDPG-style).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Episode:
    obs: np.ndarray        # (T+1, obs_dim)
    actions: np.ndarray    # (T, act_dim)
    rewards: np.ndarray    # (T,)
    dones: np.ndarray      # (T,)
    goals: np.ndarray      # (T+1, goal_dim)
    dyn_params: np.ndarray # (T, dyn_dim)


class EpisodeReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage = []
        self.ptr = 0

    def add(self, ep: Episode) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(ep)
        else:
            self.storage[self.ptr] = ep
            self.ptr = (self.ptr + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.storage)

    def sample_batch(self, batch_size: int):
        assert len(self.storage) >= batch_size
        idxs = np.random.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[i] for i in idxs]
