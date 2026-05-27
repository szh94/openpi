"""Replay buffer for RLT online RL training.

Standard circular buffer for RL transitions.
Each entry: (state [2080], action [320], reward [1], next_state [2080], done [1])
Supports uniform random sampling and stride=2 for overlapping storage.
"""

import numpy as np


class ReplayBuffer:
    """Circular replay buffer for RL transitions."""

    def __init__(
        self,
        capacity: int = 1_000_000,
        state_dim: int = 2080,
        action_dim: int = 320,
        stride: int = 2,
    ):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            state_dim: Dimension of the state vector.
            action_dim: Dimension of the action chunk.
            stride: Storage stride for overlapping transitions (2 = 5x sample efficiency).
        """
        self.capacity = capacity
        self.stride = stride
        self._ptr = 0
        self._size = 0

        # Pre-allocate numpy arrays.
        self._state = np.zeros((capacity, state_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((capacity, 1), dtype=np.float32)
        self._next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self._done = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a single transition to the buffer.

        With stride=2, only stores every `stride`-th transition.
        """
        if self._ptr % self.stride == 0:
            idx = self._ptr // self.stride % self.capacity
            self._state[idx] = state
            self._action[idx] = action
            self._reward[idx] = reward
            self._next_state[idx] = next_state
            self._done[idx] = done
            self._size = min(self._size + 1, self.capacity)
        self._ptr += 1

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            dict with keys: state, action, reward, next_state, done.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "state": self._state[indices],
            "action": self._action[indices],
            "reward": self._reward[indices],
            "next_state": self._next_state[indices],
            "done": self._done[indices],
        }

    @property
    def size(self) -> int:
        """Current number of transitions in the buffer."""
        return self._size

    @property
    def is_ready(self) -> bool:
        """Check if the buffer has enough data for training."""
        return self._size >= 1024

    def __len__(self) -> int:
        return self._size
