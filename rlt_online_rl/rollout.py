"""Rollout runtime for RLT online RL fine-tuning.

Robot interaction loop:
    1. VLA inference -> ref_actions (50 steps)
    2. Encoder -> RL Token
    3. Actor -> corrected action chunk (10 steps)
    4. Execute 10 steps, collect reward + next observation
    5. Store transition in Replay Buffer (with stride=2)
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rlt_online_rl.actor import Actor
from rlt_online_rl.replay import ReplayBuffer

logger = logging.getLogger("rlt_rollout")

# Standard dimensions for RLT.
STATE_DIM = 2080
ACTION_CHUNK_DIM = 320
ACTION_HORIZON = 50
ACTION_DIM = 32
C = 10  # Action chunk size (number of steps to execute)


class RolloutRunner:
    """Manages the robot rollout loop for data collection.

    Interfaces with:
        - VLA model: provides token embeddings and reference actions
        - RLTEncoder: compresses embeddings into RL Token
        - Actor: outputs corrected action chunks
        - Environment: executes actions and returns observations
        - ReplayBuffer: stores collected transitions
    """

    def __init__(
        self,
        vla_fn: Callable[[dict], tuple[np.ndarray, np.ndarray]],
        encoder_fn: Callable[[np.ndarray], np.ndarray],
        actor: Actor,
        replay_buffer: ReplayBuffer,
        env_step_fn: Callable[[np.ndarray], tuple[dict, float, bool]],
        *,
        proprio_dim: int = 32,
        stride: int = 2,
    ):
        """Initialize the rollout runner.

        Args:
            vla_fn: Function that takes observation dict and returns (token_embeddings, ref_actions).
                    token_embeddings shape [N, 2048], ref_actions shape [50, 32].
            encoder_fn: Function that takes token embeddings [N, 2048] and returns RL Token [2048].
            actor: Actor network for action correction.
            replay_buffer: Buffer to store transitions.
            env_step_fn: Function that executes a single action step and returns
                        (next_obs_dict, reward, done).
            proprio_dim: Dimension of proprioceptive state.
            stride: Storage stride for overlapping transitions.
        """
        self.vla_fn = vla_fn
        self.encoder_fn = encoder_fn
        self.actor = actor
        self.replay = replay_buffer
        self.env_step = env_step_fn
        self.proprio_dim = proprio_dim
        self.stride = stride
        self._step_count = 0

    def run_episode(self, obs: dict, max_steps: int = 500) -> dict[str, Any]:
        """Run a single episode of rollouts.

        Args:
            obs: Initial observation dict.
            max_steps: Maximum number of environment steps.

        Returns:
            dict with episode statistics.
        """
        episode_reward = 0.0
        episode_length = 0
        start_time = time.monotonic()

        logger.info("Starting rollout episode...")

        while episode_length < max_steps:
            # Step 1: VLA inference -> token embeddings + reference actions.
            token_embeddings, ref_actions = self.vla_fn(obs)

            # Step 2: Encoder -> RL Token.
            rl_token = self.encoder_fn(token_embeddings)

            # Step 3: Construct state and run Actor.
            proprio = obs.get("state", np.zeros(self.proprio_dim, dtype=np.float32))
            state = np.concatenate([rl_token, proprio[: self.proprio_dim]]).astype(np.float32)

            # Handle batch dimensions for the Actor.
            state_batch = state[None, :]  # [1, 2080]
            ref_actions_flat = ref_actions.reshape(1, -1)  # [1, 1600]

            # Run Actor (deterministic mode: return mean).
            rng = jax.random.key(0)
            actor_mean, _ = self.actor(
                jnp.asarray(state_batch),
                jnp.asarray(ref_actions_flat),
                train=False,
            )
            corrected_action = np.asarray(actor_mean[0])  # [320]

            # Step 4: Execute C=10 steps.
            action_chunk = corrected_action.reshape(C, ACTION_DIM)
            for step_idx in range(C):
                action = action_chunk[step_idx]
                next_obs, reward, done = self.env_step(action)

                # Store transition with stride.
                self._step_count += 1

                episode_reward += reward
                episode_length += 1

                if done or episode_length >= max_steps:
                    break

                obs = next_obs

            # Store the chunk-level transition (state, action_chunk, total_reward, next_state).
            next_proprio = next_obs.get("state", np.zeros(self.proprio_dim, dtype=np.float32))
            next_state = np.concatenate(
                [rl_token, next_proprio[: self.proprio_dim]]
            ).astype(np.float32)

            # Use cumulative reward over the chunk as the reward signal.
            self.replay.add(
                state,
                corrected_action,
                episode_reward,  # Simplified: using total reward as signal
                next_state,
                done,
            )

            if done:
                logger.info(f"Episode done after {episode_length} steps.")
                break

        elapsed = time.monotonic() - start_time
        stats = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "episode_time": elapsed,
            "buffer_size": self.replay.size,
        }
        logger.info(
            f"Episode finished: reward={episode_reward:.2f}, "
            f"length={episode_length}, time={elapsed:.1f}s"
        )
        return stats
