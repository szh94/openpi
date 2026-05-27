"""Critic network for RLT online RL fine-tuning (TD3).

Architecture (twin critics):
    Input: concat([state (2080), action_chunk_flat (320)])  -> [2400]
    Hidden: 256 -> ReLU -> 256 -> ReLU
    Output: Q-value [1]

TD3 features:
    - Twin critics (two copies for min-Q target)
    - Target networks with Polyak averaging
"""

from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.shared import array_typing as at

# Dimensions.
STATE_DIM = 2080  # RL Token (2048) + proprio (32)
ACTION_DIM = 320  # Actor output: C=10 steps x 32 dim
CRITIC_INPUT_DIM = STATE_DIM + ACTION_DIM  # 2400


class _CriticNetwork(nnx.Module):
    """Single critic network."""

    def __init__(self, hidden_dim: int = 256, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.fc1 = nnx.Linear(CRITIC_INPUT_DIM, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self,
        state: at.Float[at.Array, "b 2080"],
        action_chunk: at.Float[at.Array, "b 320"],
    ) -> at.Float[at.Array, "b 1"]:
        """Compute Q(s, a).

        Args:
            state: Concatenated RL token + proprioception [B, 2080].
            action_chunk: Action chunk [B, 320] (C=10 steps x 32 dim, flattened).

        Returns:
            Q-value [B, 1].
        """
        x = jnp.concatenate([state, action_chunk], axis=-1)
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.out(x)


class Critic(nnx.Module):
    """Twin critics for TD3 with target network management."""

    def __init__(
        self,
        hidden_dim: int = 256,
        polyak_tau: float = 0.005,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.polyak_tau = polyak_tau
        # Online critics.
        self.q1 = _CriticNetwork(hidden_dim, rngs=rngs)
        self.q2 = _CriticNetwork(hidden_dim, rngs=rngs)
        # Target critics (initialized as copies).
        self.target_q1 = _CriticNetwork(hidden_dim, rngs=rngs)
        self.target_q2 = _CriticNetwork(hidden_dim, rngs=rngs)
        # Sync targets with online initially.
        self._sync_targets()

    def _sync_targets(self):
        """Copy online params to target networks."""
        # Use nnx.State to copy parameters.
        for target, source in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
            target_graphdef, target_state = nnx.split(target)
            _, source_state = nnx.split(source)
            target_state = source_state  # replace target state with source
            nnx.merge(target_graphdef, target_state)

    def soft_update_targets(self):
        """Polyak averaging: target = tau * online + (1 - tau) * target."""
        tau = self.polyak_tau
        for target, source in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
            target_graphdef, target_state = nnx.split(target)
            _, source_state = nnx.split(source)
            # Apply Polyak update to each parameter.
            updated_state = jax.tree.map(
                lambda t, s: tau * s + (1 - tau) * t,
                target_state,
                source_state,
            )
            nnx.update(target, updated_state)

    def __call__(
        self,
        state: at.Float[at.Array, "b 2080"],
        action_chunk: at.Float[at.Array, "b 320"],
        *,
        use_target: bool = False,
    ) -> tuple[at.Float[at.Array, "b 1"], at.Float[at.Array, "b 1"]]:
        """Compute Q-values from both critics.

        Args:
            state: Batch of states [B, 2080].
            action_chunk: Batch of action chunks [B, 320].
            use_target: If True, use target networks instead of online.

        Returns:
            Tuple of (q1, q2) values, each [B, 1].
        """
        if use_target:
            return self.target_q1(state, action_chunk), self.target_q2(state, action_chunk)
        return self.q1(state, action_chunk), self.q2(state, action_chunk)

    def compute_loss(
        self,
        state: at.Float[at.Array, "b 2080"],
        action_chunk: at.Float[at.Array, "b 320"],
        reward: at.Float[at.Array, "b 1"],
        next_state: at.Float[at.Array, "b 2080"],
        done: at.Float[at.Array, "b 1"],
        gamma: float = 0.99,
        target_actor_output: at.Float[at.Array, "b 320"] | None = None,
        smooth_noise_scale: float = 0.2,
        noise_clip: float = 0.5,
    ) -> dict[str, Any]:
        """Compute the critic (TD3) loss.

        Args:
            state: Batch of states [B, 2080].
            action_chunk: Batch of action chunks [B, 320].
            reward: Batch of rewards [B, 1].
            next_state: Batch of next states [B, 2080].
            done: Batch of done flags [B, 1].
            gamma: Discount factor.
            target_actor_output: Target actor's output on next_state [B, 320].
            smooth_noise_scale: Std for target policy smoothing noise.
            noise_clip: Max noise magnitude for target smoothing.

        Returns:
            dict with "loss", "q1", "q2", "q_target".
        """
        # Compute target Q-value.
        if target_actor_output is not None:
            # Target policy smoothing.
            noise = jax.random.normal(
                jax.random.key(0), target_actor_output.shape
            ) * smooth_noise_scale
            noise = jnp.clip(noise, -noise_clip, noise_clip)
            next_action = target_actor_output + noise

            q1_target, q2_target = self(next_state, next_action, use_target=True)
            q_target_val = jnp.minimum(q1_target, q2_target)
        else:
            q_target_val = jnp.zeros_like(reward)

        q_target = reward + gamma * (1 - done) * q_target_val

        # Current Q estimates.
        q1_val, q2_val = self(state, action_chunk, use_target=False)

        # TD error.
        loss = jnp.mean(jnp.square(q1_val - q_target)) + jnp.mean(
            jnp.square(q2_val - q_target)
        )

        return {
            "loss": loss,
            "q1": q1_val.mean(),
            "q2": q2_val.mean(),
            "q_target": q_target.mean(),
        }
