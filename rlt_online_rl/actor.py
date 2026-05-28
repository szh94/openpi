"""Actor network for RLT online RL fine-tuning.

Architecture:
    Input: concat([state (2080), ref_actions_flat (1600)])  -> [3680]
    Hidden: 512 -> ReLU -> 512 -> ReLU
    Output: mean [320] (C=10 steps x 32 dim) + log_std [320]
    Loss: -Q + beta * MSE(a, ref_actions[:C])
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at

# Dimensions for the standard RLT setup.
STATE_DIM = 2080  # RL Token (2048) + proprio (32)
REF_ACTION_DIM = 1600  # VLA ref_actions (50 x 32) flattened
ACTION_CHUNK_DIM = 320  # Actor output: C=10 steps x 32 dim
ACTOR_INPUT_DIM = STATE_DIM + REF_ACTION_DIM  # 3680


class Actor(nnx.Module):
    """Actor network that produces corrected action chunks.

    Given the state (RL token + proprioception) and VLA reference actions,
    outputs a distribution over corrected action chunks for the next C steps.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = ACTION_CHUNK_DIM,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        input_dim = ACTOR_INPUT_DIM

        self.fc1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.mean_out = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.log_std_out = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def __call__(
        self,
        state: at.Float[at.Array, "b 2080"],
        ref_actions: at.Float[at.Array, "b 1600"],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "b 320"], at.Float[at.Array, "b 320"]]:
        """Forward pass of the Actor.

        Args:
            state: Concatenated RL token + proprioception [B, 2080].
            ref_actions: VLA reference actions flattened [B, 1600].
            train: If True, samples from the Gaussian; if False, returns the mean.

        Returns:
            mean: Mean of the corrected action chunk [B, 320].
            std: Standard deviation of the corrected action chunk [B, 320].
        """
        x = jnp.concatenate([state, ref_actions], axis=-1)
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))

        mean = self.mean_out(x)
        log_std = self.log_std_out(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)

        return mean, std

    def sample(
        self,
        state: at.Float[at.Array, "b 2080"],
        ref_actions: at.Float[at.Array, "b 1600"],
        rng: at.KeyArrayLike,
    ) -> at.Float[at.Array, "b 320"]:
        """Sample an action from the policy.

        Args:
            state: Concatenated RL token + proprioception [B, 2080].
            ref_actions: VLA reference actions flattened [B, 1600].
            rng: JAX random key.

        Returns:
            Sampled action chunk [B, 320].
        """
        mean, std = self(state, ref_actions, train=True)
        noise = jax.random.normal(rng, mean.shape)
        return mean + noise * std

    def compute_loss(
        self,
        state: at.Float[at.Array, "b 2080"],
        ref_actions: at.Float[at.Array, "b 1600"],
        q1: at.Float[at.Array, "b 1"],
        q2: at.Float[at.Array, "b 1"],
        beta: float = 2.0,
    ) -> dict[str, Any]:
        """Compute the Actor loss: -Q + beta * BC regularization.

        Args:
            state: Batch of states [B, 2080].
            ref_actions: VLA reference actions flattened [B, 1600].
            q1: Q-values from critic 1 [B, 1].
            q2: Q-values from critic 2 [B, 1].
            beta: BC regularization coefficient.

        Returns:
            dict with "loss", "q_loss", "bc_loss", "q_value".
        """
        mean, _ = self(state, ref_actions, train=True)
        q = jnp.minimum(q1, q2)

        # Q loss: negative because we want to maximize Q.
        q_loss = -q.mean()

        # BC regularization: MSE between actor output and ref actions (first C steps x dim).
        bc_loss = jnp.mean(jnp.square(mean - ref_actions[:, : self.action_dim]))

        loss = q_loss + beta * bc_loss

        return {
            "loss": loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "q_value": q.mean(),
        }
