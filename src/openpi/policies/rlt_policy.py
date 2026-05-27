"""RLT Policy wrapper.

The RLTPolicy wraps a frozen VLA model + frozen RLTEncoder + online Actor
for online RL fine-tuning inference.

Inference flow:
    1. VLA.get_token_embeddings(obs) -> (all_embeddings, ref_actions)
    2. RLTEncoder(all_embeddings) -> rl_token
    3. State = concat(rl_token, proprio)
    4. Actor(state, ref_actions) -> corrected action chunk a_1:C
    5. Return corrected actions
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rl_token as _rl_token
from openpi.policies import policy as _policy
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi_client import base_policy as _base_policy

logger = logging.getLogger("openpi")

# RLT dimensions.
STATE_DIM = 2080  # RL Token (2048) + proprio (32)
REF_ACTION_FLAT = 1600  # ref_actions [50, 32] flattened


class RLTPolicy(_base_policy.BasePolicy):
    """RLT inference policy: VLA -> RL Token -> Actor correction -> execute.

    Supports both frozen VLA inference and online Actor inference.
    """

    def __init__(
        self,
        vla_model: _model.BaseModel,
        encoder: _rl_token.RLTEncoder,
        actor: Callable | None = None,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[Callable] = (),
        output_transforms: Sequence[Callable] = (),
        proprio_dim: int = 32,
        action_chunk: int = 10,
        action_dim: int = 32,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the RLT policy.

        Args:
            vla_model: Frozen VLA model (Pi0/Pi05).
            encoder: Frozen RLT Encoder.
            actor: Optional Actor network. If None, uses VLA reference actions directly.
            rng: JAX random key.
            transforms: Input transforms.
            output_transforms: Output transforms.
            proprio_dim: Proprioception dimension.
            action_chunk: Number of steps in each action chunk (C).
            action_dim: Per-step action dimension.
            metadata: Policy metadata.
        """
        self._vla = vla_model
        self._encoder = encoder
        self._actor = actor
        self._proprio_dim = proprio_dim
        self._action_chunk = action_chunk
        self._action_dim = action_dim
        self._metadata = metadata or {}
        self._rng = rng or jax.random.key(0)

        # JIT-compile the VLA embedding extraction.
        self._vla_embed_fn = nnx_utils.module_jit(vla_model.get_token_embeddings)

        # JIT-compile the encoder.
        self._encoder_fn = nnx_utils.module_jit(encoder.__call__)

        # JIT the actor if provided.
        if actor is not None:
            if isinstance(actor, Callable) and hasattr(actor, "__self__") and isinstance(actor.__self__, object):
                try:
                    self._actor_fn = nnx_utils.module_jit(actor.__call__)
                except (ValueError, AttributeError):
                    self._actor_fn = actor
            else:
                self._actor_fn = actor
        else:
            self._actor_fn = None

        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def infer(self, obs: dict) -> dict:
        """Run RLT inference.

        Args:
            obs: Input observation dict.

        Returns:
            dict with "actions" and other metadata.
        """
        # Apply input transforms.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Batch and convert to JAX.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        # Step 1: VLA forward -> token embeddings + reference actions.
        all_embeddings, ref_actions = self._vla_embed_fn(observation)

        # Step 2: Encoder -> RL Token.
        rl_token = self._encoder_fn(all_embeddings)

        # Step 3: Construct state.
        proprio = observation.state[:, : self._proprio_dim]
        state = jnp.concatenate([rl_token, proprio], axis=-1)  # [B, 2080]

        if self._actor_fn is not None and self._actor is not None:
            # Step 4a: Actor correction.
            ref_actions_flat = ref_actions.reshape(ref_actions.shape[0], -1)
            self._rng, actor_rng = jax.random.split(self._rng)
            # Use deterministic mode (mean) for inference.
            mean, _ = self._actor_fn(state, ref_actions_flat)
            corrected_actions = mean.reshape(-1, self._action_chunk, self._action_dim)
            actions = np.asarray(corrected_actions[0])
        else:
            # Step 4b: Use VLA reference actions directly (no actor).
            actions = np.asarray(ref_actions[0, : self._action_chunk])

        # Apply output transforms.
        outputs = {"actions": actions, "state": np.asarray(proprio[0])}
        outputs = self._output_transform(outputs)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
