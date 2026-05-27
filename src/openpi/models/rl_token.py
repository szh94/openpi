"""RL Token Encoder-Decoder for RLT (RL Token) fine-tuning.

The RLTEncoder compresses VLA token embeddings [B, N, 2048] into a single RL Token [B, 2048]
using a learnable special [RL] token prepended to the input sequence.

The RLTDecoder reconstructs [B, N, 2048] from [B, 2048] for Stage 1 reconstruction training.
"""

import dataclasses
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class RLTEncoderConfig:
    """Configuration for the RL Token Encoder."""

    width: int = 2048
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0


@dataclasses.dataclass(frozen=True)
class RLTDecoderConfig:
    """Configuration for the RL Token Decoder."""

    width: int = 2048
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0


def _default_init(scale: float = 0.02):
    """Default weight initializer for Transformer layers."""
    return nnx.initializers.truncated_normal(stddev=scale)


class _Attention(nnx.Module):
    """Multi-head self-attention with pre-norm."""

    def __init__(self, width: int, num_heads: int, rngs: nnx.Rngs):
        self.width = width
        self.num_heads = num_heads
        assert width % num_heads == 0, f"width={width} must be divisible by num_heads={num_heads}"
        self.head_dim = width // num_heads

        self.ln = nnx.LayerNorm(width, rngs=rngs)
        self.qkv = nnx.Linear(width, 3 * width, kernel_init=_default_init(), rngs=rngs)
        self.proj = nnx.Linear(width, width, kernel_init=_default_init(), rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b n d"], mask: at.Float[at.Array, "b n n"] | None = None):
        B, N, D = x.shape
        x = self.ln(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = jnp.einsum("bthd,bshd->bhts", q, k) * scale
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None, :, :]  # [B, 1, T, S] for broadcasting with heads
            attn = jnp.where(mask, attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(B, N, D)
        return self.proj(out)


class _MLP(nnx.Module):
    """Two-layer MLP with GeLU activation."""

    def __init__(self, width: int, mlp_ratio: float, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(width, int(width * mlp_ratio), kernel_init=_default_init(), rngs=rngs)
        self.fc2 = nnx.Linear(int(width * mlp_ratio), width, kernel_init=_default_init(), rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b n d"]):
        return self.fc2(nnx.gelu(self.fc1(x)))


class _TransformerBlock(nnx.Module):
    """Pre-norm Transformer block with self-attention + MLP."""

    def __init__(self, width: int, num_heads: int, mlp_ratio: float, rngs: nnx.Rngs):
        self.attn = _Attention(width, num_heads, rngs=rngs)
        self.mlp = _MLP(width, mlp_ratio, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b n d"], mask: at.Float[at.Array, "b n n"] | None = None):
        x = x + self.attn(x, mask)
        x = x + self.mlp(x)
        return x


class _CrossAttentionBlock(nnx.Module):
    """Transformer decoder block with self-attention + cross-attention + MLP."""

    def __init__(self, width: int, num_heads: int, mlp_ratio: float, rngs: nnx.Rngs):
        self.width = width
        self.num_heads = num_heads
        assert width % num_heads == 0
        self.head_dim = width // num_heads

        # Self-attention (causal-like, but full attention for reconstruction).
        self.self_ln = nnx.LayerNorm(width, rngs=rngs)
        self.self_qkv = nnx.Linear(width, 3 * width, kernel_init=_default_init(), rngs=rngs)
        self.self_proj = nnx.Linear(width, width, kernel_init=_default_init(), rngs=rngs)

        # Cross-attention (queries from decoder, keys/values from encoder).
        self.cross_ln_q = nnx.LayerNorm(width, rngs=rngs)
        self.cross_ln_kv = nnx.LayerNorm(width, rngs=rngs)
        self.cross_q = nnx.Linear(width, width, kernel_init=_default_init(), rngs=rngs)
        self.cross_k = nnx.Linear(width, width, kernel_init=_default_init(), rngs=rngs)
        self.cross_v = nnx.Linear(width, width, kernel_init=_default_init(), rngs=rngs)
        self.cross_proj = nnx.Linear(width, width, kernel_init=_default_init(), rngs=rngs)

        # MLP.
        self.mlp = _MLP(width, mlp_ratio, rngs=rngs)

    def _attn(self, q, k, v, mask=None):
        B, Nq, D = q.shape
        _, Nkv, _ = k.shape
        q = q.reshape(B, Nq, self.num_heads, self.head_dim)
        k = k.reshape(B, Nkv, self.num_heads, self.head_dim)
        v = v.reshape(B, Nkv, self.num_heads, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None, :, :]  # [B, 1, Q, K] for broadcasting with heads
            attn = jnp.where(mask, attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, Nq, D)
        return out

    def __call__(
        self,
        x: at.Float[at.Array, "b n d"],
        context: at.Float[at.Array, "b m d"],
        self_mask: at.Float[at.Array, "b n n"] | None = None,
        cross_mask: at.Float[at.Array, "b n m"] | None = None,
    ):
        # Self-attention.
        skip = x
        x = self.self_ln(x)
        qkv = self.self_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        x = self._attn(q, k, v, self_mask)
        x = self.self_proj(x)
        x = skip + x

        # Cross-attention.
        skip = x
        q = self.cross_q(self.cross_ln_q(x))
        k = self.cross_k(self.cross_ln_kv(context))
        v = self.cross_v(self.cross_ln_kv(context))
        x = self._attn(q, k, v, cross_mask)
        x = self.cross_proj(x)
        x = skip + x

        # MLP.
        x = x + self.mlp(x)
        return x


class RLTEncoder(nnx.Module):
    """RL Token Encoder.

    Compresses VLA token embeddings [B, N, 2048] into a single RL Token [B, 2048].
    Appends a learnable special [RL] token to the input sequence and takes its output.
    """

    def __init__(self, config: RLTEncoderConfig, rngs: nnx.Rngs):
        self.config = config
        # Learnable [RL] special token embedding.
        self.rl_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, config.width)) * 0.02
        )
        # Learnable absolute position embeddings (max 4096 tokens).
        max_len = 4096
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, max_len, config.width)) * 0.02
        )
        # Transformer blocks.
        self.blocks = [
            _TransformerBlock(config.width, config.num_heads, config.mlp_ratio, rngs=rngs)
            for _ in range(config.depth)
        ]
        # Output norm.
        self.norm = nnx.LayerNorm(config.width, rngs=rngs)

    def __call__(
        self, embeddings: at.Float[at.Array, "b n 2048"]
    ) -> at.Float[at.Array, "b 2048"]:
        """Encode VLA token embeddings into a single RL Token.

        Args:
            embeddings: VLA output embeddings [B, N, 2048].

        Returns:
            rl_token: RL Token [B, 2048] from the [RL] special token position.
        """
        B, N, D = embeddings.shape

        # Append [RL] special token to the sequence.
        rl_tok = jnp.broadcast_to(jnp.asarray(self.rl_token), (B, 1, D))
        x = jnp.concatenate([embeddings, rl_tok], axis=1)  # [B, N+1, D]

        # Add positional embeddings.
        x = x + jnp.asarray(self.pos_embed[:, : N + 1, :])

        # Create full attention mask (all tokens attend to all tokens).
        mask = jnp.ones((B, N + 1, N + 1), dtype=jnp.bool_)

        # Run Transformer blocks.
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # Return the [RL] token output (last position).
        return x[:, -1, :]


class RLTDecoder(nnx.Module):
    """RL Token Decoder.

    Reconstructs [B, N, 2048] from [B, 2048] for Stage 1 reconstruction training.
    Uses cross-attention to attend to the RL Token from the encoder.
    """

    def __init__(self, config: RLTDecoderConfig, rngs: nnx.Rngs):
        self.config = config
        # Learnable position embeddings for the reconstructed sequence (max N tokens).
        max_len = 4096
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, max_len, config.width)) * 0.02
        )
        # Learnable output tokens that will be refined by the decoder.
        # We start from learned embeddings rather than zeros for faster convergence.
        self.query_tokens = nnx.Param(
            jax.random.normal(rngs.params(), (1, max_len, config.width)) * 0.02
        )
        # Decoder blocks with self-attention + cross-attention.
        self.blocks = [
            _CrossAttentionBlock(config.width, config.num_heads, config.mlp_ratio, rngs=rngs)
            for _ in range(config.depth)
        ]
        # Output projection (to match input dim, which is already 2048).
        self.norm = nnx.LayerNorm(config.width, rngs=rngs)

    def __call__(
        self, rl_token: at.Float[at.Array, "b 2048"], num_tokens: int
    ) -> at.Float[at.Array, "b n 2048"]:
        """Decode an RL Token into reconstructed token embeddings.

        Args:
            rl_token: RL Token [B, 2048] from the encoder.
            num_tokens: Number of tokens N to reconstruct.

        Returns:
            Reconstructed embeddings [B, N, 2048].
        """
        B, D = rl_token.shape

        # Get learned query tokens for the output sequence.
        query = jnp.broadcast_to(jnp.asarray(self.query_tokens[:, :num_tokens, :]), (B, num_tokens, D))
        # Add position embeddings.
        query = query + jnp.asarray(self.pos_embed[:, :num_tokens, :])

        # Context is the RL Token expanded to [B, 1, D].
        context = rl_token[:, None, :]

        # Full self-attention mask and cross-attention mask.
        self_mask = jnp.ones((B, num_tokens, num_tokens), dtype=jnp.bool_)
        cross_mask = jnp.ones((B, num_tokens, 1), dtype=jnp.bool_)

        x = query
        for block in self.blocks:
            x = block(x, context, self_mask, cross_mask)

        return self.norm(x)
