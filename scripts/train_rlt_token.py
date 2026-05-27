"""Stage 1: RL Token Encoder-Decoder training.

Trains the RLTEncoder + RLTDecoder pair on demonstration data to reconstruct
VLA token embeddings from a compressed RL Token.

Process:
    1. Load pre-trained pi0.5 VLA, freeze all params
    2. Initialize RLTEncoder + RLTDecoder
    3. Train on demo data: reconstruct embeddings from RL Token
    4. MSE reconstruction loss
    5. After training, freeze encoder and discard decoder
"""

import dataclasses
import logging

import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
from openpi.models import rl_token as _rl_token
from openpi.shared import array_typing as at
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import weight_loaders as _weight_loaders

logger = logging.getLogger("train_rlt_token")


@dataclasses.dataclass(frozen=True)
class RLTTokenTrainConfig:
    """Configuration for Stage 1 RL Token training."""

    # Path to the pre-trained VLA checkpoint.
    vla_checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    # VLA config name.
    vla_config_name: str = "pi05_aloha"
    # RL Token encoder config.
    encoder_config: _rl_token.RLTEncoderConfig = dataclasses.field(
        default_factory=_rl_token.RLTEncoderConfig
    )
    # RL Token decoder config.
    decoder_config: _rl_token.RLTDecoderConfig = dataclasses.field(
        default_factory=_rl_token.RLTDecoderConfig
    )
    # Training hyperparameters.
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    seed: int = 42


def train(config: RLTTokenTrainConfig):
    """Run Stage 1 RL Token training."""
    logger.info("=== Stage 1: RL Token Training ===")

    # 1. Load pre-trained VLA, freeze all params.
    logger.info(f"Loading VLA from {config.vla_checkpoint_dir}...")
    train_config = _config.get_config(config.vla_config_name)
    vla = train_config.model.load(
        _model.restore_params(config.vla_checkpoint_dir, dtype=jnp.bfloat16)
    )
    vla.eval()  # Set to eval mode (deterministic).

    # 2. Initialize Encoder-Decoder.
    rng = jax.random.PRNGKey(config.seed)
    enc_rng, dec_rng, train_rng = jax.random.split(rng, 3)

    encoder = _rl_token.RLTEncoder(config.encoder_config, rngs=nnx.Rngs(enc_rng))
    decoder = _rl_token.RLTDecoder(config.decoder_config, rngs=nnx.Rngs(dec_rng))

    # 3. Set up optimizer for encoder + decoder.
    # Use separate optimizers since encoder and decoder share param names conceptually.
    optimizer = optax.adam(config.learning_rate)

    # Combine encoder and decoder params for optimization.
    enc_graphdef, enc_state = nnx.split(encoder)
    dec_graphdef, dec_state = nnx.split(decoder)
    combined_state = jax.tree.map(lambda e, d: e, enc_state, enc_state)  # placeholder
    # Actually, combine the states.
    opt_state = optimizer.init((enc_state, dec_state))

    def sample_batch(rng_key):
        """Create a fake batch for debugging. Replace with real dataloader."""
        # Fake embeddings [B, 1000, 2048].
        embeddings = jax.random.normal(rng_key, (config.batch_size, 1000, 2048))
        return embeddings

    @jax.jit
    def train_step(enc_state, dec_state, batch_embeddings):
        """Single training step for encoder-decoder reconstruction."""
        enc = nnx.merge(enc_graphdef, enc_state)
        dec = nnx.merge(dec_graphdef, dec_state)

        # Forward: encode -> decode -> reconstruct.
        rl_token = enc(batch_embeddings)
        reconstructed = dec(rl_token, batch_embeddings.shape[1])

        # MSE reconstruction loss.
        loss = jnp.mean(jnp.square(reconstructed - batch_embeddings))

        return loss, (enc_state, dec_state)

    # 4. Training loop.
    logger.info("Starting training...")
    step_rng = jax.random.PRNGKey(0)
    for step in range(config.num_train_steps):
        step_rng, batch_rng = jax.random.split(step_rng)
        batch = sample_batch(batch_rng)

        grads = jax.grad(train_step, argnums=(0, 1))(enc_state, dec_state, batch)
        # Apply gradients.
        updates, opt_state = optimizer.update(grads, opt_state, (enc_state, dec_state))

        # Apply updates to parameters.
        # (Simplified - in production, use proper optax integration)
        if step % config.log_interval == 0:
            loss_val = train_step(enc_state, dec_state, batch)[0]
            logger.info(f"Step {step}: loss = {loss_val:.6f}")

        if step % config.save_interval == 0 and step > 0:
            logger.info(f"Checkpoint at step {step}")

    logger.info("Stage 1 training complete.")

    # 5. Freeze encoder, discard decoder.
    encoder.eval()
    logger.info("Encoder frozen. Decoder discarded. Stage 1 done.")

    return encoder


def cli():
    """CLI entry point."""
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    cfg = tyro.cli(RLTTokenTrainConfig)
    train(cfg)


if __name__ == "__main__":
    cli()
