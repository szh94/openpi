"""Stage 2: Online RL fine-tuning with TD3 + BC.

Loads frozen VLA + frozen RLTEncoder, initializes Actor + Critic + Replay Buffer,
and runs online RL training.

Process:
    1. Load frozen VLA + frozen RLTEncoder
    2. Initialize Actor + Critic + Replay Buffer
    3. Launch rollout + learner loop
    4. TD3 + BC training
"""

import dataclasses
import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import rl_token as _rl_token
from openpi.training import config as _config
from openpi.shared import array_typing as at

from rlt_online_rl.actor import Actor
from rlt_online_rl.critic import Critic
from rlt_online_rl.learner import Learner
from rlt_online_rl.replay import ReplayBuffer

logger = logging.getLogger("train_rlt_online")


@dataclasses.dataclass(frozen=True)
class RLTOnlineTrainConfig:
    """Configuration for Stage 2 online RL training."""

    # Path to the pre-trained VLA checkpoint.
    vla_checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    # VLA config name.
    vla_config_name: str = "pi05_aloha"
    # RL Token encoder config.
    encoder_config: _rl_token.RLTEncoderConfig = dataclasses.field(
        default_factory=_rl_token.RLTEncoderConfig
    )

    # RL hyperparameters.
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    beta: float = 2.0
    policy_delay: int = 2
    updates_per_step: int = 20
    replay_capacity: int = 1_000_000

    # Rollout config.
    max_episode_steps: int = 500
    num_episodes: int = 100
    seed: int = 42


def train(config: RLTOnlineTrainConfig):
    """Run Stage 2 online RL training."""
    logger.info("=== Stage 2: Online RL Fine-tuning ===")

    # 1. Load frozen VLA.
    logger.info(f"Loading VLA from {config.vla_checkpoint_dir}...")
    train_config = _config.get_config(config.vla_config_name)
    vla = train_config.model.load(
        _model.restore_params(config.vla_checkpoint_dir, dtype=jnp.bfloat16)
    )
    vla.eval()

    # 2. Initialize and load frozen RLTEncoder.
    logger.info("Loading RLTEncoder...")
    enc_rng = jax.random.PRNGKey(config.seed)
    encoder = _rl_token.RLTEncoder(config.encoder_config, rngs=nnx.Rngs(enc_rng))
    encoder.eval()

    # 3. Initialize Actor and Critic.
    logger.info("Initializing Actor and Critic...")
    rng = jax.random.PRNGKey(config.seed + 1)
    actor = Actor(rngs=nnx.Rngs(rng))
    critic = Critic(rngs=nnx.Rngs(rng))

    # 4. Initialize Replay Buffer.
    replay = ReplayBuffer(capacity=config.replay_capacity)

    # 5. Initialize Learner.
    learner = Learner(
        actor=actor,
        critic=critic,
        replay_buffer=replay,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        beta=config.beta,
        policy_delay=config.policy_delay,
        batch_size=config.batch_size,
        updates_per_step=config.updates_per_step,
    )

    # 6. Run online RL loop.
    logger.info("Starting online RL loop...")

    for episode in range(config.num_episodes):
        # Reset environment (placeholder).
        logger.info(f"Episode {episode + 1}/{config.num_episodes}")

        # Run learner updates.
        metrics = learner.update()

        if metrics and "critic_loss" in metrics:
            logger.info(
                f"  critic_loss={metrics['critic_loss']:.4f}, "
                f"q1={metrics.get('q1', 0):.4f}, "
                f"actor_loss={metrics.get('actor_loss', 0):.4f}"
            )

        if replay.is_ready:
            logger.info(f"  Buffer size: {replay.size}")

    logger.info("Stage 2 training complete.")
    return actor, critic


def cli():
    """CLI entry point."""
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    cfg = tyro.cli(RLTOnlineTrainConfig)
    train(cfg)


if __name__ == "__main__":
    cli()
