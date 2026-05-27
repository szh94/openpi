"""RLT Policy serving script.

Loads frozen VLA + frozen RLTEncoder + trained Actor and serves over WebSocket.
Uses the same pattern as serve_policy.py.

Supports:
    - Loading a pre-trained RLT policy from checkpoints
    - WebSocket-based inference
    - Health check endpoint
"""

import dataclasses
import enum
import logging
import socket

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tyro

from openpi.models import model as _model
from openpi.models import rl_token as _rl_token
from openpi.policies import rlt_policy as _rlt_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server as _server
from openpi.training import config as _config

from rlt_online_rl.actor import Actor

logger = logging.getLogger("serve_rlt_policy")


class EnvMode(enum.Enum):
    """Supported environments for RLT."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    RLT = "rlt"


@dataclasses.dataclass
class Args:
    """Arguments for the RLT policy server."""

    # Environment to serve the policy for.
    env: EnvMode = EnvMode.RLT

    # If provided, will be used as the default prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Path to the VLA checkpoint.
    vla_checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    # VLA config name.
    vla_config_name: str = "pi05_aloha"

    # Path to the trained RLT checkpoint (encoder + actor).
    rlt_checkpoint_dir: str | None = None

    # Proprioception dimension.
    proprio_dim: int = 32
    # Action chunk size for the Actor.
    action_chunk: int = 10
    # Per-step action dimension.
    action_dim: int = 32

    # Record policy behavior for debugging.
    record: bool = False


def create_rlt_policy(args: Args) -> _policy.BasePolicy:
    """Create an RLT policy from the given arguments."""
    logger.info("Loading VLA model...")
    train_config = _config.get_config(args.vla_config_name)
    vla = train_config.model.load(
        _model.restore_params(args.vla_checkpoint_dir, dtype=jnp.bfloat16)
    )
    vla.eval()

    # Initialize encoder.
    logger.info("Initializing RLTEncoder...")
    encoder = _rl_token.RLTEncoder(
        _rl_token.RLTEncoderConfig(),
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    encoder.eval()

    # Load encoder checkpoint if available.
    if args.rlt_checkpoint_dir:
        logger.info(f"Loading RLT encoder from {args.rlt_checkpoint_dir}/encoder/params")
        encoder_params = _model.restore_params(
            f"{args.rlt_checkpoint_dir}/encoder/params", dtype=jnp.bfloat16
        )
        enc_graphdef, enc_state = nnx.split(encoder)
        enc_state.replace_by_pure_dict(encoder_params)
        encoder = nnx.merge(enc_graphdef, enc_state)

    # Load or initialize Actor.
    logger.info("Initializing Actor...")
    actor = Actor(rngs=nnx.Rngs(jax.random.PRNGKey(1)))
    if args.rlt_checkpoint_dir:
        logger.info(f"Loading Actor from {args.rlt_checkpoint_dir}/actor/params")
        actor_params = _model.restore_params(
            f"{args.rlt_checkpoint_dir}/actor/params", dtype=jnp.bfloat16
        )
        act_graphdef, act_state = nnx.split(actor)
        act_state.replace_by_pure_dict(actor_params)
        actor = nnx.merge(act_graphdef, act_state)
    actor.eval()

    # Get data config for transforms.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    return _rlt_policy.RLTPolicy(
        vla_model=vla,
        encoder=encoder,
        actor=actor,
        transforms=data_config.model_transforms.inputs,
        output_transforms=data_config.model_transforms.outputs,
        proprio_dim=args.proprio_dim,
        action_chunk=args.action_chunk,
        action_dim=args.action_dim,
        metadata=train_config.policy_metadata,
    )


def main(args: Args) -> None:
    """Main entry point."""
    policy = create_rlt_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "rlt_policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating RLT server (host: %s, ip: %s)", hostname, local_ip)

    server = _server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
