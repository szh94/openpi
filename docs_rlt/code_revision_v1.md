# Code Revision v1 — RLT (RL Token) Implementation

## Overview

This document describes all changes made to implement RL Token (RLT) functionality on top of the openpi codebase. RLT freezes a pre-trained VLA (pi0.5), adds an RL Token Encoder-Decoder, and attaches a lightweight Actor-Critic for online RL fine-tuning.

Based on *RL Token: Bootstrapping Online RL with Vision-Language-Action Models* (Physical Intelligence, 2026).

> **Note on model versions**: The RLT paper uses **pi0.6** (Gemma 3 4B backbone + 860M action expert) as the base VLA. Our implementation builds on **pi0.5** (Gemma 2B + 300M action expert) from the openpi codebase, since pi0.6 is proprietary. The RLT method is architecture-agnostic and works with both models.

---

## Files Created

### 1. `src/openpi/models/rl_token.py` — RL Token Encoder-Decoder

- **`RLTEncoder`**: 4-layer Transformer encoder (width=2048, 8 heads) that compresses VLA token embeddings `[B, N, 2048]` into a single RL Token `[B, 2048]`. Uses a learnable `[RL]` special token appended to the input sequence.
- **`RLTDecoder`**: Transformer decoder that reconstructs `[B, N, 2048]` from `[B, 2048]` using cross-attention (Stage 1 training only).
- **`RLTEncoderConfig` / `RLTDecoderConfig`**: Configuration dataclasses with `width`, `depth`, `num_heads`.
- **Architecture**: Standard pre-norm Transformer with multi-head attention, MLP blocks, and optional cross-attention.

**Rationale**: The VLA's intermediate token embeddings contain rich visual-linguistic information about the scene. Compressing them into a single vector (RL Token) provides a compact state representation for the online RL Actor-Critic.

### 2. `src/openpi/policies/rlt_policy.py` — RLT Policy Wrapper

- **`RLTPolicy`**: Wraps a frozen VLA model + frozen RLTEncoder + online Actor.
- **`infer()` flow**: VLA → token embeddings → RLTEncoder → RL Token → concat with proprioception → Actor → corrected actions.
- Falls back to VLA reference actions if no Actor is provided.

**Rationale**: Provides a clean `BasePolicy` interface compatible with the existing serving infrastructure while supporting the RLT inference pipeline.

### 3. `rlt_online_rl/actor.py` — Actor Network

- **`Actor`**: 2-layer MLP (512 hidden units).
- Input: `concat([state (2080), ref_actions_flat (1600)])` → `[3680]`.
- Output: `mean [320]` + `std [320]` (via softplus-clamped log_std).
- Loss: `-Q + beta * MSE(a, ref_actions[:C])` (TD3 + BC).

### 4. `rlt_online_rl/critic.py` — Critic Network

- **`Critic`**: 2-layer MLP (256 hidden units) with twin copies for TD3.
- Input: `concat([state (2080), action_chunk_flat (320)])` → `[2400]`.
- Output: Q-value `[1]`.
- Features: target networks with Polyak soft updates (tau=0.005).

### 5. `rlt_online_rl/replay.py` — Replay Buffer

- Circular buffer with capacity up to 1M transitions.
- Each entry: `(state [2080], action [320], reward [1], next_state [2080], done [1])`.
- Uniform random sampling.
- Stride=2 storage for overlapping transitions (5x sample efficiency).

### 6. `rlt_online_rl/learner.py` — TD3 + BC Learner

- Updates Critic: minimize TD error using standard TD3 target.
- Updates Actor: maximize Q + BC regularization (delayed every 2 steps).
- Target network soft updates (Polyak tau=0.005).
- `updates_per_step=20`: reuses each collected transition for multiple gradient steps.

### 7. `rlt_online_rl/rollout.py` — Rollout Runtime

- Robot interaction loop: VLA → Encoder → Actor → execute C=10 steps.
- Chunk-level transition storage with stride=2.
- Interfaces with environment step function, collects rewards.

### 8. `scripts/train_rlt_token.py` — Stage 1 Training

- Loads pre-trained VLA (pi0.5/pi0.6), freezes all params.
- Initializes RLTEncoder + RLTDecoder.
- MSE reconstruction loss on VLA token embeddings.
- After training, freezes encoder and discards decoder.

### 9. `scripts/train_rlt_online.py` — Stage 2 Training

- Loads frozen VLA + frozen RLTEncoder.
- Initializes Actor + Critic + Replay Buffer.
- TD3 + BC training loop with configurable hyperparameters.

### 10. `scripts/serve_rlt_policy.py` — RLT Policy Serving

- Loads frozen VLA + frozen RLTEncoder + trained Actor.
- Serves over WebSocket (compatible with `WebsocketPolicyServer`).
- Supports all standard environment modes plus `RLT` mode.

---

## Files Modified

### 11. `src/openpi/models/model.py`

- **`ModelType` enum**: Added `RLT = "rlt"`.
- **`RLTBaseModelConfig`**: New frozen dataclass extending `BaseModelConfig` with:
  - `rl_token_width: int = 2048`
  - `rl_encoder_depth: int = 4`
  - `rl_encoder_num_heads: int = 8`
  - `action_chunk: int = 10`
  - `proprio_dim: int = 32`

### 12. `src/openpi/models/pi0.py`

- **`get_token_embeddings()`**: New method on `Pi0` class that:
  - Runs full VLA forward pass (embed_prefix + embed_suffix + Transformer LLM).
  - Returns ALL token embeddings `[B, N, 2048]` (both prefix_out and suffix_out, unified to 2048 dim via projection).
  - Returns reference actions `[B, 50, 32]`.
  - Applies `jax.lax.stop_gradient` (critical for frozen VLA).
- **`_suffix_emb_proj`**: New projection layer to unify suffix_out (1024-dim) to prefix_out (2048-dim).

### 13. `src/openpi/training/config.py`

- **`rlt_debug`**: New config entry for development/testing with fake data.
- Imported `rl_token` module.

---

## Architecture

### Stage 1: RL Token Training (Offline)

```
Demo Data → VLA (frozen) → Token Embeddings [B, N, 2048]
                                    ↓
                            RLTEncoder → RL Token [B, 2048]
                                    ↓
                            RLTDecoder → Reconstructed [B, N, 2048]
                                    ↓
                            MSE Loss(reconstructed, original)
```

### Stage 2: Online RL Fine-tuning

```
Observation → VLA (frozen) → Token Embeddings → RLTEncoder (frozen) → RL Token [2048]
                                                                              ↓
                                    Proprioception [32] → concat → State [2080]
                                                                        ↓
                                                    Actor → Corrected Actions [320]
                                                                        ↓
                                                    Environment → Reward + Next State
                                                                        ↓
                                                    Replay Buffer → Learner (TD3+BC)
```

### Key Dimensions

| Component | Input Shape | Output Shape |
|-----------|------------|--------------|
| VLA | Observation | `[50, 32]` actions + `[N, 2048]` embeddings |
| RLTEncoder | `[N, 2048]` | `[2048]` RL Token |
| State | RL Token `[2048]` + Proprio `[32]` | `[2080]` |
| Actor | State `[2080]` + Ref Actions `[1600]` | `[320]` = 10×32 |
| Critic | State `[2080]` + Action `[320]` | `[1]` Q-value |

---

## Hyperparameters

### Stage 1
| Parameter | Value |
|-----------|-------|
| Encoder depth | 4 |
| Encoder width | 2048 |
| Encoder heads | 8 |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Training steps | 30,000 |

### Stage 2
| Parameter | Value |
|-----------|-------|
| Actor hidden | 512 |
| Critic hidden | 256 |
| Actor LR | 3e-4 |
| Critic LR | 3e-4 |
| Gamma | 0.99 |
| Beta (BC reg) | 2.0 |
| Policy delay | 2 |
| Batch size | 256 |
| Updates/step | 20 |
| Polyak tau | 0.005 |
| Smooth noise | 0.2 |
| Noise clip | 0.5 |

---

## Framework Dependencies

All new RLT code is **pure JAX / Flax NNX**, with zero PyTorch dependencies, consistent with the existing openpi codebase.

| File | Framework | Matches openpi |
|------|-----------|:--------------:|
| `rl_token.py` | `flax.nnx` + `jax.numpy` | ✅ |
| `rlt_policy.py` | `jax` + `jax.numpy` | ✅ |
| `rlt_online_rl/actor.py` | `flax.nnx` + `jax` | ✅ |
| `rlt_online_rl/critic.py` | `flax.nnx` + `jax.numpy` | ✅ |
| `rlt_online_rl/learner.py` | `flax.nnx` + `jax` + `optax` | ✅ |
| `rlt_online_rl/rollout.py` | `jax` + `jax.numpy` | ✅ |
| `rlt_online_rl/replay.py` | `numpy` (data only) | ✅ |

All neural network modules inherit from `flax.nnx.Module`, optimizers use `optax`, and the training pipeline runs on JAX's functional transform system.

---

## Verification

```bash
# 1. Module imports
python -c "from openpi.models.rl_token import RLTEncoder; print('OK')"

# 2. Config loading
python -c "from openpi.training.config import get_config; cfg = get_config('rlt_debug')"

# 3. RLT policy creation
python -c "from openpi.policies.rlt_policy import RLTPolicy; print('OK')"

# 4. Online RL modules
cd rlt_online_rl && python -c "
from rlt_online_rl.actor import Actor;
from rlt_online_rl.critic import Critic;
from rlt_online_rl.replay import ReplayBuffer;
print('All OK')
"
```
