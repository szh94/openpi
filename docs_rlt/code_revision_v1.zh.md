# 代码修订 v1 — RLT (RL Token) 实现

## 概述

本文档描述了在 openpi 代码库上实现 RL Token (RLT) 功能所做的全部更改。RLT 冻结预训练的 VLA（pi0.5），添加 RL Token 编码器-解码器，并附加一个轻量级的 Actor-Critic 用于在线 RL 微调。

基于论文 *RL Token: Bootstrapping Online RL with Vision-Language-Action Models* (Physical Intelligence, 2026)。

> **模型版本说明**：RLT 原论文使用 **pi0.6**（Gemma 3 4B 主干 + 860M action expert）作为基座 VLA。我们的实现基于 openpi 代码库的 **pi0.5**（Gemma 2B + 300M action expert），因为 pi0.6 未开源。RLT 方法与具体模型版本无关，两种架构均可适用。

---

## 创建的文件

### 1. `src/openpi/models/rl_token.py` — RL Token 编码器-解码器

- **`RLTEncoder`**：4 层 Transformer 编码器（width=2048, 8 heads），将 VLA token 嵌入 `[B, N, 2048]` 压缩为单个 RL Token `[B, 2048]`。使用一个可学习的特殊 `[RL]` token 附加到输入序列末尾。
- **`RLTDecoder`**：Transformer 解码器，使用交叉注意力从 `[B, 2048]` 重建 `[B, N, 2048]`（仅 Stage 1 训练使用）。
- **`RLTEncoderConfig` / `RLTDecoderConfig`**：包含 `width`、`depth`、`num_heads` 的配置数据类。
- **架构**：标准 Pre-norm Transformer，包含多头注意力、MLP 块和可选的交叉注意力。

**设计理由**：VLA 的中间 token 嵌入包含丰富的场景视觉-语言信息。将它们压缩为单个向量（RL Token）为在线 RL Actor-Critic 提供了紧凑的状态表示。

### 2. `src/openpi/policies/rlt_policy.py` — RLT 策略包装器

- **`RLTPolicy`**：包装冻结的 VLA 模型 + 冻结的 RLTEncoder + 在线 Actor。
- **`infer()` 流程**：VLA → token 嵌入 → RLTEncoder → RL Token → 与本体感受拼接 → Actor → 修正后的动作。
- 如果未提供 Actor，则回退到 VLA 参考动作。

**设计理由**：提供与现有服务基础设施兼容的干净 `BasePolicy` 接口，同时支持 RLT 推理流程。

### 3. `rlt_online_rl/actor.py` — Actor 网络

- **`Actor`**：2 层 MLP（512 隐藏单元）。
- 输入：`concat([state (2080), ref_actions_flat (1600)])` → `[3680]`。
- 输出：`mean [320]` + `std [320]`（通过 softplus 裁剪的 log_std）。
- 损失：`-Q + beta * MSE(a, ref_actions[:C])`（TD3 + BC）。

### 4. `rlt_online_rl/critic.py` — Critic 网络

- **`Critic`**：2 层 MLP（256 隐藏单元），双副本用于 TD3。
- 输入：`concat([state (2080), action_chunk_flat (320)])` → `[2400]`。
- 输出：Q 值 `[1]`。
- 特性：带 Polyak 软更新的目标网络（tau=0.005）。

### 5. `rlt_online_rl/replay.py` — 回放缓冲区

- 容量高达 100 万条转换的循环缓冲区。
- 每条记录：`(state [2080], action [320], reward [1], next_state [2080], done [1])`。
- 均匀随机采样。
- stride=2 存储，用于重叠转换（5 倍样本效率）。

### 6. `rlt_online_rl/learner.py` — TD3 + BC 学习器

- 更新 Critic：使用标准 TD3 目标最小化 TD 误差。
- 更新 Actor：最大化 Q + BC 正则化（每 2 步延迟更新）。
- 目标网络软更新（Polyak tau=0.005）。
- `updates_per_step=20`：每条收集的转换被多次梯度步骤复用。

### 7. `rlt_online_rl/rollout.py` — Rollout 运行时

- 机器人交互循环：VLA → Encoder → Actor → 执行 C=10 步。
- 块级转换存储，stride=2。
- 与环境步骤函数交互，收集奖励。

### 8. `scripts/train_rlt_token.py` — Stage 1 训练

- 加载预训练的 VLA（pi0.5/pi0.6），冻结所有参数。
- 初始化 RLTEncoder + RLTDecoder。
- 在 VLA token 嵌入上计算 MSE 重建损失。
- 训练完成后冻结编码器，丢弃解码器。

### 9. `scripts/train_rlt_online.py` — Stage 2 训练

- 加载冻结的 VLA + 冻结的 RLTEncoder。
- 初始化 Actor + Critic + 回放缓冲区。
- 可配置超参数的 TD3 + BC 训练循环。

### 10. `scripts/serve_rlt_policy.py` — RLT 策略服务

- 加载冻结的 VLA + 冻结的 RLTEncoder + 训练好的 Actor。
- 通过 WebSocket 提供服务（兼容 `WebsocketPolicyServer`）。
- 支持所有标准环境模式加 `RLT` 模式。

---

## 修改的文件

### 11. `src/openpi/models/model.py`

- **`ModelType` 枚举**：添加 `RLT = "rlt"`。
- **`RLTBaseModelConfig`**：新的冻结数据类，继承 `BaseModelConfig`，包含：
  - `rl_token_width: int = 2048`
  - `rl_encoder_depth: int = 4`
  - `rl_encoder_num_heads: int = 8`
  - `action_chunk: int = 10`
  - `proprio_dim: int = 32`

### 12. `src/openpi/models/pi0.py`

- **`get_token_embeddings()`**：`Pi0` 类的新方法：
  - 运行完整的 VLA 前向传播（embed_prefix + embed_suffix + Transformer LLM）。
  - 返回所有 token 嵌入 `[B, N, 2048]`（prefix_out 和 suffix_out，通过投影统一为 2048 维）。
  - 返回参考动作 `[B, 50, 32]`。
  - 应用 `jax.lax.stop_gradient`（对冻结 VLA 至关重要）。
- **`_suffix_emb_proj`**：新的投影层，将 suffix_out（1024 维）统一为 prefix_out（2048 维）。

### 13. `src/openpi/training/config.py`

- **`rlt_debug`**：用于开发/测试的新配置条目，使用假数据。
- 导入 `rl_token` 模块。

---

## 架构

### Stage 1：RL Token 训练（离线）

```
演示数据 → VLA（冻结）→ Token 嵌入 [B, N, 2048]
                                    ↓
                            RLTEncoder → RL Token [B, 2048]
                                    ↓
                            RLTDecoder → 重建 [B, N, 2048]
                                    ↓
                            MSE 损失(重建, 原始)
```

### Stage 2：在线 RL 微调

```
观测 → VLA（冻结）→ Token 嵌入 → RLTEncoder（冻结）→ RL Token [2048]
                                                                  ↓
                        本体感受 [32] → 拼接 → 状态 [2080]
                                                    ↓
                                    Actor → 修正动作 [320]
                                                    ↓
                                    环境 → 奖励 + 下一状态
                                                    ↓
                                    回放缓冲区 → 学习器 (TD3+BC)
```

### 关键维度

| 组件 | 输入形状 | 输出形状 |
|-----------|------------|--------------|
| VLA | 观测 | `[50, 32]` 动作 + `[N, 2048]` 嵌入 |
| RLTEncoder | `[N, 2048]` | `[2048]` RL Token |
| 状态 | RL Token `[2048]` + 本体感受 `[32]` | `[2080]` |
| Actor | 状态 `[2080]` + 参考动作 `[1600]` | `[320]` = 10×32 |
| Critic | 状态 `[2080]` + 动作 `[320]` | `[1]` Q 值 |

---

## 超参数

### Stage 1
| 参数 | 值 |
|-----------|-------|
| 编码器深度 | 4 |
| 编码器宽度 | 2048 |
| 编码器注意力头数 | 8 |
| 学习率 | 1e-4 |
| 批次大小 | 32 |
| 训练步数 | 30,000 |

### Stage 2
| 参数 | 值 |
|-----------|-------|
| Actor 隐藏层 | 512 |
| Critic 隐藏层 | 256 |
| Actor 学习率 | 3e-4 |
| Critic 学习率 | 3e-4 |
| 折扣因子 Gamma | 0.99 |
| BC 正则化 Beta | 2.0 |
| 策略延迟 | 2 |
| 批次大小 | 256 |
| 每步更新次数 | 20 |
| Polyak tau | 0.005 |
| 平滑噪声 | 0.2 |
| 噪声裁剪 | 0.5 |

---

## 框架依赖检查

所有新增的 RLT 代码均为**纯 JAX / Flax NNX 实现，零 PyTorch 依赖**，与现有 openpi 代码库一致。

| 文件 | 框架 | 是否与 openpi 一致 |
|------|------|:---:|
| `rl_token.py` | `flax.nnx` + `jax.numpy` | ✅ |
| `rlt_policy.py` | `jax` + `jax.numpy` | ✅ |
| `rlt_online_rl/actor.py` | `flax.nnx` + `jax` | ✅ |
| `rlt_online_rl/critic.py` | `flax.nnx` + `jax.numpy` | ✅ |
| `rlt_online_rl/learner.py` | `flax.nnx` + `jax` + `optax` | ✅ |
| `rlt_online_rl/rollout.py` | `jax` + `jax.numpy` | ✅ |
| `rlt_online_rl/replay.py` | `numpy`（纯数据） | ✅ |

所有神经网络模块均继承自 `flax.nnx.Module`，优化器使用 `optax`，训练流程运行在 JAX 的函数式变换系统之上。

---

## 验证

```bash
# 1. 模块导入
python -c "from openpi.models.rl_token import RLTEncoder; print('OK')"

# 2. 配置加载
python -c "from openpi.training.config import get_config; cfg = get_config('rlt_debug')"

# 3. RLT 策略创建
python -c "from openpi.policies.rlt_policy import RLTPolicy; print('OK')"

# 4. 在线 RL 模块
cd rlt_online_rl && python -c "
from rlt_online_rl.actor import Actor;
from rlt_online_rl.critic import Critic;
from rlt_online_rl.replay import ReplayBuffer;
print('All OK')
"
```
