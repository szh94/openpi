# 项目代码结构与功能分析

> 分析 `scripts/`、`src/`、`rlt_online_rl/` 中的文件对应 PI0/PI0.5 或 RLT 的训练和推理中的功能和环节。

---

## 整体架构概览

本项目包含**两条主要 pipeline**：

1. **PI0 / PI0.5 pipeline** — 标准的模仿学习（Behavior Cloning）训练 + 推理部署
2. **RLT pipeline** — 两阶段的在线强化学习微调（Stage 1: RL Token 训练, Stage 2: TD3 + BC 在线 RL）

---

## 一、`scripts/` — 训练/推理/工具入口

| 文件 | 对应环节 | 说明 |
|---|---|---|
| **`train.py`** | PI0/PI0.5 **监督训练** | 标准的 BC 训练入口。加载配置 -> 初始化模型 -> 数据加载 -> train loop（`compute_loss` -> 梯度更新 -> 日志/保存）。适用于 pi0、pi0.5 的 fine-tuning。 |
| **`train_pytorch.py`** | PI0/PI0.5 **PyTorch 训练** | 同上，但使用 PyTorch 后端（通过 `pi0_pytorch.py`）。支持单卡/DDP/多节点。 |
| **`serve_policy.py`** | PI0/PI0.5 **推理服务** | 加载训练好的 checkpoint，启动 WebSocket 策略服务器，监听端口接收观测返回动作。 |
| **`compute_norm_stats.py`** | **数据预处理** | 遍历数据集计算 state/action 的均值、标准差、分位数等归一化统计量，保存到 assets 目录。**训练前必须先跑这个。** |
| **`train_rlt_token.py`** | RLT **Stage 1** | RL Token Encoder-Decoder 的**重建训练**。加载冻结的 VLA -> 初始化 Encoder+Decoder -> 以 MSE 重建 loss 训练 Encoder 压缩 VLA embedding 为单个 RL Token。训练后冻结 Encoder，丢弃 Decoder。 |
| **`train_rlt_online.py`** | RLT **Stage 2** | **在线 RL 微调**。加载冻结 VLA + 冻结 RLTEncoder -> 初始化 Actor + Critic + Replay Buffer -> 运行 TD3 + BC 在线 RL loop。 |
| **`serve_rlt_policy.py`** | RLT **推理服务** | 加载冻结 VLA + 冻结 RLTEncoder + 训练好的 Actor，组合成 `RLTPolicy`，启动 WebSocket 服务。 |
| **`train_test.py`** | **测试脚本** | 用于验证训练流程正确性。 |

---

## 二、`src/openpi/` — 核心库

### 2.1 `src/openpi/models/` — 模型定义

| 文件 | 对应环节 | 说明 |
|---|---|---|
| **`model.py`** | **基类 + 数据结构** | 定义 `BaseModel`（`compute_loss` + `sample_actions` 抽象接口）、`Observation` 数据结构（images/state/tokenized_prompt）、`restore_params()` 加载 checkpoint。PI0 和 PI0.5 都继承自 `BaseModel`。 |
| **`pi0.py`** | **核心模型** | `Pi0` 类：PI0 和 PI0.5 共享的模型实现。关键方法：`compute_loss()`（flow matching loss）、`sample_actions()`（反向扩散采样）、`embed_prefix()`（编码图像+语言）、`embed_suffix()`（编码带噪动作+时间步）、`get_token_embeddings()`（RLT 专用：提取 VLA 所有 token embedding 供 Encoder 压缩）。PI0 vs PI0.5 的区别通过 `config.pi05` 标志控制。 |
| **`pi0_config.py`** | **配置** | `Pi0Config`：定义 action_dim=32, action_horizon=50, paligemma_variant, action_expert_variant, pi05 开关等。 |
| **`pi0_fast.py`** | **pi0-FAST 模型** | 自回归 VLA 变体，使用 FAST action tokenizer。 |
| **`gemma.py`** | **语言模型** | Gemma LLM 的 JAX 实现（PaliGemma backbone）。支持 adaRMSNorm（PI0.5 特性）。 |
| **`siglip.py`** | **视觉编码器** | SigLIP So400m/14 视觉 Transformer。 |
| **`tokenizer.py`** | **分词器** | PaliGemma 分词器 + FAST tokenizer。 |
| **`rl_token.py`** | **RLT 核心组件** | `RLTEncoder`：4-layer Transformer，在 VLA embedding 序列末尾添加可学习的 `[RL]` token，取该位置输出作为压缩后的 RL Token（[B, 2048]）。`RLTDecoder`：从 RL Token 通过 cross-attention 重建原始 embedding 序列。**这是 RLT Stage 1 的训练目标和 Stage 2 的特征提取器。** |
| **`lora.py`** | **LoRA 微调** | 低秩适配器，用于低显存 fine-tuning。 |

### 2.2 `src/openpi/policies/` — 策略封装（推理层）

| 文件 | 对应环节 | 说明 |
|---|---|---|
| **`policy.py`** | **策略基类** | `Policy` 类封装了推理流程：输入 transform -> 模型 `sample_actions()` -> 输出 transform。 |
| **`policy_config.py`** | **策略工厂** | `create_trained_policy()`：加载 checkpoint -> 创建模型 -> 组装 transforms（归一化/图像resize/分词等） -> 返回 `Policy` 对象。 |
| **`rlt_policy.py`** | **RLT 推理** | `RLTPolicy`：流水线是 VLA(冻结) -> `get_token_embeddings()` -> all_embeddings -> RLTEncoder(冻结) -> RL Token -> concat proprio -> Actor -> 修正后的 action chunk。支持有/无 Actor 两种模式。 |
| **`aloha_policy.py`** | ALOHA 策略 | ALOHA 机器人平台的数据格式转换（joint space <-> pi internal space, delta actions 等）。 |
| **`droid_policy.py`** | DROID 策略 | DROID 平台的数据格式转换。 |
| **`libero_policy.py`** | LIBERO 策略 | LIBERO 基准的数据格式转换。 |

### 2.3 `src/openpi/training/` — 训练基础设施

| 文件 | 对应环节 | 说明 |
|---|---|---|
| **`config.py`** | **配置中心** | `_CONFIGS` 列表定义了所有可用的训练配置（pi0_libero, pi05_droid, rlt_debug 等）。每个 `TrainConfig` 包含模型配置、数据配置、优化器、学习率调度、权重加载器、冻结过滤器等。通过 `get_config(name)` 按名获取。 |
| **`data_loader.py`** | **数据加载** | LeRobot 数据集和 RLDS 数据集的加载器。 |
| **`checkpoints.py`** | **检查点管理** | 保存/恢复训练状态，支持 Orbax checkpointing。 |
| **`optimizer.py`** | **优化器** | AdamW + cosine decay schedule + 梯度裁剪等。 |
| **`sharding.py`** | **分布式分片** | FSDP 分片策略。 |
| **`weight_loaders.py`** | **权重加载** | 从预训练 checkpoint 加载（部分）权重。 |
| **`droid_rlds_dataset.py`** | DROID 数据集 | DROID RLDS 格式数据集的专用处理。 |

### 2.4 `src/openpi/shared/` — 共享工具

| 文件 | 对应环节 | 说明 |
|---|---|---|
| **`normalize.py`** | 数据归一化 | RunningStats 计算、保存/加载 norm_stats。 |
| **`image_tools.py`** | 图像处理 | resize_with_pad 等。 |
| **`download.py`** | 下载工具 | 自动从 GCS 下载 checkpoint/assets。 |

### 2.5 `src/openpi/serving/` — 推理服务

| 文件 | 对应环节 | 说明 |
|---|---|---|
| **`websocket_policy_server.py`** | **推理部署** | WebSocket 策略服务器，接收观测、调用 policy.infer()、返回动作。 |

---

## 三、`rlt_online_rl/` — RLT 在线 RL 模块

这是**独立于 `src/` 的包**，对应 RLT 论文的 Stage 2 在线 RL 组件：

| 文件 | 对应环节 | 说明 |
|---|---|---|
| **`actor.py`** | **Actor 网络** | 输入 = `[RL Token(2048) + proprio(32), ref_actions_flat(1600)]` -> `[3680]` -> MLP(512->512) -> 输出 `mean[320] + log_std[320]`（C=10步 x 32维）。Loss = `-Q + β * BC(a, ref_actions)`。 |
| **`critic.py`** | **Critic 网络 (TD3)** | 双 Critic（Twin Q-networks）：输入 = `[state(2080) + action_chunk(320)]` -> `[2400]` -> MLP(256->256) -> Q 值。包含 target networks + Polyak soft update。 |
| **`learner.py`** | **TD3 + BC 训练器** | 核心训练 loop：从 Replay Buffer 采样 -> critic 更新（TD loss）-> 延迟 actor 更新（policy_delay=2）-> target network soft update。 |
| **`replay.py`** | **经验回放缓冲** | 循环数组，存储 `(state[2080], action[320], reward, next_state[2080], done)`。stride=2 提高样本效率。 |
| **`rollout.py`** | **数据收集** | 机器人与环境交互 loop：VLA 推理 -> Encoder 压缩 -> Actor 修正 -> 执行 C=10 步动作 -> 收集 reward -> 存入 Replay Buffer。 |

---

## 四、两条 Pipeline 的完整数据流

### PI0/PI0.5 训练 -> 推理

```
[训练]
scripts/compute_norm_stats.py  →  计算数据集归一化统计量
       ↓
scripts/train.py               →  加载预训练权重 → 数据加载 → flow matching loss → 梯度更新 → 保存 checkpoint
       ↓
[推理]
scripts/serve_policy.py        →  加载 checkpoint → Policy(model + transforms) → WebSocket 服务
       ↓
policy.infer(obs)              →  transforms → embed_prefix(图像+文本) → embed_suffix(带噪动作+t)
                                   → LLM 前向 → action_out_proj → 反向扩散采样(while_loop) → 动作
```

### RLT 两阶段训练 -> 推理

```
[Stage 1: RL Token 训练]
scripts/train_rlt_token.py     →  加载冻结 VLA → 初始化 Encoder+Decoder
                                   → VLA.get_token_embeddings() → Encoder 压缩为 [RL] Token
                                   → Decoder 重建 → MSE loss → 训练 Encoder → 冻结 Encoder
       ↓
[Stage 2: 在线 RL 微调]
scripts/train_rlt_online.py    →  加载冻结 VLA + 冻结 RLTEncoder → 初始化 Actor+Critic+Replay
       ↓
RolloutRunner.run_episode()    →  VLA推理 → RLTEncoder → RL Token → Actor修正 → 执行10步 → Replay
       ↓
Learner.update()               →  采样 → Critic TD loss → Actor -Q+β*BC loss → Polyak update
       ↓
[推理]
scripts/serve_rlt_policy.py    →  加载 VLA + RLTEncoder + Actor → RLTPolicy → WebSocket 服务
       ↓
RLTPolicy.infer(obs)           →  VLA.get_token_embeddings() → RLTEncoder(all_embeddings) → RL Token
                                   → concat proprio → Actor(state, ref_actions) → 修正后的动作
```

---

## 五、关键要点总结

### PI0 vs PI0.5 的区别（在 `pi0.py` 中通过 `self.pi05` 标志控制）

- PI0.5 使用 `discrete_state_input`（state 作为离散语言 token 输入，而非连续 suffix 输入）
- PI0.5 使用 adaRMSNorm 注入 flow matching 时间步信息（`adarms_cond`），PI0 则是 concat action 和 time embedding 后过 MLP
- PI0.5 的 `max_token_len` 默认 200（PI0 默认 48），因为 state 被 tokenize 了
- PI0.5 使用分位数归一化（quantile norm），PI0 使用标准 z-score 归一化

### RLT 的核心思想

- 用一个小型 Transformer Encoder（`RLTEncoder`）将 VLA 的所有 token embedding 压缩为单个 RL Token（2048 维）
- 这个 RL Token 捕获了 VLA 对整个观测的理解（图像+语言+动作上下文），作为 RL 策略的状态表示
- Stage 1 通过自重建（auto-encoding）训练 Encoder，Stage 2 冻结 Encoder，在 RL Token 之上训练 Actor-Critic
- Actor 的输出是对 VLA 参考动作的**修正**，Loss 包含 `-Q` 项（强化）和 `β·BC` 项（约束在 VLA 附近）
