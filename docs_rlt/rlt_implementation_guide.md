# RLT (RL Token) 实现指南 — 基于 openpi 代码库

> 基于 Physical Intelligence 论文 *RL Token: Bootstrapping Online RL with Vision-Language-Action Models* (2026)

---

## RLT 核心思想（一句话）

> **冻住约 48.6 亿参数的 VLA 大模型（pi0.6），外挂一个 2-3 层 MLP 的 Actor-Critic 做在线 RL 微调**

核心创新是 **RL Token**：一个 Encoder-Decoder Transformer，把 VLA 最后一层输出的 N×2048 维 token 序列，压缩成一个 2048 维的"状态向量"，喂给轻量 Actor-Critic。

---

## 1. 需要新增的文件

### 1.1 模型层

#### `src/openpi/models/rl_token.py` — RL Token Encoder-Decoder

```python
class RLTokenEncoder(nn.Module):
    """把 VLA 的 N×2048 token embeddings 压缩为单个 2048 维 RL Token"""

    def __call__(self, embeddings: [N, 2048], special_token) -> [2048]:
        # 输入: VLA 所有 token embeddings + 可学习 special token
        # 输出: special token 位置 → 2048 维 RL Token
        ...


class RLTokenDecoder(nn.Module):
    """从 RL Token 自回归重建原始 embeddings（仅在训练时使用）"""

    def __call__(self, rl_token: [2048]) -> [N, 2048]:
        # 训练时: 从 RL Token 重建原始 embeddings
        # 损失: MSE(重建, 原始) — 类似 VAE 瓶颈
        ...
```

**训练损失（Stage 1）**：
```
L_ro(ϕ) = E[ Σ_i ‖h_ϕ(d_ϕ([z_rl, ẑ₁:i-1]))_i - ẑ_i‖² ]
```

- VLA 冻结（stop-gradient）
- Encoder ϕ 和 Decoder ϕ 共享参数
- 训练完成后 **VLA + Encoder + Decoder 全部冻结**

---

### 1.2 策略层

#### `src/openpi/policies/rlt_policy.py` — RLT 策略包装器

```python
class RLTPolicy:
    """RLT 推理策略: VLA → RL Token → Actor 修正 → 执行动作"""

    def __init__(self, vla, encoder, actor):
        self.vla = vla       # 冻结
        self.encoder = encoder  # 冻结
        self.actor = actor   # 在线训练

    def act(self, obs) -> actions:
        # 1. 冻结的 VLA 生成参考动作 + token embeddings
        ref_actions, embeddings = self.vla(obs)
        # 2. Encoder 压缩为 RL Token
        rl_token = self.encoder(embeddings)
        # 3. Actor 输出修正后的动作
        state = (rl_token, obs.state)         # RL Token + 本体感受
        corrected = self.actor(state, ref_actions)  # Actor 修正
        return corrected
```

---

### 1.3 在线 RL 训练模块

建议放在独立目录 `rlt_online_rl/` 下，与现有训练系统解耦：

| 文件 | 说明 |
|---|---|
| `rlt_online_rl/actor.py` | **Actor 网络**：2-3 层 MLP，输入 (rl_token + state + ref_actions)，输出修正动作 |
| `rlt_online_rl/critic.py` | **Critic 网络**：2-3 层 MLP，输入 (rl_token + state + action_chunk)，输出 Q 值（TD3 双 Critic） |
| `rlt_online_rl/learner.py` | **TD3 + BC 训练循环**：从 Replay Buffer 采样 → 更新 Critic → 更新 Actor |
| `rlt_online_rl/replay.py` | **Replay Buffer**：存 (state, action, reward, next_state) |
| `rlt_online_rl/rollout.py` | **Rollout 运行时**：机器人交互、奖励计算、数据收集 |

#### Actor 网络

```python
class Actor(nn.Module):
    """输入状态 + VLA 参考动作，输出修正后的动作"""
    # 2 层 MLP（256 隐藏层）或 3 层 MLP（512 隐藏层）

    def forward(self, state, ref_actions):
        x = concat([state, ref_actions])
        x = MLP(x)          # 2-3 层
        mean = x            # 高斯分布的均值（修正后的动作）
        std = softplus(x)   # 高斯分布的标准差
        return mean, std
```

**Actor 损失**：
```
L_π(θ) = E[-Q_ψ(x, a₁:C) + β·‖a₁:C - \tilde{a}₁:C‖²]
```
- 第一项：最大化 Q 值
- 第二项：BC 正则化，不偏离 VLA 参考动作太远

#### Critic 网络

```python
class Critic(nn.Module):
    """评估状态-动作对的价值（TD3 双 Critic 取最小值）"""
    # 2-3 层 MLP

    def forward(self, state, action_chunk):
        x = concat([state, action_chunk.flatten()])
        return MLP(x)  # Q 值
```

**Critic 损失**（标准 TD3）：
```
L_Q(ψ) = E[(ˆQ - Q_ψ(x, a₁:C))²]
ˆQ = Σ γ^{t'-1} r_{t'} + γ^C · Q_ψ'(x', π_θ(x'))
```

#### 关键训练技巧

| 技巧 | 说明 |
|---|---|
| **Action Chunk C=10** | VLA 预测 H=50 步，Actor 决策 C=10 步，horizon 压缩 5 倍 |
| **参考动作 Dropout** | 50% 概率将 ref_actions 置零，防止 Actor 偷懒复制 VLA |
| **BC 正则化 β** | 控制 Actor 偏离 VLA 的程度 |
| **Stride=2** | 重叠存储 transition，样本效率提高 5 倍 |
| **异步架构** | Rollout 进程 + Learner 进程独立运行 |

---

### 1.4 训练入口

| 文件 | 说明 |
|---|---|
| `scripts/train_rlt_token.py` | **Stage 1**：在 demo 数据上训练 RL Token Encoder-Decoder（冻住 VLA） |
| `scripts/train_rlt_online.py` | **Stage 2**：启动在线 RL（异步 rollout + learner） |

---

### 1.5 部署

| 文件 | 说明 |
|---|---|
| `scripts/serve_rlt_policy.py` | RLT 策略服务（冻结 VLA + Encoder + Actor） |

---

## 2. 需要修改的现有文件

| 文件 | 修改内容 | 优先级 |
|---|---|---|
| `src/openpi/models/pi0.py` | **新增方法暴露 VLA 中间 token embeddings**（当前 `__call__` 只返回 action tokens，不输出 N×2048 的中间表示） | **必须** |
| `src/openpi/models/model.py` | 新增 `RLTBaseModel` / `RLTConfig` 基类 | 推荐 |
| `src/openpi/models/pi0_config.py` | 新增 RLT 配置项（RL Token 维度、encoder/decoder 层数、action chunk C 等） | 推荐 |
| `src/openpi/policies/policy.py` | 注册 RLT 策略类型 | 需要 |
| `src/openpi/training/config.py` | 注册 RLT 训练配置 | 需要 |
| `scripts/serve_policy.py` | 支持 RLT 策略服务 | 可选 |

### `pi0.py` 最关键的修改

当前代码中，`compute_loss` 和 `sample_actions` 只输出了最终的 action tokens。RLT 需要拿到 **VLA 最后一层所有 token 的 embeddings**（不只是 action 部分）。

```python
# pi0.py — 需要新增的方法
class Pi0:
    def get_token_embeddings(self, observation) -> tuple[Array, Array]:
        """返回 VLA 最后一层的 token embeddings + 参考动作"""
        # embed_prefix 产生图片+文本 tokens
        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        # embed_suffix 产生 action tokens（用随机噪声占位，因为不需要去噪）
        dummy_noise = jnp.zeros((batch, self.action_horizon, self.action_dim))
        suffix_tokens, suffix_mask, _, adarms_cond = self.embed_suffix(
            observation, dummy_noise, jnp.ones(batch)
        )
        # 一次前向传播得到所有 token 的输出
        all_tokens = jnp.concatenate([prefix_tokens, suffix_tokens], axis=1)
        all_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=..., positions=..., adarms_cond=[None, adarms_cond]
        )
        # 返回所有 token embeddings (N×2048) + 参考动作
        return jnp.concatenate([prefix_out, suffix_out], axis=1), suffix_out[:, -self.action_horizon:]
```

---

## 3. 完整流程图

### 3.1 PI0.5 现有推理流程（sample_actions）

```
Observation 输入
┌────────────────────────────────────────────────────────────────┐
│ images:         dict[str, [b, 224, 224, 3]]                    │
│ tokenized_prompt:  [b, L_text]  (L_text ≤ 200, 含离散化 state)│
│ state:         [b, 32]        (仅 PI0 使用, PI0.5 已离散化)    │
└────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   embed_prefix()       │
                    │   (pi0.py:106-137)     │
                    └───────────┬───────────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
    SigLIP ViT           Gemma embedder      (PI0.5: state 已
    (siglip.py)          (gemma.py:354)      在 tokenized_prompt
           │                    │             中作为文本 token)
           ▼                    ▼
    image_tokens          text_tokens
    [b, 256, 2048]        [b, L_text, 2048]
    × 3 cameras
    ─────────────
    [b, 768, 2048]         [b, ≤200, 2048]
           │                    │
           └────────┬───────────┘
                    ▼
           prefix_tokens = concat
           [b, ~968, 2048]       ← Expert 0 (PaliGemma 2B) 的输入
                                 ← 图片 768 + 文本 ≤200
                                 ← 双向注意力 (ar_mask=False)


  noisy_actions (推理入口: 纯噪声)
  [b, 50, 32]
        │
        ▼
  action_in_proj                   time步 t (从 1.0 → 0.0 去噪)
  Linear(32 → 1024)                │
        │                          │
        ▼                          ▼
  action_embeds               time_emb (posemb_sincos)
  [b, 50, 1024]               [b, 1024]
        │                          │
        │                          ▼
        │                    time_mlp_in → swish → time_mlp_out → swish
        │                    [b, 1024]  (PI0.5 特有, adaRMSNorm 条件)
        │                          │
        └──────────┬───────────────┘
                   ▼
          suffix_tokens
          [b, 50, 1024]         ← Expert 1 (Action Expert 300M) 的输入
                                 ← 因果注意力 (ar_mask=[1,0,0,...])

                   │
                   ▼
     ┌───────────────────────────────────────┐
     │  Gemma Transformer (18 层)            │
     │  (gemma.py:340-411)                   │
     │                                       │
     │  prefix_tokens [b, 968, 2048]         │
     │      → Expert 0 (PaliGemma 权重)      │
     │      → 双向注意力                     │
     │                                       │
     │  suffix_tokens [b, 50, 1024]          │
     │      → Expert 1 (Action Expert 权重)  │
     │      → 因果注意力                     │
     │      → adaRMSNorm 受 time_emb 调制    │
     │                                       │
     │  两个专家共享 Attention 层             │
     │  QKV 拼在一起算注意力再拆开            │
     └──────────────────┬────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  prefix_out (Expert 0)       suffix_out (Expert 1)
  [b, ~968, 2048]            [b, 50, 1024]
                                         │
                                         ▼
                                  action_out_proj
                                  Linear(1024 → 32)
                                         │
                                         ▼
                                  flow matching 速度 v_t
                                  [b, 50, 32]
                                         │
                                  Euler 积分 x_t + dt·v_t
                                  循环 10 步 (t=1.0 → t=0.0)
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │ 预测动作 x₀  │
                                  │ [b, 50, 32]  │
                                  └─────────────┘
```

### 3.2 PI0.6 + RLT 推理流程（冻结 VLA + Actor 修正）

```
  Observation 输入 (与 PI0.5 相同)
  images [b, 224, 224, 3] × 3, text tokens [b, ≤200]
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │               PI0.6 VLA (全部冻结)                    │
  │                                                       │
  │  1. embed_prefix → prefix_tokens [b, ~968, 2048]    │
  │  2. Transformer 前向 (prefix + suffix)               │
  │  3. 输出:                                              │
  │     prefix_out: [b, ~968, 2048]   (Expert 0)        │
  │     suffix_out: [b, 50, 1024]     (Expert 1)        │
  │  4. action_out_proj → ref_actions [b, 50, 32]       │
  └──────────────────────┬───────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
  VLA 参考动作                    VLA 中间 token embeddings
  ref_actions                      ┌─────────────────────┐
  [b, 50, 32]                      │ prefix_out [b,968,2048]│
                                   │ suffix_out  [b,50,1024]│
                                   │ 注: 两个 expert 宽度   │
                                   │ 不同, 需要统一维度     │
                                   │ (project 或 concat     │
                                   │  后统一处理)           │
                                   └──────────┬──────────┘
                                              │
                                              ▼
                              ┌───────────────────────────┐
                              │  RL Token Encoder (冻结)   │
                              │  (rl_token.py)            │
                              │                           │
                              │  输入: [b, ~1018, 2048]   │
                              │         (统一维度后)       │
                              │                           │
                              │  [z₁:M, e_rl] → Encoder   │
                              │                           │
                              │  输出: RL Token z_rl      │
                              │         [b, 2048]         │
                              └───────────┬───────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          │ State x = (z_rl, proprio)     │
                          │ z_rl:    [b, 2048]            │
                          │ proprio: [b, 32]              │
                          │ (注: proprio = robot state)   │
                          └───────────────┬───────────────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │               在线 RL (仅训练这部分)                    │
              │                                                       │
              │  ┌─────────────────────────────────────┐              │
              │  │ Actor π_θ (2-3 层 MLP)              │              │
              │  │                                     │              │
              │  │ 输入:                               │              │
              │  │   state:    [b, 2048 + 32]          │              │
              │  │   ref_actions: [b, 50, 32]          │              │
              │  │          (flatten → [b, 1600])      │              │
              │  │                                     │              │
              │  │ 输出: 修正后动作 chunk a₁:C         │              │
              │  │   C=10 (flat: [b, 10×32=320])       │              │
              │  │                                     │              │
              │  │ 损失: -Q + β·‖a - ref‖²             │              │
              │  └──────────────┬──────────────────────┘              │
              │                 │                                     │
              │  ┌──────────────┴──────────────────────┐              │
              │  │ Critic Q_ψ (2-3 层 MLP, ×2 TD3)    │              │
              │  │                                     │              │
              │  │ 输入:                               │              │
              │  │   state:    [b, 2048 + 32]          │              │
              │  │   action chunk a₁:C                 │              │
              │  │          (flat: [b, 320])           │              │
              │  │                                     │              │
              │  │ 输出: Q 值 [b, 1]                   │              │
              │  │                                     │              │
              │  │ 损失: TD error (标准 TD3)           │              │
              │  └─────────────────────────────────────┘              │
              │                                                       │
              └───────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │ 执行动作块 a₁:C  │
                          │ 执行 C=10 步后   │
                          │ 获取 reward +    │
                          │ 下一 obs         │
                          └─────────────────┘
```

### 3.3 Shape 变化对照表

| 阶段 | 变量 | Shape | 说明 |
|---|---|---|---|
| **输入** | images | `[b, 224, 224, 3]` × 3 | 3 个摄像头, 归一化到 [-1, 1] |
| | tokenized_prompt | `[b, ≤200]` | 文本指令 token IDs |
| | state (proprio) | `[b, 32]` | 机器人关节角度 (PI0.5 离散化后进 prefix) |
| **embed_prefix** | image_tokens | `[b, 256, 2048]` × 3 = `[b, 768, 2048]` | SigLIP 输出, 224/14=16, 16²=256 |
| | text_tokens | `[b, L_text, 2048]` | Gemma embedder 输出 |
| | → prefix_tokens | `[b, ~968, 2048]` | 768 + L_text |
| **embed_suffix** | noisy_actions | `[b, 50, 32]` | 含噪动作 (推理开始时是纯噪声) |
| | action_embeds | `[b, 50, 1024]` | action_in_proj Linear(32→1024) |
| | time_emb | `[b, 1024]` | posemb_sincos + time_mlp |
| | → suffix_tokens | `[b, 50, 1024]` | 动作 tokens, 因果注意力 |
| **Transformer** | prefix_out | `[b, ~968, 2048]` | Expert 0 (PaliGemma 2B) 输出 |
| | suffix_out | `[b, 50, 1024]` | Expert 1 (Action Expert 300M) 输出 |
| **Flow Matching** | v_t | `[b, 50, 32]` | action_out_proj Linear(1024→32) |
| | x₀ | `[b, 50, 32]` | 去噪 10 步后得到的干净动作 |
| **RLT 新增** | | | |
| | token_embeddings | `[b, ~1018, 2048/1024]` | 两 expert 宽度不同, 需统一处理 |
| | RL Token z_rl | `[b, 2048]` | Encoder 压缩后的状态表示 |
| | 状态 x | `[b, 2080]` | z_rl(2048) + proprio(32) |
| | ref_actions | `[b, 1600]` | VLA 输出 50×32, flatten 后喂 Actor |
| | Actor 输出 a₁:C | `[b, 320]` | C=10 步, 每步 32 维, flatten |
| | Q 值 | `[b, 1]` | Critic 输出标量 |

> **关于 action_dim=32**：这是 VLA 预训练时对所有机器人动作空间取"最大公约数"的固定维度（见 pi0.5 论文）。RLT 原论文使用双臂 6-DOF 机器人（6+1+6+1=14 维），实际动作为 14 维。openpi 代码中 action_dim=32 是模型内部维度，低维机器人的多余维度会被填充为 0。具体使用时应只取你机器人对应的维度。

---

## 4. 简明架构对比图 — 原始 VLA vs RLT 新增模块

新手直接看这张图就够了。左侧是基础 VLA（pi0.5/pi0.6），右侧是 RLT 新增的部分（标 `+`）。

> **注意**：RLT 原论文使用 pi0.6，我们的实现基于 pi0.5。两种模型的架构原理相同，RLT 的适用不依赖具体版本。

### 4.1 宏观对比

```
原始 VLA                                  VLA + RLT（在线 RL）
┌─────────────────────┐                  ┌─────────────────────────────────┐
│                     │                  │                                 │
│  Observation        │                  │  Observation                    │
│  (images + text)    │                  │  (images + text)                │
│         │           │                  │         │                       │
│         ▼           │                  │         ▼                       │
│  ┌─────────────┐    │                  │  ┌─────────────┐               │
│  │   VLA 模型  │    │                  │  │   VLA 模型  │ （冻结 ✅）    │
│  │  (~5B params)│   │                  │  │  (~5B params)│              │
│  │             │    │                  │  │             │               │
│  │  输出动作   │    │                  │  │  输出:      │               │
│  │  [50, 32]   │    │                  │  │  ① 动作 [50,32]            │
│  └─────────────┘    │                  │  │  ② 中间 token embeddings   │
│         │           │                  │  │     [~1018, 2048/1024]     │
│         ▼           │                  │  └──────┬────────────────────┘
│  ┌─────────────┐    │                  │         │
│  │ 执行动作     │    │                  │  ┌──────┴────────────────────┐
│  └─────────────┘    │                  │  │ + RL Token Encoder (冻结) │
│                     │                  │  │   压缩 N tokens → 1 向量   │
│  ✅ 直接执行 VLA     │                  │  │   [~1018, 2048] → [2048]  │
│     预测的动作，     │                  │  └──────┬────────────────────┘
│     不做任何修正     │                  │         │
│                     │                  │  ┌──────┴────────────────────┐
│                     │                  │  │ + State = z_rl + proprio │
│                     │                  │  │   [2048] + [32] = [2080]  │
│                     │                  │  └──────┬────────────────────┘
│                     │                  │         │
│                     │                  │  ┌──────┴────────────────────┐
│                     │                  │  │ + Actor π_θ（在线训练）    │
│                     │                  │  │   输入: state [2080]      │
│                     │                  │  │        + ref_actions      │
│                     │                  │  │   输出: 修正动作 [10, 32] │
│                     │                  │  └──────┬────────────────────┘
│                     │                  │         │
│                     │                  │  ┌──────┴────────────────────┐
│                     │                  │  │ + Critic Q_ψ（在线训练）   │
│                     │                  │  │   评估动作价值 → [1] 标量 │
│                     │                  │  └──────┬────────────────────┘
│                     │                  │         │
│                     │                  │  ┌──────┴────────────────────┐
│                     │                  │  │ + Replay Buffer           │
│                     │                  │  │   (存交互数据, 供 RL 训练) │
│                     │                  │  └───────────────────────────┘
│                     │                  │
│                     │                  │  ✅ VLA 不动，外挂小模块
│                     │                  │     通过在线 RL 微调动作
└─────────────────────┘                  └─────────────────────────────────┘
```

### 4.2 RLT 新增模块详解

```
┌─────────────────────────────────────────────────────────────────────┐
│ + 模块 1: RL Token Encoder                                          │
│                                                                     │
│  输入: VLA 最后一层所有 token embeddings                             │
│        prefix_out [b, ~968, 2048]  ← 图像 + 文本 tokens (Expert 0) │
│        suffix_out [b, 50, 1024]    ← 动作 tokens (Expert 1)        │
│        ─────────────────────────────────────────────────            │
│        统一投影到 2048 维后拼接 → [b, ~1018, 2048]                  │
│                                                                     │
│  处理: Transformer Encoder（4 层）                                   │
│        在输入末尾加 special token [RL]                               │
│        所有 token 做双向自注意力                                     │
│                                                                     │
│  输出: [b, 2048]  ← 取 [RL] 位置的输出，即"RL Token"                │
│                                                                     │
│  Stage 1 还有个 Decoder 用于重建训练（训练完就冻结, 推理不用）       │
│  重建: RL Token → Decoder → [b, ~1018, 2048]                       │
│  损失: MSE(重建, 原始)                                              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────────┐
│ + 模块 2: Actor π_θ（2-3 层 MLP, 在线训练）                         │
│                                                                     │
│  输入: concat(state, ref_actions)                                   │
│        state:      [b, 2080]    ← RL Token [2048] + proprio [32]   │
│        ref_actions: [b, 1600]   ← VLA 输出的 50×32, flatten        │
│        ─────────────────────────────────────────────────            │
│        拼接:      [b, 3680]                                          │
│                                                                     │
│  网络: Linear(3680 → 512) → ReLU → Linear(512 → 512) → ReLU        │
│        → Linear(512 → 320) → 输出修正动作均值 μ                      │
│  （RLT 论文中简单任务用 2 层 256，复杂任务用 2-3 层 512）            │
│        → softplus → 输出标准差 σ                                     │
│                                                                     │
│  输出: action_chunk a₁:C  [b, 320]  ← C=10 步, 每步 32 维, flatten │
│        动作 = N(μ, σ)  在推理时直接取 μ                             │
│                                                                     │
│  损失: L = -E[Q(s, a)] + β · ‖a - ref_actions‖²                    │
│         ↑ 最大化 Q 值     ↑ BC 正则化, 不偏离 VLA 参考动作太远      │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────────┐
│ + 模块 3: Critic Q_ψ（2-3 层 MLP, ×2 双网络 = TD3, 在线训练）      │
│                                                                     │
│  输入: concat(state, action_chunk)                                  │
│        state:       [b, 2080]  ← RL Token + proprio                 │
│        action_chunk: [b, 320]   ← Actor 输出的 C=10 步动作, flatten │
│        ─────────────────────────────────────────────────            │
│        拼接:       [b, 2400]                                         │
│                                                                     │
│  网络: Linear(2400 → 256) → ReLU → Linear(256 → 256) → ReLU        │
│        → Linear(256 → 1) → Q 值                                     │
│                                                                     │
│  输出: Q(s, a)  [b, 1]   ← 标量, 表示"这个状态-动作组合好/坏"      │
│        TD3 技巧: 两个 Critic 网络独立训练, 取 min(Q1, Q2)            │
│                                                                     │
│  损失: MSE(Q_pred, Q_target)  —— 标准 TD3                           │
│        Q_target = r + γ · min(Q1', Q2')(s', a')                     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────────┐
│ + 模块 4: Replay Buffer & Rollout 运行时                            │
│                                                                     │
│  Rollout 进程: 机器人交互, 收集数据                                  │
│  ┌──────────────────────────────────────────────────────┐           │
│  │ 1. VLA 推理 → 参考动作 (50 步)                       │           │
│  │ 2. Encoder → RL Token                                │           │
│  │ 3. Actor → 修正后动作 chunk (10 步)                  │           │
│  │ 4. 机器人执行 10 步, 收集 reward + 下一观测          │           │
│  │ 5. 将 transition 存入 Replay Buffer                  │           │
│  │    格式: (state_t, action_t, r_t, state_{t+C})       │           │
│  │                                                        │           │
│  │    Replay Buffer 中每条 transition 的 shape:           │           │
│  │    ┌─────────────────┬──────────┬─────────────────┐   │           │
│  │    │ 字段            │ Shape    │ 说明             │   │           │
│  │    ├─────────────────┼──────────┼─────────────────┤   │           │
│  │    │ state_t         │ [2080]   │ z_rl+proprio     │   │           │
│  │    │ action_t        │ [320]    │ a₁:C, C=10×32   │   │           │
│  │    │ reward_t        │ [1]      │ 标量奖励         │   │           │
│  │    │ state_{t+C}     │ [2080]   │ 下个状态         │   │           │
│  │    │ done_t          │ [1]      │ 是否终止 (bool)  │   │           │
│  │    └─────────────────┴──────────┴─────────────────┘   │           │
│  │                                                       │           │
│  │    Buffer 整体 shape（容量 N=1e6）:                    │           │
│  │      state:     [N, 2080]   ← 100 万条历史状态        │           │
│  │      action:    [N, 320]    ← 对应的动作              │           │
│  │      reward:    [N, 1]      ← 对应的奖励              │           │
│  │      next_state: [N, 2080]  ← 对应的下一状态          │           │
│  │      done:      [N, 1]      ← 对应的终止标志          │           │
│  │                                                       │           │
│  │ 6. 回到第 1 步, 重叠执行 (stride=2, 每 2 步存一次)   │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                     │
│  Learner 进程: 从 Replay Buffer 采样, 更新 Actor + Critic           │
│  ┌──────────────────────────────────────────────────────┐           │
│  │ 每收集 G 步数据后:                                    │           │
│  │ 1. 从 Buffer 随机采样 batch                          │           │
│  │ 2. 更新 Critic: 最小化 TD error                      │           │
│  │ 3. 更新 Actor: 最大化 Q + BC 正则化                  │           │
│  │ 4. 软更新目标网络 (polyak = 0.995)                   │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                     │
│  异步: Rollout 和 Learner 独立运行, 不互等                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 一句话总结每个模块的作用

| 模块 | 一句话作用 | 是否冻结 |
|---|---|---|
| **VLA（pi0.5/pi0.6）** | 基础视觉-语言-动作大模型，输出参考动作 | ✅ 冻结 |
| **RL Token Encoder** | 把 VLA 内部几千个 token 压缩成一个状态向量 | ✅ 冻结 |
| **RL Token Decoder** | Stage 1 训练 Encoder 用的"翻译器" (重建 token) | ✅ 冻结 (训练完即弃) |
| **Actor** | 参考动作上做小修正, 让它更"好"（Q 值更高） | ❌ 在线训练 |
| **Critic** | 判断"这个动作好不好", 指导 Actor 改进 | ❌ 在线训练 |
| **Replay Buffer** | 存机器人交互历史, 供 Critic+Actor 学习 | — |

---

## 5. 训练流程（两阶段）

### Stage 1: RL Token 训练（Offline）

```python
# scripts/train_rlt_token.py
1. 加载预训练 VLA（pi0.5/pi0.6），冻结所有参数
2. 初始化 Encoder-Decoder ϕ
3. 在任务 demo 数据上训练:
   for batch in dataloader:
       embeddings, _ = vla.get_token_embeddings(batch)
       rl_token = encoder(embeddings)
       reconstructed = decoder(rl_token)
       loss = MSE(reconstructed, embeddings)
       loss.backward()
       update(ϕ)
4. 冻结 ϕ
```

### Stage 2: 在线 RL 训练

```
循环 t = 0, C, 2C, ...:
  1. VLA 生成参考动作块 \tilde{a} ~ π_vla(s_t)
  2. Encoder 生成 RL Token: z_rl = encoder(z₁:M)
  3. 构成状态: x_t = (z_rl, s_t^p)
  4. Actor 选择动作: a ~ π_θ(·|x_t, \tilde{a})
  5. 执行 a₁:C，收集奖励 r_t，下一状态
  6. 存入 Replay Buffer B

内循环（每步 G 次更新）:
  1. 从 B 采样 batch
  2. TD3 更新 Critic: min L_Q(ψ)
  3. 更新 Actor: min -Q + β·BC_reg
```

#### 内循环详解

##### 前置：为什么内循环要跑 G 次更新？

这是 **off-policy** 算法的核心优势：数据收集（rollout）和学习（training）的频率可以解耦。

```
┌─ Rollout 进程（慢）────────────────┐
│ 机器人每走 10 步 → 存 1 条 transition │  ← 受物理世界速度限制
└────────────────────────────────────┘
                    ↓
┌─ Learner 进程（快）─────────────────┐
│ 每收集到 1 条 transition → 复用 G 次  │  ← G 通常 ≈ 10-50
│ 从 buffer 反复采样旧数据学习           │     一条数据用多次
└────────────────────────────────────┘
```

物理世界 1 秒只能走 ~10 步，但 GPU 1 秒能学 ~100 次。不让 Learner 闲着，提高样本效率。

---

##### Step 1：从 Replay Buffer 采样 batch

```
从 Buffer B 中均匀随机采样 N 条 transition（N 通常 256-1024）:

  sampled_batch = {
      state:       [N, 2080]    ← RL Token + proprio
      action:      [N, 320]     ← 当时 Actor 执行的动作 a₁:C
      reward:      [N, 1]       ← 执行后得到的标量奖励
      next_state:  [N, 2080]    ← 执行 C 步后的状态
      done:        [N, 1]       ← 是否终止
  }
```

**为什么要随机采样，而不是用最新数据？**

- **打破时间相关性**：连续采集的 transition 高度相关（前一步和后一步状态几乎一样），直接顺序学习会让网络"记住"局部模式而非真正理解任务。随机混洗解决了这个问题。
- **提高数据利用率**：旧数据仍然有价值，随机采样让每条数据能被复用多次。

---

##### Step 2：TD3 更新 Critic

Critic 的损失函数：

```
L_Q(ψ) = E[(ˆQ - Q_ψ(x, a₁:C))²]
```

这是一个 **TD（Temporal Difference）回归问题**：

- **预测值 Q_ψ(x, a)**：Critic 网络当前对"在状态 x 做动作 a 的未来累积奖励"的估计
- **目标值 ˆQ**: 用实际观察到的 reward + 对下一状态的估计来构造，公式：

```
ˆQ = r + γ · (1-done) · min(Q_ψ₁'(x', a'), Q_ψ₂'(x', a'))
   ↑                 ↑
   实际奖励          TD3 双 Critic 取最小值（防高估）
```

##### TD3 的三项关键技术

| 技术 | 做法 | 解决什么问题 |
|---|---|---|
| **Twin Critics（双 Critic）** | 独立训练两个 Critic 网络 Q₁, Q₂，计算目标时取 **min(Q₁, Q₂)** | 标准 DQN 会系统性高估 Q 值 → 策略学偏。双网络取最小值提供更保守的估计 |
| **Target Networks（目标网络）** | 用延迟更新的 ψ' 计算目标 ˆQ，而非用当前 ψ。每 K 步用 Polyak 平均软同步：`ψ' ← τ·ψ + (1-τ)·ψ'`，τ=0.005 | 防止"移动目标"问题——如果目标和预测用同一网络同步更新，会形成正反馈振荡 |
| **Target Policy Smoothing（目标平滑）** | 计算目标 Q 时，在下一个动作 a' 上加小噪声：`a' = π_θ'(x') + clip(ε, -c, c)`，ε ~ N(0, σ) | 防止 Critic 在动作空间的尖峰上过拟合，让它学会更平滑的 Q 面 |

**数据流图**：

```
sampled_batch
  ├→ (x, a, r, x', done)

  # ── 计算目标 Q 值（用目标网络，不计算梯度）──
  a' = target_actor(x')                        # 目标 Actor 给出下一动作
  a' = a' + clip(ε, -c, c)                     # 目标平滑（添加噪声）
  Q1_target = target_critic_1(x', a')           # 目标 Critic 1
  Q2_target = target_critic_2(x', a')           # 目标 Critic 2
  Q_target = r + γ · (1-done) · min(Q1_target, Q2_target)  # TD target（取最小值）

  # ── 更新在线 Critic（计算梯度，反向传播）──
  Q1_current = critic_1(x, a)                   # 当前 Critic 1 的预测
  Q2_current = critic_2(x, a)                   # 当前 Critic 2 的预测
  L_Q = MSE(Q1_current, Q_target) + MSE(Q2_current, Q_target)  # 两个 Critic 的损失之和
  ψ ← ψ - α · ∇L_Q                              # 梯度下降更新
```

##### Step 3：更新 Actor

Actor 的损失函数：

```
L_π(θ) = E[-Q_ψ(x, a₁:C) + β · ‖a₁:C - \tilde{a}₁:C‖²]
```

两种力的博弈：

| 项 | 含义 | 效果 |
|---|---|---|
| **-Q(s, a)** | Critic 认为的动作价值，加负号 → 最大化 Q | 推动 Actor 选择"Critic 认为好"的动作 |
| **β · ‖a - \tilde{a}‖²** | 与 VLA 参考动作的 MSE，即 BC 正则化 | 拉住 Actor，不让它偏离 VLA 太远 |

**为什么需要 BC 正则化？**

```
Q 面不是完美准确的 ──── 特别是在 out-of-distribution 区域
                        │
  Actor 如果只追求 Q 最大化 ──→ 会找到 Q 面的"假高峰"（adversarial exploitation）
                        │
  加上 BC 正则 ──→ 把 Actor 约束在 VLA 附近 ← VLA 见过的分布内
                        │
                        └→ 让 Q 的估计可靠，不会问"从没见过的动作好不好"
```

**β 的调节作用**：

β 控制 Actor 偏离 VLA 的**自由度**：

| β | 行为 | 场景 |
|---|---|---|
| **β=0** | Actor 完全自由，只追求 Q 最大化 | 容易发散，Q 估计爆炸 |
| **β→∞** | Actor 完全复制 VLA | 等于没做 RL |
| **β=1~10** | 在 VLA 附近做小修正 | **RLT 的默认设置**——只在参考动作附近微调 |

**数据流图**：

```
# Actor 更新（每隔 d 步才执行一次，d 通常是 2-3）
# 这是 TD3 的 "Delayed Policy Update" 技巧——让 Critic 先学稳，Actor 再更新

a_current = actor(x, ref_actions)              # Actor 输出当前修正后的动作
Q = min(critic_1(x, a_current), critic_2(x, a_current))  # 双 Critic 取最小值
bc_loss = MSE(a_current, ref_actions[:, :C*32])  # 只约束前 C 步

L_π = -Q.mean() + β * bc_loss                 # 总损失
θ ← θ - α · ∇L_π                              # 梯度下降更新 Actor
```

##### 完整内循环伪代码

```python
# 超参数
BATCH_SIZE = 256        # 每批采样数量
G = 20                  # 每收集一次数据，内循环跑 20 次更新
POLICY_DELAY = 2        # 每 2 次 Critic 更新才更新 1 次 Actor
POLYAK_TAU = 0.005      # 目标网络软更新系数
GAMMA = 0.99            # 折扣因子
BETA = 2.0              # BC 正则化系数
SMOOTH_NOISE = 0.2      # 目标平滑噪声标准差
NOISE_CLIP = 0.5        # 噪声裁剪范围

# 内循环
for _ in range(G):
    # Step 1: 从 Replay Buffer 采样
    batch = replay_buffer.sample(BATCH_SIZE)
    # batch.state     → [256, 2080]
    # batch.action    → [256, 320]
    # batch.reward    → [256, 1]
    # batch.next_state → [256, 2080]
    # batch.done      → [256, 1]

    # Step 2: 更新 Critic
    with torch.no_grad():
        next_action = target_actor(batch.next_state)  # [256, 320]
        noise = torch.randn_like(next_action) * SMOOTH_NOISE
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (next_action + noise).clamp(-1, 1)
        # 也可以 clamp 到动作空间的实际范围
        q1_target = target_critic_1(batch.next_state, next_action)  # [256, 1]
        q2_target = target_critic_2(batch.next_state, next_action)  # [256, 1]
        q_target = batch.reward + GAMMA * (1 - batch.done) * torch.min(q1_target, q2_target)

    q1 = critic_1(batch.state, batch.action)  # [256, 1]
    q2 = critic_2(batch.state, batch.action)  # [256, 1]
    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Step 3: 每 POLICY_DELAY 步更新一次 Actor
    if step_count % POLICY_DELAY == 0:
        current_action = actor(batch.state)  # [256, 320]
        q = torch.min(critic_1(batch.state, current_action),
                      critic_2(batch.state, current_action))
        bc_loss = F.mse_loss(current_action, batch.ref_actions[:, :C*ACTION_DIM])
        actor_loss = -q.mean() + BETA * bc_loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(target_critic_1.parameters(), critic_1.parameters()):
            target_param.data.copy_(POLYAK_TAU * param + (1 - POLYAK_TAU) * target_param)
        # 相同方式更新 target_critic_2 和 target_actor
```

| 方面 | RECAP（pi0.6 论文方法） | RLT（RL Token） |
|---|---|---|---|---|
| **方法** | 优势条件化 (Advantage Conditioning) | 在线 Actor-Critic (TD3 + BC) |
| **VLA 参数** | 重新训练整个 VLA | **冻结** VLA，只训练小模块 |
| **数据需求** | 大量离线数据 + 自动收集数据 | 少量 demo + 15min-2h 在线数据 |
| **额外模型** | 670M Value Function | 2-3 层 MLP Actor-Critic |
| **适合场景** | 从头训练通用策略 | 在已有 VLA 上快速微调 |
| **精度** | 通用任务提升 | 高精度任务（拧螺丝、插线） |

---

## 7. 参考资料

- [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://browse-export.arxiv.org/abs/2604.23073) — RLT 原论文
- [openpi-RLT (GitHub)](https://github.com/Yyshadow/openpi-RLT) — 基于 openpi 的开源复现
- [π*₀.₆: a VLA That Learns From Experience](https://arxiv.org/html/2511.14759v2) — pi0.6 原论文
- [Physical Intelligence 官网](https://www.pi.website/)
