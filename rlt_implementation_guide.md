# RLT (RL Token) 实现指南 — 基于 openpi 代码库

> 基于 Physical Intelligence 论文 *RL Token: Bootstrapping Online RL with Vision-Language-Action Models* (2026) 和 *π*₀.₆: a VLA That Learns From Experience* (2025)

---

## RLT 核心思想（一句话）

> **冻住 50 亿参数的 VLA 大模型，外挂一个 2 层 MLP 的 Actor-Critic 做在线 RL 微调**

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
| `rlt_online_rl/src/rlt_online_rl/actor.py` | **Actor 网络**：2-3 层 MLP，输入 (rl_token + state + ref_actions)，输出修正动作 |
| `rlt_online_rl/src/rlt_online_rl/critic.py` | **Critic 网络**：2-3 层 MLP，输入 (rl_token + state + action_chunk)，输出 Q 值（TD3 双 Critic） |
| `rlt_online_rl/src/rlt_online_rl/learner.py` | **TD3 + BC 训练循环**：从 Replay Buffer 采样 → 更新 Critic → 更新 Actor |
| `rlt_online_rl/src/rlt_online_rl/replay.py` | **Replay Buffer**：存 (state, action, reward, next_state) |
| `rlt_online_rl/src/rlt_online_rl/rollout.py` | **Rollout 运行时**：机器人交互、奖励计算、数据收集 |

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

### 3.2 PI0.5 + RLT 推理流程（冻结 VLA + Actor 修正）

```
  Observation 输入 (与 PI0.5 相同)
  images [b, 224, 224, 3] × 3, text tokens [b, ≤200]
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │               PI0.5 VLA (全部冻结)                    │
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

---

## 4. 简明架构对比图 — PI0.5 vs PI0.6 RLT 新增模块

新手直接看这张图就够了。左侧是基础 PI0.5，右侧是 RLT 新增的部分（标 `+`）。

### 4.1 宏观对比

```
PI0.5（原始 VLA）                        PI0.6 RLT（VLA + 在线 RL）
┌─────────────────────┐                  ┌─────────────────────────────────┐
│                     │                  │                                 │
│  Observation        │                  │  Observation                    │
│  (images + text)    │                  │  (images + text)                │
│         │           │                  │         │                       │
│         ▼           │                  │         ▼                       │
│  ┌─────────────┐    │                  │  ┌─────────────┐               │
│  │  PI0.5 VLA  │    │                  │  │  PI0.5 VLA  │ （冻结 ✅）    │
│  │  (5B params)│    │                  │  │  (5B params)│               │
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
| **PI0.5 VLA** | 基础视觉-语言-动作大模型, 输出参考动作 | ✅ 冻结 |
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
1. 加载预训练 VLA（pi0.5），冻结所有参数
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

---

## 6. 与 pi0.6 (RECAP) 的区别

| 方面 | pi0.6 (RECAP) | RLT (RL Token) |
|---|---|---|
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
