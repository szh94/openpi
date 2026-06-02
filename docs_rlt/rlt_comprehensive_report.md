# RLT (RL Token) 完整实现报告

> 基于 Physical Intelligence 论文 *RL Token: Bootstrapping Online RL with Vision-Language-Action Models* (2026)
>
> 本文档是对 `docs_rlt/` 下所有文档的合并与整理，涵盖：代码结构分析、实现指南、代码修订记录、常见问题解答。

---

## 目录

1. [RLT 核心思想](#1-rlt-核心思想)
2. [项目整体架构](#2-项目整体架构)
3. [PI0 vs PI0.5 关键区别](#3-pi0-vs-pi05-关键区别)
4. [RLT 原理详解](#4-rlt-原理详解)
   - 4.1 [RLT = VLA + 微调修正](#41-rlt--vla--微调修正)
   - 4.2 [RL Token：压缩场景理解](#42-rl-token压缩场景理解)
   - 4.3 [Actor：在参考动作附近做微调](#43-actor在参考动作附近做微调)
   - 4.4 [为什么 C=10 而不是 H=50？](#44-为什么-c10-而不是-h50)
   - 4.5 [什么是 BC 正则化？](#45-什么是-bc-正则化)
   - 4.6 [为什么 Encoder 后还需要 Decoder？](#46-为什么-encoder-后还需要-decoder)
5. [VLA 推理流程详解](#5-vla-推理流程详解)
6. [RLT 推理流程详解](#6-rlt-推理流程详解)
7. [Shape 变化对照表](#7-shape-变化对照表)
8. [文件清单](#8-文件清单)
   - 8.1 [新增文件](#81-新增文件)
   - 8.2 [修改的现有文件](#82-修改的现有文件)
9. [训练流程（两阶段）](#9-训练流程两阶段)
   - 9.1 [Stage 1：RL Token 训练](#91-stage-1rl-token-训练)
   - 9.2 [Stage 2：在线 RL 训练](#92-stage-2在线-rl-训练)
   - 9.3 [内循环详解](#93-内循环详解)
10. [AC 网络训练深入分析](#10-ac-网络训练深入分析)
    - 10.1 [Critic 训练的底层逻辑](#101-critic-训练的底层逻辑)
    - 10.2 [Actor 训练的底层逻辑](#102-actor-训练的底层逻辑)
    - 10.3 [梯度流详解](#103-梯度流详解)
    - 10.4 [训练稳定性与调试](#104-训练稳定性与调试)
11. [超参数速查](#11-超参数速查)
12. [奖励函数设计](#12-奖励函数设计)
13. [人工干预需求分析](#13-人工干预需求分析)
14. [执行时序分析](#14-执行时序分析)
15. [框架依赖检查](#15-框架依赖检查)
16. [验证命令](#16-验证命令)
17. [Ubuntu / Conda 迁移指南](#17-ubuntu--conda-迁移指南)
18. [参考资料](#18-参考资料)

---

## 1. RLT 核心思想

> **冻结约 48.6 亿参数的 VLA 大模型（pi0.6），外挂一个 2-3 层 MLP 的 Actor-Critic（AC 网络）做在线 RL 微调。**

核心创新是 **RL Token**：一个 Encoder-Decoder Transformer，把 VLA 最后一层输出的 N×2048 维 token 序列，压缩成一个 2048 维的"状态向量"，喂给轻量 Actor-Critic（AC 网络）。

| 核心原则 | 说明 |
|---------|------|
| **VLA 冻结** | 基座 VLA 模型参数在 RLT 全过程中不变 |
| **小模块外挂** | 所有新增模块（Encoder + Actor + Critic）参数量不到 VLA 的 1% |
| **两阶段训练** | Stage 1 离线训练 Encoder，Stage 2 在线训练 AC 网络 |
| **BC 正则化** | 约束 Actor 输出不偏离 VLA 参考动作太远 |

> **模型版本说明**：RLT 原论文使用 **pi0.6**（Gemma 3 4B 主干 + 860M action expert）作为基座 VLA。我们的实现基于 openpi 代码库的 **pi0.5**（Gemma 2B + 300M action expert），因为 pi0.6 未开源。RLT 方法与具体模型版本无关，两种架构均可适用。

---

## 2. 项目整体架构

本项目包含 **两条主要 pipeline**：

### Pipeline 1: PI0 / PI0.5 — 标准模仿学习（Behavior Cloning）

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

### Pipeline 2: RLT — 两阶段在线强化学习微调

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

## 3. PI0 vs PI0.5 关键区别

（在 `pi0.py` 中通过 `self.pi05` 标志控制）

| 特性 | PI0 | PI0.5 |
|------|-----|-------|
| **State 输入方式** | 连续值 concat 到 suffix emb | `discrete_state_input`：作为离散语言 token 输入 prefix |
| **时间步注入** | concat action 和 time emb 后过 MLP | adaRMSNorm（`adarms_cond`） |
| **max_token_len** | 48 | 200（state 被 tokenize 了） |
| **归一化方式** | 标准 z-score | 分位数归一化（quantile norm） |

---

## 4. RLT 原理详解

### 4.1 RLT = VLA + 微调修正

**RLT 的原理就是在 VLA 预测的动作局部做微小调整。**

#### 三种策略对比

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **纯 VLA** | 从演示数据中模仿行为 | 大局观正确 | 无法通过试错自我改进 |
| **纯 RL** | 从零开始探索 | 可以优化到极致 | 5B 参数 + 高维动作空间 → 探索空间无穷大 |
| **RLT** | VLA 画了个圈，Actor 在圈里微调 | 既保持 VLA 的大局观，又能在线优化局部细节 | — |

#### 物理类比：投篮

| 角色 | 类比 | 说明 |
|------|------|------|
| **VLA** | 教练示范标准动作 | 提供了正确的发力方式、出手角度 |
| **RL Token** | 球员对篮筐位置的感觉 | 对当前场景的理解（距离、角度） |
| **Actor** | 微调出手角度 | 在教练示范的基础上，根据实际感觉调整 |
| **BC 正则化** | "别改太多，先按教练的来" | 防止大改导致动作变形 |
| **Q 值** | "这球进了" | 实际结果的反馈 |

### 4.2 RL Token：压缩场景理解

VLA 的 N×2048 维 token 序列包含了对场景的完整理解，但 Actor（2 层 MLP）吃不下几千个 token。RLTEncoder 用 Transformer Encoder 压缩成单个向量：

```python
# 在 rl_token.py 的 RLTEncoder 中
rl_token = encoder(all_embeddings)  # [B, N, 2048] → [B, 2048]
```

这个 RL Token 是 VLA 对当前场景理解的 **"精华摘要"**。

### 4.3 Actor：在参考动作附近做微调

```python
# Actor 的输入是两个东西的拼接
x = jnp.concatenate([state, ref_actions], axis=-1)
#   ↑                       ↑
#   RL Token + proprio       VLA 的参考动作 (1600 维 = 50×32)
#   [2080]                   — 作为"锚点"

# 输出：修正后的动作
mean = self.mean_out(x)  # [B, 320] = C=10 × 32 dim
```

关键在损失函数：

```python
loss = q_loss + beta * bc_loss
#      ↑           ↑
#  让动作"更好"   让动作"不离谱"
```

- **`q_loss = -Q.mean()`**：最大化 Critic 评估的动作价值，推动 Actor 选择更好的动作
- **`bc_loss = MSE(mean, ref_actions[:C])`**：MSE 惩罚 Actor 输出偏离 VLA 参考动作的程度
- **`beta`**：控制两者的平衡

### 4.4 为什么 C=10 而不是 H=50？

```python
action_chunk: int = 10    # Actor 只预测前 10 步
action_horizon: int = 50  # VLA 预测 50 步
```

VLA 预测完整 50 步的"宏观轨迹"，Actor 只截取前 10 步做修正——因为 10 步之内环境不会大变，VLA 的参考已经够准。执行 10 步后重新观测，再次 VLA + Actor 修正。这比直接修正 50 步更合理：**远期的动作不需要微调，因为未来会有新的观测进来重新规划**。

### 4.5 什么是 BC 正则化？

BC 正则化中的 **BC = Behavioral Cloning（行为克隆）**。

在 RLT 的 Actor 损失函数中：

```
L_actor = -Q(s, a) + β · ‖a - a_ref‖²
          ↑               ↑
         "变更好"        "别学歪"
          (RL 目标)       (BC 正则化)
```

#### 为什么需要它？

Critic 是用 Actor 收集的数据训练的。如果 Actor 输出了 VLA 从未见过的奇怪动作，Critic 没有数据知道这个动作好不好，就会给出不可靠的高 Q 值（adversarial exploitation）。BC 正则化就是用一根"绳子"把 Actor 拴在 VLA 附近——VLA 输出的动作是在训练数据分布内的，Critic 对这个区域的评估是可靠的。

#### Beta 系数的作用

| Beta | 行为 | 效果 |
|------|------|------|
| β=0 | Actor 完全自由 | Q 估计爆炸，容易发散 |
| β=1~5 | 在 VLA 附近"微调" | RLT 的理想工作区间 |
| β→∞ | Actor 完全复制 VLA | 等于没做 RL |

**默认值 β=2.0**：在参考动作附近微调，安全有效。

### 4.6 为什么 Encoder 后还需要 Decoder？

Decoder 只用于 **Stage 1（离线预训练）**，在线 RL 阶段完全不用。它的作用是提供监督信号，形成一个自编码器结构：

```
VLA token embeddings [B, N, 2048]
        │
        ▼
  ┌─────────────┐
  │  Encoder    │  ← 压缩
  └──────┬──────┘
         ▼
  RL Token [B, 2048]   ← 瓶颈（信息被压缩了）
         │
         ▼
  ┌─────────────┐
  │  Decoder    │  ← 重建
  └──────┬──────┘
         ▼
  重建的 embeddings [B, N, 2048]
        │
  对比 MSE(重建, 原始) ← 监督信号
```

**没有 Decoder 的后果**：Encoder 训练缺少监督信号，容易模式坍塌——不管输入是什么都输出相同的 RL Token，或只编码了部分信息。

Decoder 就像脚手架——建完房子就拆掉了。Stage 1 结束 → Decoder 丢弃 → Encoder 冻结 → Stage 2 只用 Encoder 提取 RL Token。

---

## 5. VLA 推理流程详解

### PI0.5 现有推理流程（sample_actions）

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

---

## 6. RLT 推理流程详解

### PI0.6 + RLT 推理流程（冻结 VLA + Actor 修正）

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
  ref_actions                      ┌─────────────────────────┐
  [b, 50, 32]                      │ prefix_out [b,968,2048] │
                                   │ suffix_out  [b,50,1024] │
                                   │ 注: 两 expert 宽度不同   │
                                   │ 需投影层统一为 2048 维   │
                                   └──────────┬──────────────┘
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
                          └───────────────┬───────────────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │               在线 RL (仅训练这部分)                    │
              │                                                       │
              │  ┌─────────────────────────────────────────────┐      │
              │  │ Actor π_θ (2-3 层 MLP)                      │      │
              │  │ 输入: state [2080] + ref_actions [1600]     │      │
              │  │ 输出: 修正后动作 chunk [320] = 10×32        │      │
              │  │ 损失: -Q + β·‖a - ref‖²                    │      │
              │  └──────────────────┬──────────────────────────┘      │
              │                     │                                 │
              │  ┌──────────────────┴──────────────────────────┐      │
              │  │ Critic Q_ψ (2-3 层 MLP, ×2 TD3)            │      │
              │  │ 输入: state [2080] + action_chunk [320]     │      │
              │  │ 输出: Q 值 [1]                              │      │
              │  │ 损失: TD error (标准 TD3)                   │      │
              │  └─────────────────────────────────────────────┘      │
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

### 完整推理数据流（RLTPolicy）

```
观测 → VLA（冻结）
  │
  ├─→ ref_actions [50×32]          "拿杯子 -> 向左移动 -> 放到目标位置"
  │      └──→ Actor 截取前 10 步并微调
  │             修正前：[0.82, 0.15, -0.33, ...]  ← VLA 的原始动作
  │             修正后：[0.79, 0.18, -0.30, ...]  ← 只差了几个百分点
  │
  └─→ token embeddings [N×2048]    "场景理解"
       └──→ Encoder → RL Token     "压缩摘要"
              └──→ 拼接 proprio     "加上本体感受"
                     └──→ Actor 输入  "知道场景+知道VLA的建议"
```

---

## 7. Shape 变化对照表

| 阶段 | 变量 | Shape | 说明 |
|------|------|-------|------|
| **输入** | images | `[b, 224, 224, 3]` × 3 | 3 个摄像头，归一化到 [-1, 1] |
| | tokenized_prompt | `[b, ≤200]` | 文本指令 token IDs |
| | state (proprio) | `[b, 32]` | 机器人关节角度 (PI0.5 离散化后进 prefix) |
| **embed_prefix** | image_tokens | `[b, 256, 2048]` × 3 = `[b, 768, 2048]` | SigLIP 输出，224/14=16, 16²=256 |
| | text_tokens | `[b, L_text, 2048]` | Gemma embedder 输出 |
| | → prefix_tokens | `[b, ~968, 2048]` | 768 + L_text |
| **embed_suffix** | noisy_actions | `[b, 50, 32]` | 含噪动作 (推理开始时是纯噪声) |
| | action_embeds | `[b, 50, 1024]` | action_in_proj Linear(32→1024) |
| | time_emb | `[b, 1024]` | posemb_sincos + time_mlp |
| | → suffix_tokens | `[b, 50, 1024]` | 动作 tokens，因果注意力 |
| **Transformer** | prefix_out | `[b, ~968, 2048]` | Expert 0 (PaliGemma 2B) 输出 |
| | suffix_out | `[b, 50, 1024]` | Expert 1 (Action Expert 300M) 输出 |
| **Flow Matching** | v_t | `[b, 50, 32]` | action_out_proj Linear(1024→32) |
| | x₀ | `[b, 50, 32]` | 去噪 10 步后得到的干净动作 |
| **RLT 新增** | | | |
| | token_embeddings | `[b, ~1018, 2048/1024]` | 两 expert 宽度不同，需统一处理 |
| | RL Token z_rl | `[b, 2048]` | Encoder 压缩后的状态表示 |
| | 状态 x | `[b, 2080]` | z_rl(2048) + proprio(32) |
| | ref_actions | `[b, 1600]` | VLA 输出 50×32, flatten 后喂 Actor |
| | Actor 输出 a₁:C | `[b, 320]` | C=10 步，每步 32 维，flatten |
| | Q 值 | `[b, 1]` | Critic 输出标量 |

> **关于 action_dim=32**：这是 VLA 预训练时对所有机器人动作空间取"最大公约数"的固定维度。RLT 原论文使用双臂 6-DOF 机器人（6+1+6+1=14 维），实际动作为 14 维。openpi 代码中 action_dim=32 是模型内部维度，低维机器人的多余维度会被填充为 0。

---

## 8. 文件清单

### 8.1 新增文件

#### 模型层

| 文件 | 说明 |
|------|------|
| `src/openpi/models/rl_token.py` | **RL Token Encoder-Decoder**：4 层 Transformer 编码器压缩 VLA token embeddings [B, N, 2048] → [B, 2048]；解码器用交叉注意力重建原始序列 |

- **`RLTEncoder`**：4 层 pre-norm Transformer（width=2048, 8 heads, mlp_ratio=4.0），使用可学习的特殊 `[RL]` token 附加到输入序列末尾，取该位置输出作为 RL Token
- **`RLTDecoder`**：cross-attention decoder，仅 Stage 1 训练使用
- **`RLTEncoderConfig` / `RLTDecoderConfig`**：配置数据类

#### 策略层

| 文件 | 说明 |
|------|------|
| `src/openpi/policies/rlt_policy.py` | **RLT 策略包装器**：VLA → token embeddings → RLTEncoder → RL Token → concat proprio → Actor → 修正动作 |

- `RLTPolicy` 包装冻结 VLA + 冻结 RLTEncoder + 在线 Actor
- `infer()` 实现完整推理流水线
- 无 Actor 时回退到 VLA 参考动作

#### 在线 RL 训练模块（`rlt_online_rl/` 独立目录）

| 文件 | 说明 | 网络结构 |
|------|------|----------|
| `rlt_online_rl/actor.py` | **Actor 网络** | 2 层 MLP (512 hidden)，输入 [3680] → 输出 mean[320] + log_std[320] |
| `rlt_online_rl/critic.py` | **Critic 网络** | 2 层 MLP (256 hidden)，双拷贝 TD3，输入 [2400] → Q 值 [1] |
| `rlt_online_rl/learner.py` | **TD3 + BC 训练器** | 采样 → Critic 更新 → 延迟 Actor 更新 → Polyak 软更新 |
| `rlt_online_rl/replay.py` | **经验回放缓冲** | 循环数组，容量 1M，stride=2 存储 |
| `rlt_online_rl/rollout.py` | **Rollout 运行时** | 机器人交互循环，奖励收集，transition 存储 |

**Actor 损失**：
```
L_π(θ) = E[-Q_ψ(x, a₁:C) + β·‖a₁:C - \tilde{a}₁:C‖²]
```

**Critic 损失**（标准 TD3）：
```
L_Q(ψ) = E[(ˆQ - Q_ψ(x, a₁:C))²]
ˆQ = r + γ · (1-done) · min(Q_ψ₁'(x', a'), Q_ψ₂'(x', a'))
```

#### 训练入口

| 文件 | 说明 |
|------|------|
| `scripts/train_rlt_token.py` | **Stage 1**：在 demo 数据上训练 RL Token Encoder-Decoder（冻住 VLA） |
| `scripts/train_rlt_online.py` | **Stage 2**：启动在线 RL（异步 rollout + learner） |

#### 部署

| 文件 | 说明 |
|------|------|
| `scripts/serve_rlt_policy.py` | RLT 策略服务（冻结 VLA + Encoder + Actor） |
| `examples/test_rlt_inference.py` | RLT 推理流程测试脚本（虚拟数据验证 shape 正确性） |

### 8.2 修改的现有文件

| 文件 | 修改内容 |
|------|----------|
| `src/openpi/models/model.py` | 新增 `ModelType.RLT = "rlt"`，新增 `RLTBaseModelConfig` 数据类（rl_token_width, rl_encoder_depth, rl_encoder_num_heads, action_chunk, proprio_dim） |
| `src/openpi/models/pi0.py` | 新增 `get_token_embeddings()` 方法提取 VLA 中间 token embeddings 和参考动作；新增 `_suffix_emb_proj` 投影层统一 suffix_out（1024→2048） |
| `src/openpi/training/config.py` | 新增 `rlt_debug` 训练配置；导入 rl_token 模块 |

#### `pi0.py` 最关键的修改：`get_token_embeddings()`

```python
class Pi0:
    def get_token_embeddings(self, observation) -> tuple[Array, Array]:
        """返回 VLA 最后一层的 token embeddings + 参考动作"""
        # embed_prefix 产生图片+文本 tokens
        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        # embed_suffix 产生 action tokens（用随机噪声占位）
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

## 9. 训练流程（两阶段）

### 9.1 Stage 1：RL Token 训练（Offline）

```
Demo Data → VLA (frozen) → Token Embeddings [B, N, 2048]
                                    ↓
                            RLTEncoder → RL Token [B, 2048]
                                    ↓
                            RLTDecoder → Reconstructed [B, N, 2048]
                                    ↓
                            MSE Loss(reconstructed, original)
```

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
4. 冻结 ϕ (Encoder)
5. 丢弃 Decoder
```

#### Stage 1 深度解析：RL Token 的信息代表性

##### 核心问题：为什么单 token 压缩不丢失关键信息

将 N × 2048 维的 token 序列压缩成单个 2048 维的向量，表面上压缩比约 500:1。保证信息不丢失依赖以下机制。

##### 一、信息瓶颈理论视角

Stage 1 的训练目标等价于最大化互信息下界：

```
L_recon(ϕ) = E[‖decoder(encoder(embeddings)) - embeddings‖²]

→ 最小化 MSE = 最大化 I(embeddings; RL_token) 的下界
```

当重建误差趋近于零时，RL Token 包含了足以完整重建原始 embeddings 的全部信息。关键洞察：压缩维度为 2048 不是因为信息量只需要 2048 维，而是因为原始 VLA 的隐藏维度就是 2048。**信息损失来自"从 N 个位置到 1 个位置的聚合"，而非来自"维度降低"**。

##### 二、架构保证：全注意力 + 可学习 [RL] Token

| 方法 | 信息聚合方式 | 问题 |
|------|-------------|------|
| **Mean Pooling** | 所有位置等权平均 | 破坏空间结构，假设所有 token 同等重要 |
| **Max Pooling** | 每维取最大值 | 只保留最显著特征 |
| **RL Token (Ours)** | 全注意力自适应加权 | **模型学习哪些 token 重要** |

RLTEncoder 使用全注意力 mask，[RL] token 可以通过 4 层 Transformer 的注意力头，学习到对每个输入 token 的差异化权重。可学习的位置编码让 [RL] token 能区分视觉 token、语言 token 和动作 token。

##### 三、Decoder 的信息压力测试

Decoder 必须从单个 2048 维向量重建 N × 2048 个值。这个 **N × 2048 ≈ 2M 个标量值的 MSE** 是信息保留的直接监督信号：

- 如果 MSE 降到很低 → RL Token 确实包含足够信息
- 如果 MSE 降不下去 → 2048 维是瓶颈，需增大容量

RLT 论文中 MSE 收敛到 ~0.01，说明 2048 维对于 VLA token embeddings 的"有效信息"是足够的。

**注意**：VLA embeddings 并非独立同分布——相邻视觉 patch 高度相关，语言 token 存在句法依赖，动作 token 时序平滑。Encoder 的注意力机制可以利用这些相关性实现高效压缩。

##### 四、与其他压缩方案的对比

| 方案 | 表达能力 | 信息保留 | 选型原因 |
|------|---------|---------|---------|
| Mean Pooling | 低 | 低 | — |
| Transformer Encoder + [RL] | **高** | **高** | ✅ 非线性交互 + 位置感知 |
| 所有 token 不压缩 | 无压缩 | 完全保留 | Actor MLP 无法处理 4M 维输入 |

##### 五、信息保留的经验验证

```python
# 重建质量检查
reconstructed = decoder(rl_token)
recon_error = jnp.mean(jnp.square(reconstructed - embeddings))
# 期望: < 0.05
```

从理论上，如果 MSE < ε，则对任意 Lipschitz 连续的 Actor 函数 f：`|f(decoder(encoder(embeddings))) - f(embeddings)| < L·ε`。即 **RL Token 是原始 embeddings 的 ε-近似充分统计量**。

### 9.2 Stage 2：在线 RL 训练

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

```
循环 t = 0, C, 2C, ...:
  1. VLA 生成参考动作块 \tilde{a} ~ π_vla(s_t)
  2. Encoder 生成 RL Token: z_rl = encoder(z₁:M)
  3. 构成状态: x_t = (z_rl, s_t^p)
  4. Actor 选择动作: a ~ π_θ(·|x_t, \tilde{a})
  5. 执行 a₁:C，收集奖励 r_t，下一状态
  6. 存入 Replay Buffer

内循环（每收集一次数据，跑 G=20 次更新）:
  1. 从 Buffer 采样 batch
  2. TD3 更新 Critic: min L_Q(ψ)
  3. 延迟更新 Actor: min -Q + β·BC_reg
  4. Polyak 软更新目标网络
```

### 9.3 内循环详解

#### 为什么内循环要跑 G 次更新？

这是 **off-policy** 算法的核心优势：数据收集和学习频率解耦。

```
┌─ Rollout 进程（慢）──────────────────┐
│ 机器人每走 10 步 → 存 1 条 transition │  ← 受物理世界速度限制
└──────────────────────────────────────┘
                    ↓
┌─ Learner 进程（快）───────────────────┐
│ 每收集到 1 条 transition → 复用 G 次   │  ← G 通常 ≈ 10-50
│ 从 buffer 反复采样旧数据学习           │     一条数据用多次
└──────────────────────────────────────┘
```

物理世界 1 秒只能走 ~10 步，但 GPU 1 秒能学 ~100 次。不让 Learner 闲着，提高样本效率。

#### 完整内循环伪代码

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
    # Step 1: 采样
    batch = replay_buffer.sample(BATCH_SIZE)
    # batch.state      → [256, 2080]
    # batch.action     → [256, 320]
    # batch.reward     → [256, 1]
    # batch.next_state → [256, 2080]
    # batch.done       → [256, 1]

    # Step 2: 更新 Critic（TD3）
    with torch.no_grad():
        next_action = target_actor(batch.next_state)
        noise = torch.randn_like(next_action) * SMOOTH_NOISE
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (next_action + noise).clamp(-1, 1)
        q1_target = target_critic_1(batch.next_state, next_action)
        q2_target = target_critic_2(batch.next_state, next_action)
        q_target = batch.reward + GAMMA * (1 - batch.done) * torch.min(q1_target, q2_target)

    q1 = critic_1(batch.state, batch.action)
    q2 = critic_2(batch.state, batch.action)
    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Step 3: 每 POLICY_DELAY 步更新一次 Actor
    if step_count % POLICY_DELAY == 0:
        current_action = actor(batch.state)
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

#### 关键训练技巧

| 技巧 | 说明 |
|------|------|
| **Action Chunk C=10** | VLA 预测 H=50 步，Actor 决策 C=10 步，horizon 压缩 5 倍 |
| **参考动作 Dropout** | 50% 概率将 ref_actions 置零，防止 Actor 偷懒复制 VLA |
| **BC 正则化 β** | 控制 Actor 偏离 VLA 的程度 |
| **Stride=2** | 重叠存储 transition，样本效率提高 5 倍 |
| **异步架构** | Rollout 进程 + Learner 进程独立运行 |
| **Twin Critics (TD3)** | 双 Critic 取最小值，防止 Q 值高估 |
| **Target Networks** | Polyak 软同步 (τ=0.005)，防止训练振荡 |
| **Target Policy Smoothing** | 计算目标 Q 时对下一动作加噪声，防止过拟合 |

---

## 10. AC 网络训练深入分析

### 10.1 训练全貌：两个网络如何协同

```
环境 ──→ (state, ref_actions) ──→ Actor ──→ action ──→ 环境（执行）
                                      │                      │
                                      │                      ▼
                                      │                reward + next_state
                                      ▼                      │
                                   Critic ←──────────────────┘
                                      │
                                      ▼
                               Q(s, a) — 指导 Actor 更新
```

Actor 和 Critic 的关系是 **"学生与老师"的动态博弈**：

| 角色 | 类比 | 目标 | 训练信号 |
|------|------|------|---------|
| **Actor** | 学生 | 输出"好"的动作 | 从 Critic 的 Q 值学习 |
| **Critic** | 老师 | 准确评估动作的好坏 | 从环境奖励学习 |

**关键矛盾**：老师（Critic）一开始也是"新手"，给出的评分不一定准确。学生（Actor）如果太听老师的话，可能学到错误的东西。这就是为什么需要 **TD3 的延迟更新机制**——让老师先多学几轮，学得更准了再指导学生学习。

### 10.2 Critic 训练的底层逻辑

#### Critic 到底在学什么？

Critic 是一个 **Q 函数近似器**，目标是学会回答：

> "在状态 s 下做动作 a，未来能获得多少累积奖励？"

即：`Q(s, a) = E[r₀ + γ·r₁ + γ²·r₂ + ... | s₀=s, a₀=a]`

Critic 用 **TD learning（时序差分学习）** 来训练：

```
标签 ˆQ = r + γ · Q_target(s', a')    ← 用"当前估计"构造"标签"
预测 Q  = Q_current(s, a)              ← 网络当前输出
损失   = MSE(Q - ˆQ)                   ← 让预测逼近标签
```

这本质上是 **bootstrap（自助法）**：用自己（或自己的旧版本）的估计来构造训练标签。

#### TD3 三项关键技术的直观理解

| 技术 | 类比 | 解决什么问题 |
|------|------|-------------|
| **Twin Critics（双 Critic）** | 请两个独立的老师评分，取较低的分数 | 单个老师可能"过度乐观"，给烂动作打高分 → Actor 被误导 |
| **Target Networks（目标网络）** | 老师不用今天的标准打分，用上周的标准 | 防止"今天标准今天用"导致的正反馈振荡 |
| **Target Policy Smoothing（目标平滑）** | 评分时稍微模糊化输入的动作 | 防止老师在个别动作上"钻牛角尖"（过拟合） |

这些技巧的共同目的：**防止 Q 值被高估（overestimation bias）**。

### 10.3 Actor 训练的底层逻辑

#### Actor 的"偷师"过程

Actor **不直接从环境 reward 学习**，而是**从 Critic 的 Q 值学习**：

```
Actor 更新方向：让 Q(s, Actor(s)) 变大
                ↑
         Critic 告诉 Actor 哪个方向是"更优"
```

梯度链：

```
state [256, 2080], ref_actions [256, 1600]
                │           │
                └─────┬─────┘
                      ▼
                Actor MLP (2-3 层)
                      │
                      ▼
               action [256, 320] ←── 这是可微的！
                      │
                      ▼
           ┌──────────┴──────────┐
           │   Critic (冻结!)    │ ←── 这里不更新 Critic
           │   只做前向传播       │
           └──────────┬──────────┘
                      │
                      ▼
                  Q 值 [256, 1]
                      │
                 L = -Q.mean()    ← 梯度反向通过 Critic 传到 Actor
                      │
                      ▼
            Actor 参数更新 θ ← θ - α·∇L_π
```

**关键洞察**：Critic 在这里充当了一个"可微分的目标函数"。Actor 通过 Critic 的梯度信号，知道"往哪个方向调整动作能让 Q 值变大"。

#### BC 正则化的梯度效应

```
总梯度 = ∇(-Q) + β · ∇(MSE)

∇(-Q): 指向"Q 值增长最快的方向"（探索性）
β·∇(MSE): 指向"拉回 VLA 参考动作的方向"（保守性）
       ↓
合力方向: 在 VLA 附近寻找 Q 值更高的动作
```

这就是 **"安全探索"** 的数学本质：Actor 被允许偏离 VLA，但不能太远。

#### 延迟更新（Delayed Policy Update）

TD3 中 Actor 的更新频率比 Critic 低（每 2-3 次 Critic 更新才更新一次 Actor）：

```
时间步  1  2  3  4  5  6  7  8  9  10 ...
Critic  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓
Actor        ✓        ✓        ✓
```

**原理**：Critic 必须先学得相对准确，才能给 Actor 提供有用的梯度。

#### 参考动作 Dropout 的作用

训练时以 50% 概率把 ref_actions 置零：

```
情况 1 (50%): ref_actions = 0          → Actor 完全靠自己探索
情况 2 (50%): ref_actions = VLA 输出    → Actor 有参考
```

**目的**：防止 Actor 过度依赖参考动作，强迫它在某些时刻完全依靠 RL Token 中的语义信息来决策。

### 10.4 训练稳定性与调试

#### 常见训练崩溃模式

| 崩溃模式 | 现象 | 原因 | 解决方法 |
|----------|------|------|---------|
| **Q 值爆炸** | Q 值持续增长到 >1000 | 双 Critic 没对齐或 target network 更新太快 | 降低 τ，增加目标平滑噪声 |
| **策略坍塌** | Actor 输出恒为零或恒定值 | Actor 找到了 Q 面的伪高峰 | 增大 β，增大动作 dropout |
| **Critic 震荡** | Q 值在训练过程中反复跳变 | 学习率过高或 batch size 太小 | 降低 lr，增大 batch size |
| **动作过激** | Actor 输出远超出正常动作范围 | β 太小，BC 正则不足以约束 Actor | 增大 β，检查动作 clamp 范围 |

#### 训练监控指标

```
1. Q 值 (critic_q1, critic_q2):
   - 正常: 缓慢上升并趋于稳定
   - 异常: 迅速增长 → Q 值高估

2. Actor loss / Q 项 (actor_loss_q):
   - 正常: 逐渐下降（Q 值上升）
   - 异常: 剧烈波动 → 训练不稳定

3. BC loss (actor_loss_bc):
   - 正常: 维持在小值 (~0.05-0.2)
   - 异常: 趋近 0 → β 太大；激增 → β 太小

4. 动作标准差 (action_std):
   - 正常: 训练初期较大 → 逐渐衰减
   - 异常: 过早降为 0 → 探索不足

5. 平均 reward:
   - 正常: 逐渐上升并收敛
   - 异常: 不升反降 → 训练有问题
```

#### 梯度流动检查

```python
# 检查 Actor 的梯度是否正常
for name, param in actor.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")  # ← 这里有问题
```

常见断梯度原因：`torch.no_grad()` 包裹了不该包裹的地方；`detach()` 放在了错误位置；Actor 和 Critic 之间用了 `stop_gradient`。

#### AC 网络参数初始化

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        if getattr(m, 'is_output_layer', False):
            nn.init.xavier_uniform_(m.weight, gain=0.01)  # 输出层小初始化
            nn.init.zeros_(m.bias)
        else:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
```

**为什么输出层用更小的权重？** 训练开始时最好让 Actor 输出 ≈ 0（即修正量为 0），这样初始策略 ≈ VLA 参考动作。如果输出层初始化太大，初始动作会严重偏离 VLA，导致 Critic 在训练初期就要处理大量 out-of-distribution 的动作，很容易崩溃。

---

## 11. 超参数速查

### Stage 1

| 参数 | 值 |
|------|------|
| Encoder depth | 4 |
| Encoder width | 2048 |
| Encoder heads | 8 |
| mlp_ratio | 4.0 |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Training steps | 30,000 |

### Stage 2

| 参数 | 默认值 | 作用 | 调参方向 |
|------|--------|------|---------|
| **BC 系数 β** | 2.0 | 控制 Actor 偏离 VLA 的程度 | ↑ 任务简单/奖励稀疏时增大；↓ 需要大幅度改进时减小 |
| **Action Chunk C** | 10 | 每次决策预测多少步 | ↑ 高动态任务增大；↓ 需要精细控制时减小 |
| **折扣因子 γ** | 0.99 | 多看重远期奖励 | ↑ 长 horizon 任务；↓ 短 horizon 任务 |
| **Polyak τ** | 0.005 | 目标网络更新速度 | ↑ 训练不稳定时增大；↓ 训练抖动时减小 |
| **延迟更新频率 d** | 2 | 每 d 步 Critic 更新一次 Actor | ↑ Critic 难度大时增大 |
| **目标平滑噪声 σ** | 0.2 | 目标 Q 计算时对下一动作加的噪声 | ↑ 鼓励 Critic 平滑化；↓ 精度要求高时减小 |
| **Replay Buffer 容量** | 1e6 | 能存多少历史 transition | ↑ 任务模式多时增大 |
| **Batch Size** | 256 | 每次采样多少条 transition | ↑ 梯度更稳但更慢 |
| **Actor hidden** | 512 | Actor MLP 隐藏层宽度 | — |
| **Critic hidden** | 256 | Critic MLP 隐藏层宽度 | — |
| **Actor LR** | 3e-4 | Actor 学习率 | — |
| **Critic LR** | 3e-4 | Critic 学习率 | — |
| **Updates/step (G)** | 20 | 每收集一条 transition 复用次数 | — |
| **Smooth noise clip** | 0.5 | 目标平滑噪声裁剪范围 | — |

#### β 调参原则

β 是最重要的超参数，直接决定 Actor 的行为：

```
β → 0:     Actor 完全自由 → 可能找到 Q 面的"假峰" → 动作离奇
β = 1-5:   在 VLA 附近"微调" → RLT 的理想工作区间
β → ∞:     Actor 完全复制 VLA → 跟没做 RL 一样
```

**快速调参法**：
1. 先用 β=2 跑一次训练
2. 观察 rollout 中 Actor 的输出动作与 ref_actions 的 MSE：
   - MSE < 0.01：β 太大（Actor 几乎没改动作），降低 β
   - MSE > 0.5：β 太小（Actor 飞得太远），增大 β
   - MSE ≈ 0.05-0.2：合适的范围

---

## 12. 奖励函数设计

奖励函数设计在 RLT 代码中以 **3 个关键步骤**体现：

### Step 1：`env_step_fn` 的调用 — `rollout.py:121`

```python
# rollout.py:119-121
for step_idx in range(C):
    action = action_chunk[step_idx]
    next_obs, reward, done = self.env_step(action)  # ← 奖励函数在这里被调用
```

**你必须在调用 RLT 之前实现这个函数。** 代码本身不包含任何奖励函数的实现——它是一个抽象接口。

### Step 2：奖励存入 Replay Buffer — `rollout.py:141-147`

```python
self.replay.add(
    state,             # [2080]
    corrected_action,  # [320]
    episode_reward,    # [1]     ← 奖励标量
    next_state,        # [2080]
    done,              # [1]
)
```

### Step 3：奖励驱动 Critic 更新 — `critic.py:159`

```python
q_target = reward + gamma * (1 - done) * q_target_val
#           ↑
#     你设计的奖励信号
```

### 奖励函数影响的全链路图

```
你的代码                                       RLT 代码
┌───────────────────────────┐      ┌──────────────────────────────────┐
│  def my_reward_fn(action):│      │  rollout.py                      │
│    obs = robot.step(a)    │ ──→  │    reward = env_step(action)     │
│    dist = ...             │      │    replay.add(state, action, reward) │
│    reward = -dist + done  │      │                                  │
│                           │      │  replay.py                       │
│  RolloutRunner(           │      │    sample() → {"reward": [B, 1]} │
│    env_step_fn=my_reward  │      │                                  │
│  )                        │      │  critic.py                       │
│                           │      │    q_target = reward + γ·Q(s')   │
│                           │      │    loss = MSE(q_pred, q_target)  │
│                           │      │                                  │
│                           │      │  actor.py                        │
│                           │      │    loss = -Q + β·‖a - ref‖²     │
│                           │      │           ↑                      │
│                           │      │     Q 受奖励间接影响             │
└───────────────────────────┘      └──────────────────────────────────┘
        ↑                                    ↑
   你必须实现的函数                   自动执行，无需干预
```

---

## 13. 人工干预需求分析

### 各阶段人工干预全景

```
Stage 1（离线训练）:    🤖 准备数据 → ⚙️ 自动训练
Stage 2 准备工作:       🤖 设计奖励函数 + 设定环境 + 确定超参
Stage 2 训练循环:       ⚙️ 自动运行（仿真可完全无人）
Stage 2 结束判定:       🤖 观察指标，判断是否收敛
Stage 2 安全监控:       ⚙️ 仿真不需人 / 🤖 真实机器人建议有人
```

### 与其他方法对比

| 方法 | 人工干预需求 | 为什么 |
|------|------------|-------|
| **纯 VLA** | 只需准备演示数据 | 离线监督学习，一键训练 |
| **RLT** | 需设计奖励函数 + 可能的安全监控 | 在线 RL 需要 reward 信号，真实场景需安全看护 |
| **纯 RL** | 需设计奖励函数 + 大幅调参 + 安全监控 | 5B 参数探索空间大，调参困难 |
| **RLT vs 纯 RL** | **RLT 人工需求更低** | BC 正则化让 Actor 不偏离 VLA，训练更稳定 |

---

## 14. 执行时序分析

### 原版 VLA vs RLT 时序

| 指标 | 原版 VLA | VLA + RLT |
|------|---------|-----------|
| VLA 调用间隔 | 每 50 步 | 每 10 步 |
| VLA 调用频率 | 1x | **5x** |
| 推理开销占比 | ~23% | ~60% |

### 为什么实际还是能跑？

#### 1. 原版 VLA 开环跑 50 步本身就有问题

开环执行 50 步，后面的动作在已经变化的状态下执行，误差累积严重。**实际上不用 RLT，你也想更频繁地重规划**。

#### 2. 控制频率可以调整

```
RLT: 300ms(VLA) + 10×50ms(执行) = 800ms → 推理开销占比 37.5%
```

#### 3. 异步流水线可以隐藏推理延迟

```
时间轴:
  VLA推理: [██推理300ms██] [██推理300ms██] [██推理300ms██]
  执行:         [─10步200ms─] [─10步200ms─] [─10步200ms─]
```

只要满足 **10 步执行时间 ≥ VLA 推理时间**，推理延迟就能被完全隐藏：

| 控制频率 | 10 步耗时 | 能否隐藏 VLA 推理（300ms）？ |
|---------|----------|---------------------------|
| 50Hz（每步 20ms） | 200ms | ❌ 不能 |
| 33Hz（每步 30ms） | 300ms | ✅ 刚好匹配 |
| 20Hz（每步 50ms） | 500ms | ✅ 完全隐藏 |

### 核心 trade-off

RLT 用 5 倍的 VLA 推理量换来了每 10 步一次重规划的精度优势。对精密操作来说，跑得快但位置不准，不如慢一点但每步都对准。

---

## 15. 框架依赖检查

所有新增的 RLT 代码均为 **纯 JAX / Flax NNX 实现，零 PyTorch 依赖**。

| 文件 | 框架 | 是否与 openpi 一致 |
|------|------|:---:|
| `rl_token.py` | `flax.nnx` + `jax.numpy` | ✅ |
| `rlt_policy.py` | `jax` + `jax.numpy` | ✅ |
| `rlt_online_rl/actor.py` | `flax.nnx` + `jax` | ✅ |
| `rlt_online_rl/critic.py` | `flax.nnx` + `jax.numpy` | ✅ |
| `rlt_online_rl/learner.py` | `flax.nnx` + `jax` + `optax` | ✅ |
| `rlt_online_rl/rollout.py` | `jax` + `jax.numpy` | ✅ |
| `rlt_online_rl/replay.py` | `numpy`（纯数据） | ✅ |

---

## 16. 验证命令

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

# 5. RLT 推理流程测试（虚拟数据）
python examples/test_rlt_inference.py
```

---

## 17. Ubuntu / Conda 迁移指南

从 Windows + uv 切换到 Ubuntu + conda，核心代码**无需任何修改**。

### 对比总结

| 项目 | Windows + uv | Ubuntu + conda | 需要改？ |
|------|-------------|----------------|---------|
| 安装方式 | `uv sync` | `pip install -e .` | ❌ 不需要 |
| `pyproject.toml` | 使用 | 使用 | ❌ 不需要 |
| `uv.lock` | 使用 | 忽略 | ❌ 可删除 |
| 运行命令 | `uv run python script.py` | `python script.py` | ⚠️ 用法不同，代码不改 |
| 强制 CPU | 曾需要（已改为 `--cpu` 标志） | 不需要（默认 GPU） | ✅ 已解决 |
| `sys.path` hack | 需要 | 同样需要 | ❌ 不变 |

### Ubuntu + conda 环境准备

```bash
# 1. 激活 conda 环境
conda activate your_env

# 2. pip 安装 openpi（自动装 jax[cuda12]）
pip install -e .

# 3. 确认 JAX 能识别 GPU
python -c "import jax; print(jax.devices())"
# 期望输出: [cuda(id=0)] 或 [gpu(id=0)]

# 4. 运行测试
python examples/test_rlt_inference.py
```

### JAX 平台自动适配

`pyproject.toml` 中已用 `sys_platform` 做平台区分：

```toml
"jax[cuda12]==0.5.3; sys_platform == 'linux'"   # Ubuntu → CUDA
"jax[cpu]==0.5.3; sys_platform == 'win32'"      # Windows → CPU
```

---

## 18. 参考资料

- [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://browse-export.arxiv.org/abs/2604.23073) — RLT 原论文
- [openpi-RLT (GitHub)](https://github.com/Yyshadow/openpi-RLT) — 基于 openpi 的开源复现
- [π*₀.₆: a VLA That Learns From Experience](https://arxiv.org/html/2511.14759v2) — pi0.6 原论文
- [Physical Intelligence 官网](https://www.pi.website/)

---

> **文档生成信息**：本文档由 `docs_rlt/` 目录下的 5 个文档合并而成（`rlt_implementation_guide.md`, `codeAnalysis.md`, `code_revision_v1.md`, `code_revision_v1.zh.md`, `rlt_qa.md`），原始文件保持不变。

---

## 附录 A：Stage 1 训练深度解析 — RL Token 的信息代表性

> 本附录是第 9.1 节 Stage 1 内容的完整扩展版本，与 `rlt_implementation_guide.md` 中的对应章节一致。

### A.1 核心问题

> 为什么把 N × 2048 维的 token 序列压缩成单个 2048 维的向量，关键信息不会丢失？

这是 RLT 最核心的理论问题。下面从信息论、架构设计和训练动态三个层面详细分析。

### A.2 信息瓶颈视角：最小充分统计量

将 RL Token 理解为 VLA embeddings 的 **最小充分统计量（minimal sufficient statistic）**：

```
I(embeddings; RL_token) → max     ← 最大化信息保留（通过 MSE 重建）
H(RL_token) → 约束为 2048 维       ← 瓶颈维度约束（过滤噪声）
```

Stage 1 的训练目标等价于最大化互信息下界：

```
L_recon(ϕ) = E[‖decoder(encoder(embeddings)) - embeddings‖²]

→ 最小化 MSE = 最大化 I(embeddings; RL_token) 的下界
                （根据信息瓶颈理论，重建误差是互信息的变分上界）
```

当重建误差趋近于零时，RL Token 包含了**足以完整重建原始 embeddings 的全部信息**。这意味着原始 embeddings 中的任何对下游任务有意义的信息（物体位置、关节状态、任务指令）都被保留在了 RL Token 中。

**关键洞察**：压缩维度为 2048 不是因为信息量只需要 2048 维，而是因为原始 VLA 的隐藏维度就是 2048。RL Token 的容量与 VLA token 的每个位置的容量相同——信息损失来自"从 N 个位置到 1 个位置的聚合"，而非来自"维度降低"。

### A.3 架构设计如何保证信息保留

#### A.3.1 全注意力机制：[RL] Token 可以"看到"所有输入

```python
# rl_token.py:209-217
rl_tok = jnp.broadcast_to(jnp.asarray(self.rl_token), (B, 1, D))
x = jnp.concatenate([embeddings, rl_tok], axis=1)  # [B, N+1, D]

# 全注意力 mask（所有 token 互相可见）
mask = jnp.ones((B, N + 1, N + 1), dtype=jnp.bool_)
```

与池化操作（mean/max pooling）的本质区别：

| 方法 | 信息聚合方式 | 问题 |
|------|-------------|------|
| **Mean Pooling** | 所有位置取平均，权重相同 | 假设所有 token 同等重要—破坏空间结构 |
| **Max Pooling** | 每维取最大值 | 只保留最显著特征—忽略细粒度信息 |
| **RL Token (Ours)** | 每个 [RL] token 通过注意力自适应加权 | **保留"哪些 token 重要"的选择权给模型学习** |

全注意力 mask 意味着 [RL] token 可以通过 4 层 Transformer 的注意力头，学习到对每个输入 token 的差异化权重。对于抓取任务，视觉 token 可能获得更高权重；对于精密操作，动作 token 可能更重要。

#### A.3.2 4 层 Transformer 的容量分析

```
宽度 2048  ×  深度 4 层  ×  8 个注意力头

每层参数量 ≈ 4 × 2048² (QKV+O) + 2 × 2048 × 8192 (MLP, mlp_ratio=4.0)
           ≈ 16M + 33M ≈ 49M 参数/层
总参数量 ≈ 4 × 49M ≈ 196M 参数
```

从容量上看：
- **足够做复杂聚合**：196M 参数可以学习到各 token 之间复杂的依赖关系
- **不会过拟合 VLA embeddings**：相对于 VLA（5B 参数），Encoder 的 196M 是小模块，迫使它学习通用压缩模式而非记忆特定样本

#### A.3.3 可学习的 [RL] 特殊 Token

初始化：`self.rl_token = nnx.Param(jax.random.normal(...) * 0.02)`
- 初始化为小随机值，避免初始偏差
- 在训练中通过梯度下降优化（从重建损失中学习"哪些信息必须保留"）

#### A.3.4 位置编码保留空间结构

```python
self.pos_embed = nnx.Param(jax.random.normal(..., (1, max_len, config.width)) * 0.02)
x = x + jnp.asarray(self.pos_embed[:, :N + 1, :])
```

可学习的绝对位置编码让 [RL] token 不仅能选择"关注哪个 token"，还能选择"关注哪个位置的 token"。例如：前 768 个 token 是视觉特征，后 200 个 token 是语言特征 — [RL] token 可以通过位置区分它们。

### A.4 Decoder 的信息压力测试

Decoder 的存在是确保信息保留的**最关键机制**：

```
输入 → Encoder → [2048] → Decoder → 重建 N × 2048
                                          ↓
                                    MSE (N × 2048 个值)
```

这里的 MSE 是 **N × 2048 = ~2M 个标量值的平均误差**。Decoder 必须从单个 2048 维向量中重建出约 2M 个值。

**信息量计算**：

| 环节 | 信息容量 | 说明 |
|------|---------|------|
| 输入 embeddings | N × 2048 × 16 bits ≈ 16 Mbits | 假设 float16, N≈1000 |
| RL Token | 2048 × 16 bits ≈ **32 Kbits** | 压缩比 ≈ 500:1 |
| 重建输出 | 同输入 ≈ 16 Mbits | 恢复原始大小 |

表面上压缩比高达 500:1，但请注意：
1. **VLA embeddings 并非独立同分布**：N 个 token 之间存在大量冗余（相邻视觉 patch 高度相关，语言 token 之间存在句法依赖）
2. **Encoder 的注意力机制利用相关性**：相似的 token 可以被压缩到同一子空间的不同维度
3. **实际信源熵远小于 N × 2048 × 16 bits**：有效信息量可能只有几十 Kbits

这正是 **Decoder 重建任务的意义**：
- 如果 MSE 降到很低 → 证明 RL Token 确实包含了足够的信息来重建全部 embeddings
- 如果 MSE 降不下去 → 说明 2048 维是瓶颈，需要增大容量或改架构

**经验结果**（来自 RLT 论文）：在 30K 训练步后，重建 MSE 收敛到 ~0.01，说明 2048 维对于 VLA token embeddings 的"有效信息"是足够的。

### A.5 自回归式重建 vs 一次性重建

```python
# Decoder 的训练方式（rl_token.py:243-246）
self.query_tokens = nnx.Param(         # 可学习的查询 token
    jax.random.normal(rngs.params(), (1, max_len, config.width)) * 0.02
)
```

RLT Decoder 不是简单的"把 RL Token 复制 N 份过 MLP"，而是使用 **可学习的独立查询 token + cross-attention**：

```
RL Token [2048]
    │
    ▼
Cross-Attention: 查询 = 可学习 query_tokens [N, 2048]
                 键/值 = RL Token [1, 2048]
    │
    ▼
重建 embeddings [N, 2048]
```

**为什么不是线性投影？** 线性投影假设 N 个输出位置**相互独立**，但 VLA 的 token 序列有明显的结构（前 768 是图像，接着是文本，最后是动作）。Cross-attention + 独立 query 让 Decoder 可以：
1. 不同的 query token 关注 RL Token 的不同子空间
2. 自回归式的逐位置重建捕捉 token 之间的依赖关系
3. 从单个 RL Token 中"解压缩"出结构化信息

**这相当于给 Encoder 施加了额外的结构约束**：RL Token 不能只是"乱序的信息包"，而必须以某种可解构的方式组织信息，让不同位置的 query 能提取到对应位置所需的信息。

### A.6 与其他压缩方案的对比

| 方案 | 表达能力 | 结构保持 | 信息保留上限 | 计算开销 | RLT 选择的原因 |
|------|---------|---------|------------|---------|--------------|
| **Mean Pooling** | ❌ 低 | ❌ 丢失 | 低（均匀混合） | 极低 | — |
| **Max Pooling** | ❌ 低 | ❌ 丢失 | 低（只保留极值） | 极低 | — |
| **Attention Pooling** | ✅ 中 | ⚠️ 部分 | 中（单层线性加权） | 低 | — |
| **Transformer Encoder + [RL]** | ✅✅ 高 | ✅ 保留 | **高（非线性交互+位置感知）** | 中 | ✅ |
| **所有 token 拼接（不压缩）** | — | — | ✅ 完全保留 | ❌❌ 极高（~2M 维） | Actor 无法处理 |

**为什么不能直接用所有 token？** Actor 是 2-3 层 MLP（512 隐藏层）：
- 输入 ~2000 个 2048 维 token → 4M 维输入 → 参数爆炸（512 × 4M ≈ 2B）
- 即使内存允许，MLP 也无法有效处理可变长度的序列
- **Transformer Encoder 将变长序列转化为定长向量，是必要的维度规约**

### A.7 训练动态与信息保留的验证

#### A.7.1 训练曲线解读

```
MSE Loss
│
│   ╲
│    ╲
│     ╲
│      ╲
│       ╲
│        ╲──────── 收敛到 ≈ 0.01
└────────────────── 训练步数
      30K
```

| 阶段 | 现象 | 信息状态 |
|------|------|---------|
| 训练初期 | MSE 高（~1.0） | RL Token 尚未学会压缩，信息大量丢失 |
| 训练中期 | MSE 快速下降 | Encoder 学习选择性聚合，Decoder 学习解压缩 |
| 训练后期 | MSE 趋于平稳 | RL Token 达到信息容量上限，收敛到近似最优压缩 |
| 收敛后 | MSE ≈ 0.01 | 重建误差极小 → **RL Token 包含了足够的信息** |

#### A.7.2 信息保留的充分条件

从理论上，如果重建损失满足：

```
E[‖decoder(encoder(embeddings)) - embeddings‖²] < ε
```

则对于任意 Lipschitz 连续的 Actor 函数 f：

```
|f(decoder(encoder(embeddings))) - f(embeddings)| < L · ε
```

其中 L 是 f 的 Lipschitz 常数。这意味着：
- 如果 Actor 只依赖于 embeddings 中的信息
- 且重建误差足够小
- 那么 Actor 在 RL Token 上的表现与在原始 embeddings 上**几乎相同**

即：**RL Token 是原始 embeddings 的 ε-近似充分统计量**。

#### A.7.3 如何验证信息保留

在实际实现中，可以通过以下实验验证：

```python
# 实验 1：重建质量检查
reconstructed = decoder(rl_token)
recon_error = jnp.mean(jnp.square(reconstructed - embeddings))
print(f"重建 MSE: {recon_error:.6f}")
# 期望: < 0.05

# 实验 2：一致性检查（不同视角的同一场景，RL Token 应相似）
rl_token_1 = encoder(embeddings_1)
rl_token_2 = encoder(embeddings_2)
similarity = jnp.dot(rl_token_1, rl_token_2.T) / (norm(rl_token_1) * norm(rl_token_2))
print(f"相似场景 RL Token 余弦相似度: {similarity:.4f}")

# 实验 3：对比 RL Token 与原始 embeddings 在简单探测任务上的表现
accuracy_rl = probe(rl_token, target)
accuracy_raw = probe(embeddings.mean(axis=1), target)
print(f"RL Token 探针准确率: {accuracy_rl:.3f}, Mean Pooling: {accuracy_raw:.3f}")
```

### A.8 总结：为什么 1 个 RL Token 就够了

```
压缩过程：
N × 2048  ← VLA 的完整表示
    │
    ├─ 图像 token (768个)：高度冗余（相邻 patch 几乎相同）
    ├─ 文本 token (~200个)：语义紧凑但维度高
    └─ 动作 token (50个)：时序上平滑变化，冗余度高
    │
    ▼
4 层 Transformer Encoder ← 自适应选择、去冗余、重组合
    │
    ▼
1 × 2048  ← RL Token
    │
    ├─ ✗ 不是"压缩"—而是"提炼"
    ├─ ✗ 不是"降维"—而是"聚合"
    └─ ✓ 是"充分统计量"—包含决策所需的全部信息
```

| 理论依据 | 解释 |
|---------|------|
| **信息瓶颈理论** | 最大化 I(RL_token; embeddings) ≈ 最小化 MSE |
| **充分统计量** | 重建损失趋近 0 → RL Token 包含重建所需的全部信息 |
| **架构保证** | 全注意力 + 4 层 Transformer + 可学习 [RL] token → 自适应信息聚合 |
| **Decoder 压力测试** | 从 1 个 token 重建 N 个 token → 迫使 Encoder 保留所有关键信息 |
| **维度分析** | 2048 = VLA 隐藏维度（非降维，仅聚合位置） |
| **经验验证** | 论文中 MSE 收敛到 ~0.01，下游 RL 性能接近不压缩的上限 |

> **一句话结论**：RL Token 不是"暴力压缩"，而是 Transformer Encoder 通过全注意力在所有 token 之间做**自适应信息提炼**，通过 Decoder 重建损失的监督信号保证不遗漏关键信息。2048 维不是瓶颈——真实的容量限制在注意力的选择性聚合，而非维度本身。
