# RLT 常见问题问答

## Q1: RLT 的原理就是在 VLA 预测的动作局部做微小调整，这么理解对吗？

**对，完全正确。** 下面从理论到实现详细说明。

### 一句话

> **RLT = VLA 提供"大致正确的动作" + Actor 在局部做"微小修正"使动作更优**

### 三种策略对比

```
┌─────────────────────────────────────────────────────────┐
│  纯 VLA（如 pi0.5/pi0.6）：从演示数据中模仿行为          │
│    优点：大局观正确（知道任务是什么、物体在哪）           │
│    缺点：无法通过试错自我改进（拧螺丝差 2°，它就学不会） │
├─────────────────────────────────────────────────────────┤
│  纯 RL（SAC/TD3）：从零开始探索                          │
│    优点：可以优化到极致                                  │
│    缺点：5B 参数 + 高维动作空间 → 探索空间 ≈ ∞ → 不可行 │
├─────────────────────────────────────────────────────────┤
│  RLT：VLA 画了个圈，Actor 在圈里微调                      │
│    优点：既保持 VLA 的大局观，又能在线优化局部细节        │
└─────────────────────────────────────────────────────────┘
```

### 结合代码的详细说明

#### 1. VLA 提供"参考动作"和"场景理解"

在 `pi0.py` 的 `get_token_embeddings()` 中：

```python
# VLA 前向传播 → 同时输出两样东西
all_embeddings, ref_actions = self.vla_embed_fn(observation)
#   ↑                            ↑
#   场景的完整表示 (N×2048)       参考动作 (50×32)
#   — "任务理解、物体位置、       — "大致的行动方案"
#     机器人关节状态"
```

VLA 的参考动作 `ref_actions [50×32]` 是按照演示数据分布预测的最可能动作序列。这个序列**大致正确**——能把物体拿起来、往目标方向移动，但精度不够（比如插 USB 接口差 1mm）。

#### 2. RL Token：提取"场景理解"供 Actor 使用

在 `rl_token.py` 的 `RLTEncoder` 中：

```python
# VLA 的 N×2048 维 token 序列包含了对场景的完整理解
# 但 Actor（2 层 MLP）吃不下几千个 token
# → 用 Transformer Encoder 压缩成单个向量
rl_token = encoder(all_embeddings)  # [B, N, 2048] → [B, 2048]
```

这个 RL Token 是 VLA 对当前场景理解的 **"精华摘要"**。

#### 3. Actor：在参考动作附近做微调

在 `rlt_online_rl/actor.py` 的 `compute_loss()` 中：

```python
# Actor 的输入是两个东西的拼接
x = jnp.concatenate([state, ref_actions], axis=-1)
#   ↑                       ↑
#   RL Token + proprio       VLA 的参考动作 (1600 维 = 50×32)
#   [2080]                   — 作为"锚点"
#
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

#### 4. 为什么是 C=10 而不是 H=50？

```python
action_chunk: int = 10  # Actor 只预测前 10 步
action_horizon: int = 50  # VLA 预测 50 步
```

这也是微调的体现：

- VLA 预测完整 50 步的"宏观轨迹"
- Actor 只截取前 10 步做修正——因为 10 步之内环境不会大变，VLA 的参考已经够准
- 执行 10 步后重新观测，再次 VLA + Actor 修正

这比直接修正 50 步更合理：**远期的动作不需要微调，因为未来会有新的观测进来重新规划**。

### 完整推理流程

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

### 物理类比：投篮

| 角色 | 类比 | 说明 |
|------|------|------|
| **VLA** | 教练示范标准动作 | 提供了正确的发力方式、出手角度 |
| **RL Token** | 球员对篮筐位置的感觉 | 对当前场景的理解（距离、角度） |
| **Actor** | 微调出手角度 | 在教练示范的基础上，根据实际感觉调整 |
| **BC 正则化** | "别改太多，先按教练的来" | 防止大改导致动作变形 |
| **Q 值** | "这球进了" | 实际结果的反馈 |
| **纯 RL** | 让新手自己瞎投摸索 | 5B 参数的探索空间不现实 |

**RLT = VLA 保证你至少投得不错 + Actor 帮你多投进几个**

---

## Q2: 什么是 BC 正则化？

BC 正则化中的 **BC = Behavioral Cloning（行为克隆）**。

### 通俗理解

**BC 正则化 = "别离 VLA 的答案太远" 的惩罚项。**

### 数学形式

在 RLT 的 Actor 损失函数中：

```
L_actor = -Q(s, a) + β · ‖a - a_ref‖²
          ↑               ↑
         "变更好"        "别学歪"
          (RL 目标)       (BC 正则化)
```

第二项就是 **BC 正则化**：计算 Actor 输出 `a` 和 VLA 参考动作 `a_ref` 之间的 **MSE（均方误差）**，作为一个惩罚项加到损失里。

对应代码（`rlt_online_rl/actor.py` 的 `compute_loss`）：

```python
bc_loss = jnp.mean(jnp.square(mean - ref_actions[:, :self.action_dim]))
#           ↑
#    Actor 输出     VLA 参考动作（前 C 步）
#   应尽量接近
loss = q_loss + beta * bc_loss
```

### 为什么需要它？

#### 问题：Q 函数在分布外不可靠

Critic 是**用 Actor 收集的数据训练的**。如果 Actor 输出了 VLA 从未见过的奇怪动作，Critic 没有数据知道这个动作好不好，就会给出不可靠的高 Q 值（adversarial exploitation）：

```
Q 面（价值平面）：
                   ╱╲
  VLA 动作 →  ╱╴╴╴╴╴╲╴╴  真实 Q 面（平滑）
              ╱         ╲
             ╱           ╲
  Actor 乱试 →  ╱╲        ╲  ← Q 面的"假高峰"
  （分布外）    ╱  ╲        ╲    Critic 没数据，瞎猜的
```

BC 正则化就是用一根"绳子"把 Actor 拴在 VLA 附近——**VLA 输出的动作是在训练数据分布内的**，Critic 对这个区域的评估是可靠的。

#### 类比：炒菜

```
你让 AI 学炒菜：
  VLA 参考：盐 3 克，酱油 10ml     ← 菜谱教的（正确的分布内）
  Actor 输出：盐 2.8 克，酱油 9.5ml  ← BC 正则化拉着，小范围调整
  Actor 如果乱来：盐 50 克，酱油 0ml  ← 没有 BC，Q 可能说"好吃"（但实际是灾难）
```

### 在 RLT 中的角色

| 不加 BC 正则化 | 加 BC 正则化 |
|---|---|
| Actor 追求 Q 最大化 → 找到 Q 面的假高峰 | Actor 在 VLA 附近搜索 → 可靠区域 |
| 可能在"从没见过的动作"上获得虚高 Q 值 | 只在 VLA 见过的动作空间微调 |
| Beta=0 时 → 相当于标准 TD3，容易发散 | Beta=2.0 → 在 VLA 附近做安全微调 |

### Beta 系数的作用

```python
beta = 2.0  # 默认值
```

| Beta | 行为 | 效果 |
|------|------|------|
| β=0 | Actor 完全自由 | Q 估计爆炸，容易发散 |
| β→∞ | Actor 完全复制 VLA | 等于没做 RL |
| **β=2.0** | **在参考动作附近微调** | **RLT 的默认设置，安全有效** |

---

## Q3: 为什么 RLT 训练使用 Encoder 后还需要 Decoder？

Decoder 只用于 **Stage 1（离线预训练）**，在线 RL 阶段完全不用。它的作用是提供监督信号。

### 核心逻辑：自编码器结构

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

### 没有 Decoder 会怎样？

Encoder 训练缺少监督信号，容易"偷懒"：

- **模式坍塌**：不管输入是什么都输出相同的 RL Token
- **信息丢失**：RL Token 只编码了部分信息

Decoder 通过要求**能从 RL Token 重建原始 embeddings**，间接迫使 RL Token 保留足够多的信息，类似于 **VAE 的瓶颈训练**：

```
L_recon(ϕ) = E[ Σ_i ‖decoder(encoder(embeddings))_i - embeddings_i‖² ]
```

### 训练完成后

```
Stage 1 结束 → Decoder 丢弃（不再需要）
             → Encoder 冻结（RL Token 已学会压缩信息）
             → Stage 2 只用 Encoder 提取 RL Token
```

Decoder 就像脚手架——建完房子就拆掉了。

---

## Q4: 已经有基于 pi0.5 训好的 VLA，需要重新训练以支持 RLT 吗？

**不需要。** 这是 RLT 设计的核心优势。

### 原因

1. **RLT 的前提就是 VLA 冻结**——已有 pi0.5（或 pi0.6）可以直接用作基座，不需要任何改动。

2. **RLT 需要的中间表示已经存在于 VLA 中**——RLT 只需要 VLA 最后一层 Transformer 输出的 token embeddings，`get_token_embeddings()` 只是把它们提取出来，没有修改 VLA 本身的任何参数或结构。

3. **维度不匹配已考虑**——pi0.5/pi0.6 两个 Expert 宽度不同（2048/1024），只需一个在 VLA **外部**的小投影层统一即可。

| 问题 | 答案 |
|------|------|
| 需要重新训练 VLA？ | ❌ 不需要 |
| 需要修改 VLA 结构？ | ❌ 不需要 |
| 需要修改 VLA 权重？ | ❌ 不需要 |
| 需要什么？ | ✅ 提取已有的中间表示 + 小投影层统一维度 |

> **注意**：RLT 原论文使用 **pi0.6** 作为基座模型（Gemma 3 4B 主干 + 860M action expert），但我们的实现基于 openpi 的 **pi0.5**（Gemma 2B + 300M action expert）。RLT 方法与具体 VLA 版本无关，在任一版本上均可实现。

---

## Q5: RLT 的训练需要人工干预吗？

**需要，但干预集中在训练开始前的准备阶段和训练过程中的安全监控，训练循环本身是自动的。** 下面按阶段详细拆解。

### 全景概览

```
┌──────────────────────────────────────────────────────────────────┐
│                    RLT 训练全流程                                  │
│                                                                   │
│  ┌─ Stage 1（离线）─────────────────────────────────────────┐     │
│  │  加载 demo 数据 → 训练 Encoder-Decoder → 冻结 Encoder     │     │
│  │  🤖 人工介入：几乎为零（数据准备好后一键运行）            │     │
│  └───────────────────────────────────────────────────────────┘     │
│                            ↓                                      │
│  ┌─ Stage 2（在线）─────────────────────────────────────────┐     │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐         │     │
│  │  │ 准备工作   │ →  │ 训练循环   │ →  │ 结束判定   │         │     │
│  │  └───────────┘    └───────────┘    └───────────┘         │     │
│  │     🤖 人工          ⚙️ 自动          🤖 人工            │     │
│  └───────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

---

### Stage 1（离线训练）：几乎零人工干预

| 环节 | 需要人工？ | 说明 |
|------|-----------|------|
| 准备 demo 数据 | ✅ 需要（但通常已有） | 与训练 VLA 时使用的数据相同，不需要额外采集 |
| 启动训练 | ❌ 不需要 | 一键运行 `train_rlt_token.py` |
| 训练过程 | ❌ 不需要 | 标准监督学习，MSE 损失自动收敛 |
| 超参数选择 | ⚠️ 一次性的 | encoder depth/width/lr 通常使用推荐默认值即可，无需反复调参 |

> **结论**：Stage 1 不需要人在训练过程中盯着。准备好数据后，跑完脚本即可。

---

### Stage 2（在线 RL）：准备工作需要人，训练循环不需要

#### 1. 准备工作（需要人工干预）

这是 RLT 训练中**最主要的人工工作**：

##### 1a. 设计奖励函数 —— **最重要的必做工作**

```python
def compute_reward(obs, task_spec) -> float:
    # 这是你必须自己写的函数 —— RLT 不知道"什么算好"
    #
    # 典型设计模式:
    #
    # 稀疏奖励（简单任务）:
    #   reward = 1.0 if task_success(obs) else 0.0
    #
    # 稠密奖励（精细操作）:
    #   dist = np.linalg.norm(obs['ee_pos'] - obs['target_pos'])
    #   reward = -dist  # 越近越好
    #   + 1.0 if task_success(obs) else 0.0  # 完成奖励
    #
    # 混合奖励（推荐）:
    #   reward = w1 * stage_reward + w2 * smoothness_penalty
    pass
```

奖励函数的设计直接影响 RLT 的微调方向，需要你对任务有深入理解。这是 RLT 训练中**不可自动化**的核心环节。

| 奖励类型 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| **稀疏奖励**（0/1） | 设计简单，不易 hack | 学习信号稀疏，收敛慢 | 简单任务或仿真环境 |
| **稠密奖励**（连续值） | 学习信号密集，收敛快 | 设计困难，可能被 hack | 精细操作任务 |
| **混合奖励**（阶段奖励+稀疏） | 折中方案 | 需要设计阶段划分 | 复杂长程任务 |

##### 1b. 设定环境 —— **可能需要人工**

| 场景 | 需要人工？ | 说明 |
|------|-----------|------|
| **仿真**（Simulation） | ❌ 不需要 | 环境重置、初始化可自动完成 |
| **真实机器人**（Real Robot） | ✅ **需要** | 摆放物体、重置场景、处理异常——这部分很难完全自动化 |

> 论文中的实验（15min-2h 在线数据）使用的是真实机器人，通常需要人在场景间重置物体位置。

##### 1c. 选择超参数 —— **需要人工，但通常复用默认值**

RLT 的核心超参数在 `code_revision_v1.md` 中有推荐默认值：

```python
BETA = 2.0           # BC 正则化系数 — 控制"偏离 VLA 的程度"
GAMMA = 0.99         # 折扣因子 — 控制"远见程度"
POLICY_DELAY = 2     # Actor 更新延迟 — TD3 标准技巧
G = 20               # 内循环更新次数 — 样本复用程度
```

| 参数 | 通常需要调？ | 原因 |
|------|------------|------|
| `beta` | ⚠️ **可能需要** | 依赖具体任务：高精度任务（插线）可能需要降低 beta 让 Actor 更自由 |
| `gamma` | ❌ 通常不用 | 0.99 是 RL 任务的通用值 |
| `learning_rate` | ⚠️ 偶尔需要 | 如果训练不稳定可微调 |
| 其他 | ❌ 通常不用 | TD3 标准参数 |

#### 2. 训练循环（无需人工干预）

一旦启动，Rollout + Learner 异步循环自动运行：

```
┌─ Rollout 进程 ──────────────────┐
│  while True:                     │
│    obs = env.reset()             │
│    VLA → Encoder → Actor → 执行  │
│    收集 reward → 存入 Buffer     │
│  ── 完全自动，无需人介入          │
└──────────────────────────────────┘
         ↓ (数据流)
┌─ Learner 进程 ──────────────────┐
│  while True:                     │
│    Buffer.sample() → 更新 Critic │
│    → 更新 Actor → 软更新目标网络  │
│  ── 完全自动，无需人介入          │
└──────────────────────────────────┘
```

**两个例外情况**：

| 情况 | 说明 | 处理方式 |
|------|------|---------|
| **Buffer 为空** | 训练刚开始，Learner 还没有数据可学 | Learner 会自动等待，不报错 |
| **Rollout 比 Learner 快** | Buffer 满了，Learner 还没消费完旧数据 | 覆盖旧数据（circular buffer 的默认行为），或调大 buffer 容量 |

#### 3. 训练结束判定（需要人工干预）

RLT 没有内置的"自动停止"机制，需要人判断训练何时收敛：

```python
# 你需要关注的指标（建议可视化记录）:
#
# 1. Episode Reward 曲线 —— 总奖励是否稳定在某个水平？
# 2. Q 值估计 —— 是否趋于稳定但不发散？
# 3. Actor loss —— BC 正则化项是否稳定？
# 4. 成功率 —— 实际任务完成率是否达到目标？
#
# 停止条件示例:
#   - 连续 50 个 episode 的成功率 > 95%
#   - 或奖励曲线连续 100 步不再上升
#   - 或达到预设的最大训练步数
```

### 安全监控：仿真 vs 真实机器人

这是仿真和真实场景最大的区别：

| 场景 | 是否需要安全监控？ | 原因 |
|------|------------------|------|
| **仿真** | ❌ 不需要 | 即使 Actor 输出危险动作，也不会损坏任何设备 |
| **真实机器人** | ✅ **强烈建议** | 虽然 BC 正则化限制了 Actor 偏离范围，但仍有小概率探索出危险动作（如关节超限、碰撞） |

### 一句话总结各阶段的人工干预需求

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
| **纯 VLA（如 pi0.5/pi0.6）** | 只需准备演示数据 | 离线监督学习，一键训练 |
| **RLT** | 需设计奖励函数 + 可能的安全监控 | 在线 RL 需要 reward 信号，真实场景需安全看护 |
| **纯 RL（SAC/TD3）** | 需设计奖励函数 + 大幅调参 + 安全监控 | 5B 参数探索空间大，调参困难，训练不稳定 |
| **RLT vs 纯 RL** | **RLT 人工需求更低** | BC 正则化让 Actor 不偏离 VLA，训练更稳定，调参更少 |

### 最终结论

> **RLT 的训练人工干预集中在"训练开始前"（奖励函数设计）和"训练结束后"（收敛判断），训练循环本身是全自动的。真实机器人场景需要人做安全监控，仿真场景可以完全无人值守。相比纯 RL，RLT 需要的人工干预大幅减少——因为 BC 正则化让训练稳定，不需要费力调参来防止发散。**

---

## Q6: 奖励函数设计在代码中体现在哪些步骤？

奖励函数设计在 RLT 代码中以 **3 个关键步骤**体现，分布在 rollout、replay buffer 和 critic 中，形成完整的"奖励驱动学习"链路。

### Step 1：`env_step_fn` 的调用 — `rollout.py:121`

这是奖励函数的**触发点**，也是**你需要实现的地方**：

```python
# rollout.py:119-121
for step_idx in range(C):
    action = action_chunk[step_idx]
    next_obs, reward, done = self.env_step(action)  # ← 奖励函数在这里被调用
```

`env_step_fn` 是 `RolloutRunner.__init__` 接收的一个 `Callable`：

```python
# rollout.py:50
env_step_fn: Callable[[np.ndarray], tuple[dict, float, bool]]
#                                ↑           ↑      ↑
#                             动作输入    奖励标量  是否终止
```

**你必须在调用 RLT 之前实现这个函数。** 代码本身不包含任何奖励函数的实现——它是一个抽象接口。举个例子，精插任务中你可能这样写：

```python
def my_reward_fn(action: np.ndarray) -> tuple[dict, float, bool]:
    obs = robot.step(action)                            # 执行动作
    dist = np.linalg.norm(obs["ee_pos"] - target_pos)   # 计算距离
    reward = -dist                                      # 距离越近奖励越高
    reward += 10.0 if task_success(obs) else 0.0        # 完成额外奖励
    return obs, reward, task_success(obs)
```

### Step 2：奖励存入 Replay Buffer — `rollout.py:141-147`

环境返回的 `reward` 被存入 Replay Buffer，作为 Critic 训练的"标签"（ground truth）：

```python
# rollout.py:141-147
self.replay.add(
    state,             # [2080]  当前状态
    corrected_action,  # [320]   执行的动作块
    episode_reward,    # [1]     ← 奖励标量（当前简化实现使用累计奖励）
    next_state,        # [2080]  下一状态
    done,              # [1]     是否终止
)
```

**注意这里的简化实现**：`rollout.py:143` 当前用的是 `episode_reward`（整个 episode 的累计奖励），而非 `step_idx` 循环内单步的 `reward`。这是需要改进的地方——正确的做法应该是**在循环内存储每步的 reward**，而不是在 chunk 级别用累计值。

奖励数据在 Buffer 中的完整流动：

```
ReplayBuffer (replay.py:35-39)
  _reward = np.zeros((capacity, 1), dtype=np.float32)  # [N, 1]

  → sample() 返回:
    {
      "state":      [batch, 2080],   # 当前状态
      "action":     [batch, 320],    # 执行的动作
      "reward":     [batch, 1],      # ← 奖励信号
      "next_state": [batch, 2080],   # 下一状态
      "done":       [batch, 1],      # 终止标志
    }
```

### Step 3：奖励驱动 Critic 更新 — `critic.py:159`

这是奖励函数**真正影响学习的地方**。Critic 的 TD target 将奖励纳入计算：

```python
# critic.py:159
q_target = reward + gamma * (1 - done) * q_target_val
#           ↑                                ↑
#     你设计的奖励信号                 下一状态的估计价值
```

完整的 Critic 损失函数：

```
L_Q = MSE(q1_pred, q_target) + MSE(q2_pred, q_target)

其中 q_target = reward_t + γ · (1-done) · min(Q₁'(s', a'), Q₂'(s', a'))
                    ↑
           你设计的奖励函数输出
           直接决定 Critic 的学习目标
```

### 奖励函数影响的全链路图

```
你的代码                                       RLT 代码
┌───────────────────────────┐      ┌──────────────────────────────────────────┐
│                           │      │                                          │
│  def my_reward_fn(action):│      │  rollout.py                               │
│    obs = robot.step(a)    │ ──→  │    reward = env_step(action)             │
│    dist = ...             │      │    replay.add(state, action, reward, ...) │
│    reward = -dist + done  │      │                                          │
│                           │      │  replay.py                                │
│  RolloutRunner(           │      │    sample() → {"reward": [B, 1]}         │
│    env_step_fn=my_reward  │      │                                          │
│  )                        │      │  critic.py                                │
│                           │      │    q_target = reward + γ·Q(s')           │
│                           │      │    loss = MSE(q_pred, q_target)          │
│                           │      │                                          │
│                           │      │  actor.py                                 │
│                           │      │    loss = -Q + β·‖a - ref‖²             │
│                           │      │           ↑                              │
│                           │      │     Q 受奖励间接影响                     │
└───────────────────────────┘      └──────────────────────────────────────────┘
        ↑                                    ↑
   你必须实现的函数                   自动执行，无需干预
```

### 关键结论

| 代码位置 | 作用 | 需要你写？ |
|---------|------|-----------|
| `rollout.py:121` `self.env_step(action)` | **奖励函数的调用入口** | **✅ 必须实现 `env_step_fn`** |
| `rollout.py:141` `replay.add(reward=...)` | 奖励存入 Buffer | ❌ 框架自动 |
| `replay.py:63` `sample()["reward"]` | 奖励采样 | ❌ 框架自动 |
| `critic.py:159` `q_target = reward + γ·...` | 奖励驱动 TD target | ❌ 框架自动 |

> **一句话**：奖励函数设计在代码中只体现为一个接口——你在 `env_step_fn` 中实现的 `reward` 标量。一旦写好，RLT 的整个训练循环自动用它驱动 Critic 和 Actor 的更新。当前代码的 `rollout.py:143` 在 chunk 级别存储累计奖励是一个简化实现，实际部署时需要注意改进。

---

## Q7: VLA 原来每 50 步重规划一次，加入 AC 网络后每 10 步就要重规划，VLA 调用频率变成 5 倍，执行会不会被拖慢？

**会引入额外开销，但不一定会拖慢实际执行。** 下面从时序分析原因和解决方案。

### 原版 VLA 的时序

原版 VLA 用 flow matching 生成 50 步动作，需要 10 步迭代去噪：

```
VLA 推理一次（5B 参数，10 步去噪）:
  ┌──────────────────────────────────────┐
  │ Transformer 前向 × 10                │  ← 这是最贵的部分
  └──────────────────────────────────────┘
      耗时 ≈ 150~300ms（A100/H100）

机器人执行 50 步:
  每步 ≈ 20~50ms（20-50Hz 控制频率）
  50 × 20ms = 1000ms

─────────────────────────────────────────
一个完整周期: 300ms（推理）+ 1000ms（执行）= 1.3s
推理开销占总周期比例: 300/1300 ≈ 23%
```

### 加入 RLT 后的时序

```
VLA 推理一次: 300ms  ← 不变，还是 10 步去噪
Actor 推理一次: <1ms  ← 2-3 层 MLP，几乎免费

机器人执行 10 步:
  10 × 20ms = 200ms

─────────────────────────────────────────
一个周期: 300ms（推理）+ 200ms（执行）= 500ms
推理开销占比: 300/500 ≈ 60%
```

| 指标 | 原版 VLA | VLA + RLT |
|------|---------|-----------|
| VLA 调用间隔 | 每 50 步 | 每 10 步 |
| VLA 调用频率 | 1x | **5x** |
| 推理开销占比 | ~23% | ~60% |

### 为什么实际还是能跑？

#### 1. 原版 VLA 开环跑 50 步本身就有问题

开环执行 50 步，后面的动作在已经变化的状态下执行，误差累积严重。**实际上不用 RLT，你也想更频繁地重规划**，只是 VLA 太贵做不到。RLT 相当于用便宜的 MLP 把"重规划精度"提上来，但 VLA 端的推理成本确实省不了。

#### 2. 控制频率可以调整

机器人控制频率不是固定的。如果把每步从 20ms 降到 50ms（20Hz → 10Hz）：

```
RLT: 300ms + 10×50ms = 800ms  →  推理开销占比 37.5%
```

慢一点但精度高得多，对精密操作（拧螺丝、插线）来说是值得的 trade-off。

#### 3. 异步流水线可以隐藏推理延迟

实际部署时是**流水线并行**的——VLA 推理和机器人执行可以重叠：

```
时间轴:
  VLA推理: [██推理300ms██] [██推理300ms██] [██推理300ms██]
  执行:         [─10步200ms─] [─10步200ms─] [─10步200ms─]
                    ↑
              推理完成后，下一轮动作已就绪，无需等待
```

只要满足 **10 步执行时间 ≥ VLA 推理时间**，推理延迟就能被完全隐藏：

| 控制频率 | 10 步耗时 | 能否隐藏 VLA 推理（300ms）？ |
|---------|----------|---------------------------|
| 50Hz（每步 20ms） | 200ms | ❌ 不能，VLA 比执行慢 |
| 33Hz（每步 30ms） | 300ms | ✅ 刚好匹配 |
| 20Hz（每步 50ms） | 500ms | ✅ 完全隐藏，还有余量 |

#### 4. 论文中的实际硬件配置

RLT 原论文使用 pi0.6（6.3B），在 H100 上 VLA 推理约 **200ms**，机器人的控制频率设计为刚好让 10 步执行时间 ≥ VLA 推理时间，从而实现零额外等待。

### 总结

| | 原版 VLA | VLA + RLT |
|---|---|---|
| VLA 调用间隔 | 每 50 步 | 每 10 步 |
| VLA 频率 | 1x | **5x**（推理总量增加） |
| 推理瓶颈条件 | 执行完 50 步后才等 VLA | 需要 10 步执行 ≥ VLA 推理时间 |
| 控制频率 | 受限于机器人硬件 | 受限于 VLA 推理速度 |
| 解决方案 | — | ① 降低控制频率 ② 异步流水线 ③ 更快的硬件 |

**核心 trade-off**：RLT 用 5 倍的 VLA 推理量换来了每 10 步一次重规划的精度优势。对精密操作来说，跑得快但位置不准，不如慢一点但每步都对准。

---

## Q8: 从 Windows + uv 切换到 Ubuntu + conda 后，代码需要做哪些修改？

**几乎不需要修改核心代码。** 下面列出需要关注的点。

### pyproject.toml — 不需要修改

JAX 版本已经在 `pyproject.toml` 中用 `sys_platform` 做了平台区分：

```toml
# pyproject.toml:18-19
"jax[cuda12]==0.5.3; sys_platform == 'linux'",   # Ubuntu 下装 CUDA 版
"jax[cpu]==0.5.3; sys_platform == 'win32'",      # Windows 下装 CPU 版
```

Ubuntu + conda 下 `pip install -e .` 会自动安装 `jax[cuda12]`，不需要改。

### uv.lock — 不需要修改，甚至可以忽略

`uv.lock` 是 **uv 专用**的锁文件。conda 环境下使用 `pip install`，不涉及 `uv.lock`，可以直接无视。

### 唯一需要改的：测试脚本中的强制 CPU 模式

`examples/test_rlt_inference.py` 中原有一行强制 CPU 的代码，已通过新增 `--cpu` 标志解决：

```python
# 以前（硬编码 CPU，Windows 专用）：
jax.config.update("jax_platform_name", "cpu")

# 现在（通过 --cpu 标志控制）：
if args.cpu:
    jax.config.update("jax_platform_name", "cpu")
```

**Ubuntu + GPU 默认用法**（删除强制 CPU 后 JAX 自动检测 GPU）：

```bash
python examples/test_rlt_inference.py
```

**需要强制 CPU 时**：

```bash
python examples/test_rlt_inference.py --cpu
```

### sys.path — conda 下同样需要

`rlt_online_rl` 是一个**顶层目录但不是一个可安装的包**（它没有 `pyproject.toml`，不在 workspace members 中）。测试脚本中的这段代码在 Ubuntu 上同样必要：

```python
# 确保项目根目录在 sys.path 中（rlt_online_rl 是顶层包，不在 src/ 下）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
```

### 对比总结

| 项目 | Windows + uv | Ubuntu + conda | 需要改？ |
|------|-------------|----------------|---------|
| 安装方式 | `uv sync` | `pip install -e .` | ❌ 不需要 |
| `pyproject.toml` | 使用 | 使用 | ❌ 不需要 |
| `uv.lock` | 使用 | 忽略 | ❌ 可删除 |
| 运行命令 | `uv run python script.py` | `python script.py` | ⚠️ 用法不同，代码不改 |
| 强制 CPU | 需要（Windows JAX 无 CUDA） | 不需要（默认 GPU） | ✅ 已改为 `--cpu` 标志 |
| `sys.path` hack | 需要 | 同样需要 | ❌ 不变 |
| `rlt_online_rl` 包 | 不在安装列表 | 不在安装列表 | ❌ 不变（sys.path hack 兜底） |

### Ubuntu + conda 环境准备 checklist

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

> **一句话**：从 Windows + uv 切换到 Ubuntu + conda，核心代码无需任何修改。只需要注意运行命令从 `uv run python` 变为 `python`，且 JAX 在 Ubuntu 上会自动使用 GPU（无需再加 `jax.config.update("jax_platform_name", "cpu")`）。

---

## Q9: RLT Stage 1 训练的 Encoder 和 Decoder 参数量有多大？

### 整体架构

| 组件 | pi0.5 版本（当前实现） | pi0.6 版本（RLT 论文） |
|------|:-------------------:|:--------------------:|
| VLA 基座 | Gemma 2B + 300M Action Expert = **~2.3B** | Gemma 3 4B + 860M Action Expert = **~4.86B** |
| **RLT 编码器（Encoder）** | **~196M**（width=2048） | **未公开**（width 与 Gemma 3 4B 的 hidden_size=2560 匹配，估计更大） |
| **RLT 解码器（Decoder）** | **~250-280M**（width=2048） | **未公开**（同上，估计 ~300-400M） |
| 冻结情况 | VLA 冻结 ✓ | VLA 冻结 ✓ |
| Stage 1 后处理 | Encoder 冻结复用，Decoder 丢弃 | Encoder 冻结复用，Decoder 丢弃 |

> **注意**：RLT 论文未公开 pi0.6 版本的 Encoder/Decoder 具体架构维度。pi0.6 的 Gemma 3 4B 的 hidden_size=2560，比 pi0.5 的 2048 大 25%，因此 Encoder/Decoder 参数量会成比例放大。上表的 pi0.6 估计值基于此比例推算。

### 编码器细节（`RLTEncoder`）

配置（`rl_token.py` 中的 `RLTEncoderConfig`）：

| 参数 | 值 |
|------|:---:|
| `width` | 2048 |
| `depth` | 4（层数） |
| `num_heads` | 8 |
| `mlp_ratio` | 4.0 |

每层参数量：

| 子层 | 参数量 |
|------|:-----:|
| Attention（QKV + 投影） | 4 × 2048² ≈ 16.8M |
| MLP（Linear → GeLU → Linear, mlp_ratio=4.0） | 2 × 2048 × 8192 ≈ 33.6M |
| LayerNorm | 2 × 2048 ≈ 4K（可忽略） |
| **每层小计** | **~50M** |

额外参数：

| 参数 | 量级 |
|------|:---:|
| 可学习 `[RL]` token | 1 × 2048 = 2K |
| 位置编码（max 4096） | 4096 × 2048 ≈ 8.4M |
| 最终 LayerNorm | 2 × 2048 = 4K |

**编码器总计：~196M 参数**

### 解码器细节（`RLTDecoder`）

配置与编码器相同（`width=2048, depth=4, num_heads=8, mlp_ratio=4.0`），但每层多一个 **cross-attention 子层**（Q、K、V + 投影 ≈ 额外 16.8M/层）。

**解码器总计：~250-280M 参数**

### 各版本对比

| 版本 | 基座 VLA | Embedding Width | 编码器 | 解码器 | Stage 1 总训练 |
|:----:|:--------:|:--------------:|:-----:|:-----:|:-------------:|
| **当前实现（pi0.5）** | Gemma 2B + 300M（~2.3B） | 2048 | ~196M | ~250-280M | ~450-475M |
| **RLT 论文（pi0.6）** | Gemma 3 4B + 860M（~4.86B） | 2560（估计） | 未公开（估计 ~300-350M） | 未公开（估计 ~350-450M） | 未公开（估计 ~650-800M） |

### Stage 1 训练目标

VLA 输出的 `N×2048` token 序列 → **Encoder 压缩为单个 2048 维 RL Token** → **Decoder 重建回 `N×2048`** → MSE 重建损失。

```
VLA token embeddings [B, N, 2048]
        │
        ▼
  ┌─────────────┐
  │  Encoder    │  4 层 Transformer，~196M
  └──────┬──────┘
         ▼
  RL Token [B, 2048]   ← 瓶颈
         │
         ▼
  ┌─────────────┐
  │  Decoder    │  4 层 Cross-Attention Transformer，~250-280M
  └──────┬──────┘
         ▼
  重建的 embeddings [B, N, 2048]
        │
  对比 MSE(重建, 原始)
```

超参数：lr=1e-4, batch_size=32, 30,000 步。训练完成后编码器冻结、解码器丢弃。

---

## Q10: RLT Stage 2 训练的 Actor 和 Critic 参数量有多大？

### 架构概览

Actor 和 Critic 都是轻量级 MLP，基于 TD3 框架。

| 组件 | 输入维度 | 隐藏层维度 | 输出维度 | 隐藏层数 |
|------|:--------:|:----------:|:--------:|:--------:|
| **Actor MLP** | 2136 | **256** | 80 | 2 |
| **Critic QNetwork** | 2136 | **256** | 1 | 2 |
| Critic TwinQ（4 份总参数） | — | — | — | 2 |

输入组合：

```
state_dim = 2048 (RL Token) + 8 (proprio) = 2056
action_chunk_dim = 10 (chunk) × 8 (action_dim) = 80

Actor 输入 = state_dim + action_chunk_dim = 2056 + 80 = 2136
          (RL Token + proprio + VLA 参考动作)

Critic 输入 = state_dim + action_chunk_dim = 2056 + 80 = 2136
          (RL Token + proprio + 待评估的动作)
```

### 参数量明细

**Actor（~638K）：**

| 层 | 参数量 |
|----|:-----:|
| LayerNorm(2136) | 4,272 |
| Linear(2136 → 256) | 547,072 |
| Linear(256 → 256) | 65,792 |
| Linear(256 → 80) | 20,560 |
| **小计** | **~638K** |

**Critic QNetwork × 2（~1.23M 可训练）：**

| 层 | 单份参数量 |
|----|:---------:|
| LayerNorm(2136) | 4,272 |
| Linear(2136 → 256) | 547,072 |
| Linear(256 → 256) | 65,792 |
| Linear(256 → 1) | 257 |
| **单份小计** | **~617K** |
| **两份 online（可训练）** | **~1.23M** |
| **四份总计（含 target 网络）** | **~2.47M** |

### Stage 2 总训练参数

| 组件 | 参数量 | Stage 2 是否训练 |
|------|:-----:|:----------------:|
| VLA pi0.5 | ~2.3B | ❌ 冻结 |
| RLT 编码器 | ~196M | ❌ 冻结（Stage 1 训完后冻结） |
| **Actor** | **~638K** | ✅ **训练** |
| **Critic（TwinQ）** | **~1.23M** | ✅ **训练** |
| **Stage 2 总训练参数** | **~1.87M** | — |

### 对比：各组件量级

```
VLA pi0.5 (2.3B)  ████████████████████████████████████████████████████████
RLT 编码器 (196M)  ████▌
Stage 2 训练参数   ▏
  Actor (638K)     ▏
  Critic (1.23M)   ▏
```

Stage 2 训练的 Actor + Critic 总共不到 **2M 参数**，与被冻结的 VLA（2.3B）和编码器（196M）相比几乎可以忽略不计。这也是 RLT 在线 RL 高效的原因——只需要更新极小的 MLP。

### 关键设计

- **Actor 最后一层零初始化**：输出初始为 0 → Actor 初始时完全复制 VLA 参考动作（residual 设计）
- **Critic 无零初始化**：标准初始化
- **TD3 特性**：TwinQ（取 min 防过估）、目标策略平滑噪声、延迟策略更新（critic 每 2 步更新一次 actor）
- **参考丢弃（ref_dropout=0.5）**：50% 样本的参考动作被置零，防止 Actor 过度依赖 VLA

### 各版本对比

| 版本 | Actor 隐藏层 | Critic 隐藏层 | action_dim | 说明 |
|:----:|:----------:|:-----------:|:---------:|:----|
| **当前 PyTorch 实现** | 256 | 256 | 8 | 已合入主仓库，面向 Franka 机器人 |
| **文档描述的 JAX 版** | 512 | 256 | 32 | 原始设计与 RLT 论文对齐的更通用配置 |
| RLT 论文（pi0.6） | 未公开 | 未公开 | 32 | Gemma 3 4B 的 embedding width=2560，Actor/Critic 输入维度更高 |

> **说明**：RLT 论文以 pi0.6 为基座，Gemma 3 4B 的 hidden_size=2560，因此 state_dim = 2560 + 32（proprio）= 2592，action_chunk_dim = 10 × 32 = 320，Actor 输入维度 = 2592 + 320 = **2912**，比 pi0.5 版本的 **2136** 大约 36%，MLP 参数量会成比例放大。此外论文版本可能使用不同的隐藏层宽度，具体数值未公开发布。
