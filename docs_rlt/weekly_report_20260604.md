# VLA + RLT 研究周报

**日期**: 2026/05/29 - 2026/06/04

---

## 一、本周完成工作

### 1. RLT Stage 1 训练跑通

完成了 RLT（RL Token）第一阶段训练的代码执行与验证：

- **Stage 1 训练流程**：加载预训练的 pi0.5 VLA 模型（冻结全部参数）→ 初始化 RLTEncoder + RLTDecoder → 在 demo 数据上以 MSE 重建损失训练 Encoder-Decoder → 训练完成后冻结 Encoder、丢弃 Decoder
- 训练脚本位于 `scripts/train_rlt_token.py`，默认配置：Encoder 4 层 Transformer（width=2048, 8 heads），30,000 训练步，batch_size=32，lr=1e-4
- 所有 RLT 新增模块均为 **纯 JAX / Flax NNX** 实现，与 openpi 代码库风格一致

### 2. RLT 代码实现完成

基于 openpi 代码库完成了 RLT 整套代码的实现，涵盖两阶段训练+推理部署：

| 模块 | 文件 | 说明 |
|------|------|------|
| RL Token Encoder-Decoder | `src/openpi/models/rl_token.py` | 4 层 Transformer 编码器压缩 N×2048→2048；解码器跨注意力重建 |
| RLT 策略包装器 | `src/openpi/policies/rlt_policy.py` | VLA→Encoder→Actor 推理流水线 |
| Actor 网络 | `rlt_online_rl/actor.py` | 2 层 MLP（512 hidden），输入 state+ref_actions，输出修正动作 |
| Critic 网络 | `rlt_online_rl/critic.py` | 双 Critic TD3，2 层 MLP（256 hidden） |
| TD3+BC Learner | `rlt_online_rl/learner.py` | Critic TD 更新 + 延迟 Actor 更新 + Polyak 软更新 |
| Replay Buffer | `rlt_online_rl/replay.py` | 容量 1M 循环数组，stride=2 存储 |
| Rollout 运行时 | `rlt_online_rl/rollout.py` | 机器人交互循环，奖励收集 |
| Stage 2 训练入口 | `scripts/train_rlt_online.py` | 加载冻结 VLA+Encoder，初始化 Actor-Critic |
| 推理服务 | `scripts/serve_rlt_policy.py` | WebSocket 策略服务部署 |

### 3. 深入调研 RLT 技术文档

系统阅读并整理了 `docs_rlt/` 目录下的核心文档：

#### 文档阅读清单

| 文档 | 核心内容 |
|------|---------|
| `rlt_implementation_guide.md` | RLT 完整实现指南，涵盖架构设计、Stage 1 信息论分析、AC 网络训练详解 |
| `rlt_qa.md` | 10 个核心 Q&A：RLT 原理、BC 正则化、奖励函数设计、人工干预需求、参数量、执行时序等 |
| `20260603_rl_token_train_flow_analysis.md` | Stage 1 训练调用链深度分析（6 个环节逐层拆解） |
| `code_revision_v1.md` | 代码变更记录，10 个新增文件 + 3 个修改文件 |
| `codeAnalysis.md` | 项目代码结构与功能分析，两条 pipeline 对比 |
| `rlt_comprehensive_report.md` | 综合报告，合并所有文档 |
| `projectReview.md` | 项目综述，含 PI0.5 vs RLT 架构对比、完整代码映射表 |
| `physicalIntelligence.md` | Physical Intelligence 公司及 pi0 系列背景调研 |

#### 关键技术要点掌握

- **RLT 核心思想**：冻结 ~2.3B（pi0.5）/ ~4.86B（pi0.6）VLA，外挂 2-3 层 MLP 的 Actor-Critic 做在线 RL 微调
- **RL Token 信息论保证**：Encoder 通过全注意力做自适应信息提炼，Decoder 重建损失保证信息不丢失（MSE 收敛到 ~0.01）
- **BC 正则化机制**：`L = -Q + β·MSE(a, ref_actions)`，β=2.0 默认，防止 Actor 偏离 VLA 太远
- **TD3 三重保障**：Twin Critics（防 Q 值高估）+ Target Networks（防振荡）+ Target Smoothing（防过拟合）
- **Stage 1 参数量**：Encoder ~196M + Decoder ~250-280M = ~450-475M（pi0.5 版本）
- **Stage 2 训练参数量**：Actor ~638K + Critic ~1.23M = **仅 ~1.87M 参数**在线训练
- **压缩比**：RL Token 将 N×2048（~2M 值）压缩为 2048 维，压缩比 ~500:1，但 MSE 仍能收敛到 0.01

#### 阅读的论文资料

| 论文 | 作者 | 来源 |
|------|------|------|
| *RL Token: Bootstrapping Online RL with Vision-Language-Action Models* | Xu et al., Physical Intelligence | arxiv 2604.23073 |
| *pi0.5: a Vision-Language-Action Model with Open-World Generalization* | Physical Intelligence | openpi 配套论文 |
| *pi0.6: a VLA That Learns from Experience* | Physical Intelligence | arxiv 2511.14759 |
| *VLAs with Long and Short-Term Memory* | — | 相关 VLA 研究 |
| *Emergence of Human to Robot Transfer in VLA* | — | 人-机器人迁移研究 |

---

## 二、遇到的问题与解决

| 问题 | 说明 | 状态 |
|------|------|------|
| pi0.5 双 Expert 维度不一致 | prefix_out (2048) 与 suffix_out (1024) 宽度不同，需要投影层统一 | 已解决：在 `pi0.py` 中新增 `_suffix_emb_proj` 投影层 |
| Windows JAX 无 CUDA | Windows 下 JAX 只有 CPU 版本，测试速度受限 | 已解决：新增 `--cpu` 标志，后续迁移到 Ubuntu+GPU 无需修改代码 |
| rlt_online_rl 包不在 workspace | 该目录是顶层包，不在 `pyproject.toml` 的 workspace members 中 | 已处理：通过 `sys.path` hack 方式导入，Ubuntu/Windows 均适用 |
| Stage 1 训练脚本为简化版本 | 当前 `train_rlt_token.py` 使用 fake batch 而非真实 dataloader | 待处理：接入真实数据 pipeline |

---

## 三、下周计划

1. **Stage 1 正式训练**：使用真实 demo 数据（替换 fake batch）进行 Encoder-Decoder 训练，验证 MSE 收敛曲线
2. **Stage 2 准备工作**：
   - 搭建仿真环境（Gymnasium），设计奖励函数
   - 配置 RLT 推理测试脚本，验证 `VLA → Encoder → Actor` 完整数据流 shape 正确性
3. **仿真环境 Stage 2 训练**：在仿真中运行 TD3+BC 在线 RL，观察 Q 值收敛、Actor 动作修正效果
4. **Stage 1 训练结果分析**：验证 RL Token 重建质量（MSE < 0.05?），以及 RL Token 在不同场景下的一致性
5. **可选：环境迁移**：将项目迁移到 Ubuntu+conda+GPU 环境，利用 CUDA JAX 加速训练
