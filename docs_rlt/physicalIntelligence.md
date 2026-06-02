# Physical Intelligence (Pi) 公司调研

---

## 一、公司概况

| 项目 | 内容 |
|------|------|
| **公司名称** | Physical Intelligence（简称 Pi 或 π） |
| **成立时间** | 2024年3月（美国特拉华州注册） |
| **总部** | 美国加利福尼亚州旧金山 |
| **员工人数** | 约 80-100 人（截至2025年底） |
| **定位** | 构建通用机器人"大脑"基础模型，不做硬件，纯软件/AI |
| **行业描述** | 被誉为"机器人领域的 OpenAI"、"机器人的 ChatGPT" |

### 核心使命

构建通用的机器人基础模型（Foundation Model），使任意机器人硬件能像人一样理解物理世界并执行复杂操作任务。公司**不生产机器人硬件**，只提供 AI 大脑软件，采用订阅制收费。

---

## 二、联合创始人（7人）

Physical Intelligence 拥有堪称"全明星"的创始团队，汇聚了来自 **Stanford、UC Berkeley、Google DeepMind、Stripe、Tesla** 等顶级机构的顶尖人才。

### 1. Karol Hausman — CEO & 联合创始人

- **前职**：Google DeepMind / Google Brain 高级研究科学家、机器人操控负责人
- **学术**：斯坦福大学兼职教授
- **博士**：南加州大学（USC）计算机科学博士
- **荣誉**：2023年 IEEE 机器人与自动化学会行业职业奖
- **研究方向**：通用机器人学习，最小监督下的真实世界机器人自主技能获取

### 2. Sergey Levine — 首席科学家 & 联合创始人

- **学术**：UC Berkeley 电气工程与计算机科学系副教授
- **实验室**：领导 **RAIL Lab**（机器人 AI 与学习实验室）
- **地位**：**深度强化学习领域全球领军人物**，谷歌学术引用超 **13万次**
- **影响力**：NeurIPS 2019/2020 论文接收量排名第一，ICML 2022 作者 Top1
- **课程**：伯克利 CS285（深度强化学习）为最热门课程之一
- **代表项目**：领导谷歌 **RT-X 机器人项目**
- **创业愿景**：为机器人领域带来类似"LLM 之于 NLP"的通用解决方案

### 3. Chelsea Finn — 联合创始人

- **学术**：斯坦福大学计算机科学与电气工程系 **助理教授**
- **教育**：MIT 本科 → UC Berkeley 博士（导师包括 Sergey Levine 和 Pieter Abbeel）
- **博士论文奖**：**2018年 ACM 博士论文奖**
- **前职**：Google Brain 研究科学家（5年）
- **代表作**：提出 **MAML（Model-Agnostic Meta-Learning）**，被广泛引用
- **影响力**：谷歌学术引用超 **4.9万次**
- **知名项目**：火爆网络的 **ALOHA 家务机器人** 项目导师
- **备注**：Sergey Levine 的学生出身，现与导师共同创业

### 4. Brian Ichter — VP 工程 & 联合创始人

- **前职**：Google Brain / Google DeepMind 研究科学家
- **博士**：斯坦福大学航空航天工程博士
- **专长**：机器人运动规划、基础模型

### 5. Lachy Groom — 联合创始人（商业/战略）

- **前职**：Stripe 高管（Head of Stripe Issuing）
- **背景**：知名科技天使投资人

### 6. Adnan Esmail — 联合创始人

- **前职**：Anduril Industries 高级副总裁兼电气工程负责人
- **背景**：前特斯拉自动驾驶/硬件专家（参与设计 Model X 鹰翼门）

### 7. Quan Vuong — 联合创始人

- **前职**：Google DeepMind 软件工程师
- **背景**：UC San Diego，专注跨具身学习（cross-embodiment learning）

---

## 三、融资历程

| 轮次 | 时间 | 金额 | 估值 | 主要投资者 |
|:----:|:----:|:----:|:----:|:----------|
| **种子轮** | 2024年3月 | **$7000万**（约5亿人民币） | ~$4亿 | Thrive Capital, OpenAI, Lux Capital, 红杉资本, Khosla Ventures |
| **A轮** | 2024年11月 | **$4亿**（约29亿人民币） | **$24亿** | Jeff Bezos（Bezos Expeditions）, OpenAI, Thrive Capital |
| **B轮** | 2025年11月 | **$6亿**（约42亿人民币） | **$56亿** | CapitalG（Alphabet/Google）, Lux Capital, Thrive Capital, Bezos, Index Ventures, T. Rowe Price, NVIDIA |
| **C轮**（传闻中） | 2026年 | **~$10亿**（约72亿人民币） | **$110亿+** | Founders Fund, Lightspeed（洽谈中） |

**累计融资**：超过 **$20亿**（约140亿人民币），成立仅约2年。

**特殊之处**：公司截至2026年初**没有商业化产品和收入**，但投资者基于技术潜力持续下注。

---

## 四、投资者阵容

Pi 的投资人名单横跨多个竞争性巨头，极具标志性：

- **Jeff Bezos**（Amazon 创始人）
- **OpenAI**（AI 领域领导者）
- **CapitalG**（Alphabet/Google 旗下投资机构）
- **NVIDIA**（NVentures）
- **Thrive Capital**
- **Lux Capital**
- **红杉资本**（Sequoia Capital）
- **Index Ventures**
- **T. Rowe Price**
- **Khosla Ventures**
- **Founders Fund**（传闻领投2026年C轮）

---

## 五、技术路线与模型家族

| 模型 | 时间 | 关键突破 |
|:----:|:----:|:---------|
| **π0** | 2024年10月 | 首个 VLA 基础模型，控制 8+ 种机器人、50+ 任务（叠衣服、做咖啡、组装盒子） |
| **π0-FAST** | 2025年1月 | 训练速度提升 5 倍 |
| **π0（开源）** | 2025年2月 | 开源模型权重 |
| **π0.5** | 2025年4月 | 在陌生环境中自主移动 |
| **π₀.6** | 2025年11月 | **RECAP** 强化学习算法；吞吐量提升 2 倍；任务成功率 >90% |
| **MEM** | 2026年3月 | 多尺度记忆系统，可保持 15 分钟任务上下文 |
| **RL Token (RLT)** | 2026年3月 | 亚毫米级精度，15 分钟学会高精度任务，速度提升 3 倍 |

**架构演进**：π₀.6 → SigLIP 408M（视觉编码器）+ Gemma 4B（语言模型）+ Action Expert 860M（动作解码器）

**预计 π1.0**：2026年发布完整版，商业化可能在 2026-2028 年之间。

---

## 六、商业模式

- **不造机器人**：只卖大脑（AI 软件/模型）
- **收费模式**：订阅制，约 **$300/月/机器人**
- **兼容性**：支持 **7+ 种机器人硬件平台**
- **数据需求**：仅需 **1-20 小时客户数据** 即可为新任务微调

### 早期客户案例

| 客户 | 场景 | 效果 |
|:----:|:----:|:----:|
| **Weave** | 叠衣机器人 | 折叠错误减少 42%，人工干预减少 50% |
| **Ultra** | 仓库包装 | 150+ 件/小时，96.4% 自主运行率 |

---

## 七、竞争格局

Pi 采取 **软件优先、硬件无关** 的策略，与以下公司形成差异化竞争：

| 类型 | 公司 | 策略 |
|:----:|:----:|:------|
| **纯软件大脑** | **Physical Intelligence** | 不造硬件，只提供通用 AI 模型 |
| **软硬一体** | Tesla（Optimus） | 全栈自研机器人 |
| **软硬一体** | Figure AI | 人形机器人+AI |
| **纯软件大脑** | **Skild AI** | 类似路线，直接竞争对手 |
| **纯软件大脑** | **Covariant** | 专注仓库抓取 AI |

---

## 八、公司特点总结

1. **学术底蕴极深**：三位核心创始人（Levine、Finn、Hausman）均为机器人/强化学习领域的世界级学者
2. **融资能力惊人**：两年内从小透明飙升至百亿估值，背后集结了 Bezos、OpenAI、Google、NVIDIA 等互相竞争的巨头
3. **技术路线清晰**：VLA 基础模型 → 在线 RL 微调 → 多尺度记忆，逐步攻克"泛化→精度→持续学习"
4. **商业化谨慎**：明确不给投资者商业化时间表，专注研究而非营收
5. **与你的项目高度相关**：手机盒包装任务可直接使用 π₀.6 + RLT 技术栈

---

## 九、参考来源

- [Physical Intelligence 官网](https://www.pi.website)
- [RLT 研究页面](https://www.pi.website/research/rlt)
- [36氪报道：谷歌斯坦福大神给机器人造大脑](https://36kr.com/p/3004539084518275)
- [虎嗅报道：OpenAI 红杉抢着投5亿](https://pro.huxiu.com/article/2892926.html)
- [机器人大讲堂：PI 获42亿融资](https://www.leaderobot.com/news/6729)
- [SiliconANGLE：Bezos-backed Pi raises $600M](https://siliconangle.com/2025/11/20/jeff-bezos-backed-physical-intelligence-raises-600m-improve-ai-robot-brains/)
- [TFN：Pi eyes $1B raise at $11B valuation](https://techfundingnews.com/physical-intelligence-1b-raise-11b-valuation-founders-fund-lightspeed/)
- [腾讯云：一口气融资40亿，最强机器人大脑诞生](https://news.qq.com/rain/a/20251201A05SS100)
