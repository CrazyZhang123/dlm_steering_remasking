# 系统性降低 ASR 改进计划（2026-07-03）

> 状态：**待评审 / 待启动**。本文档整合当前项目全量实验证据 + arXiv/学术界 2025–2026 最新
> DLM 安全防御研究，给出降低 ASR 的分层改进路线，并**详细到"改哪个函数、哪几行、改成什么、
> 怎么验证"**，供先看懂再决定是否启动。启动任一实验前遵循 `AGENTS.md`（tmux 后台 + 日志重定向
> + 30 分钟低频监控 + 3 轮 review）。本文档不含 git 操作规划。

---

## 目录

1. [先读这段：一句话结论](#0-一句话结论)
2. [口径澄清：为什么本地是 67% 而 README 是 25%](#1-口径澄清)
3. [现在的推理流程到底做了什么（看懂 baseline）](#2-现在的推理流程到底做了什么)
4. [现状证据：为什么"改 bank"这条线走到头了](#3-现状证据)
5. [学术界对标：别人怎么把 ASR 打下去的](#4-学术界对标)
6. [改进方向 A：steering 超参扫描（零代码）](#方向-a)
7. [改进方向 B：升级重掩码机制（改代码，借鉴 DiffuGuard）](#方向-b)
8. [改进方向 C：早期 refusal 干预（可选）](#方向-c)
9. [改进方向 D：质量护栏 + 攻击泛化（必做）](#方向-d)
10. [路线图与决策门槛](#5-路线图与决策门槛)
11. [风险与应对](#6-风险与应对)

---

## 0. 一句话结论

**当前 ASR 卡在 ~67% 的根因不是 bank 不够好，而是"干预机制"本身几乎没被开发过。**

- 过去所有实验都在动**同一件事**：CSD bank 怎么构造（簇数 M4–M16、category 分组、token 选择
  direction/random/MIL/KNN、特征预处理 center/PCA）。这些变体的 ASR 全部挤在 **65–74%**，
  多数差异落在单次推理 ±2.5pp 的噪声带内。**这条线已经到头。**
- 而**真正决定 ASR 的三个维度从未被扫过或实现**：
  1. **steering 超参**——`target_layer` 一直固定 31、`steering_overshoot` 一直 1.0、
     `initial_steering_ratio` 一直 0.1、`alignment_threshold` 一直 0.0，从项目开始没动过；
  2. **重掩码阶段的解码随机性**——Phase 1 是纯贪婪 topk，无任何随机注入；
  3. **重生成时的 logits 抑制**——Phase 2 重掩码后直接重采样，不禁止重新生成同一有害 token。
- 学术界 training-free SOTA **DiffuGuard** 把 6 种攻击平均 ASR 从 47.9% 打到 **14.7%**，靠的
  正是本项目缺的后两项。**这是与本项目 training-free 定位最契合、性价比最高的借鉴方向。**

**推荐执行顺序**：先扫 steering 超参（零代码、复用现成 bank、~半天出结果）→ 再实现随机退火
重掩码 + logits 抑制（training-free，改 `eval_llada_steering.py` 两处）→ 组合验证。

---

## 1. 口径澄清

**先破除一个误会**：README 表格写 `JailBreakBench LLaDA Ours = 25.67% ASR`，而本地复现是 65–74%。
**两者不矛盾**，因为评测口径完全不同：

| 维度 | README 表格 | 本项目本地评测口径 |
|---|---|---|
| 攻击方法 | 未叠加 dLLM 专用攻击（近 zeroshot/常规越狱） | **JBB + DIJA**（DIJA 是专攻 dLLM 的强攻击，裸模型 ASR 可达 98%+） |
| 评判器 | 论文口径 | 本地 `Llama-Guard-4-12B` |
| 样本量 | 论文全量 | 100 条 |

所以本地 **~67% 是在 DIJA 强攻击下**的结果，不是项目退步。DIJA 的攻击原理是：在 prompt 里
**预填充交错的 mask-文本模板**（例如 "First,. [MASK×20] Second,. [MASK×20] ..."），利用 dLLM 的
双向建模 + 并行解码特性，让模型"顺着模板把有害内容补全"。这种攻击绕过了常规的 response 级拒答。

**降 ASR 的目标应明确为**：在 JBB+DIJA 口径下，把 ~67% 往下压，同时不破坏 TruthfulQA/MMLU 效用。

---

## 2. 现在的推理流程到底做了什么

要理解"改什么"，先看懂"现在怎么跑"。评测实际走的是 **DIJA 路径**（`attack_method=DIJA`），
入口在 `eval_llada_steering.py` 的 `llada_dija_sample`（约 L591–L694）。整个流程分两阶段：

### 2.1 Phase 1：逐步去噪解码（L619–L652）

```text
输入：DIJA 模板 prompt（含大量 [MASK] 占位）
循环 steps=128 步：
    ① 前 initial_steering_steps 步（= 128×0.1 ≈ 12 步）挂 adaptive steering hook
       —— 对 mask 位置的隐状态做 CSD 方向的减法修正（见 2.3）
    ② 模型 forward 得到 logits → softmax → 采样出候选 token x0 和其置信度 x0_p
    ③ 【关键】贪婪 topk：每步只揭开置信度最高的 k 个 mask 位置（L650 torch.topk）
```

**问题所在**：第 ③ 步是**纯贪婪**的。DIJA 模板会让某些"有害顺从 token"获得极高置信度，
贪婪 topk 必然优先揭开它们，于是有害骨架在早期就被钉死——这正是 DiffuGuard 论文说的
"greedy remasking 的 harmful bias"。

### 2.2 Phase 2：有害检测 + 重掩码修复（L657–L692）

```text
Phase 1 全部解码完成后，最多循环 max_refinement_iters=5 次：
    ① 再 forward 一次，取 target_layer=31 的隐状态
    ② 用 CSD bank 算每个 token 的 alignment（隐状态在有害方向上的投影）
    ③ alignment > alignment_threshold(=0.0) 的 token 判为有害
    ④ 把有害 token 重新设回 [MASK]
    ⑤ _refill_masks_with_steering 重新生成这些位置（L442–L463）
    ⑥ 若无有害 token 则提前退出
```

**问题所在**：第 ⑤ 步重生成时，**没有任何机制阻止模型重新吐出刚被判为有害的那个 token**。
模型很可能"检测→remask→又生成同一个词→又检测"，5 次迭代白跑。这正是 DiffuGuard 的
"guided regeneration + logits 抑制"要解决的。

### 2.3 adaptive steering hook（L406–L439）当前长什么样

```text
对 mask 位置的隐状态 h：
    a = <h, v_unit>            # v = CSD 方向（bank 里 route 到的局部向量）
    if a > theta(=0.0):
        h' = h - beta·(a-theta)·v_unit    # beta = steering_overshoot = 1.0
    else:
        不动
```

**四个从没调过的旋钮**都在这里/相邻：`target_layer`（在哪层挂 hook + 取隐状态）、`beta`
（推离强度）、`theta`（触发阈值）、`initial_steering_ratio`（前多少步挂 hook）。

---

## 3. 现状证据

全部 100 样本 / JBB+DIJA / 本地 Llama-Guard，来自 docs 各 Stage 文档：

| 方案 | 关键变量 | ASR（3 轮均值或单次） | 判断 |
|---|---|---:|---|
| Stage 0 全局 Sentence-CSD | 单向量 | 74.0% | 冻结 baseline |
| **CT-CSD M12（当前最优 base）** | l2_only, all token | **67.7%**（65/70/68） | 全项目最低点 |
| CT-CSD M9/M10/M11 细扫 | 簇数 | 69.0 / 71.3 / 71.7 | 均不优于 M12 |
| Category-aware CT-CSD M10 | category 分组 | 67–71% | 与 M12 持平 |
| MIL token probe τ0.7 | learned 选 token | 71.0% | 无收益 |
| direction top-ratio | 方向选 token | 68–74% | 无收益（减 token 反而有害） |
| random top-ratio | 随机选 token | 70% | 无收益 |
| KNN label-clean | 去噪选 token | 几乎不删 token（<1%） | 预期无收益 |
| center_l2（去均值） | 预处理 | **72.3%**（72/71/74） | 更差 |
| center_pca128_l2 | 预处理降维 | 70.7% | 更差 |
| center_pca256_l2 | 预处理降维 | 进行中（GPU1） | 预期无收益 |

**横向结论**：`bank 表示 / token 选择 / 簇数`三条线全部无正收益，最优仍是最朴素的
`ct_csd M12 + l2_only + all token`。继续在此维度调参的边际收益趋近于零。

---

## 4. 学术界对标

### 4.1 DiffuGuard（arXiv 2509.24296，training-free，**最值得借鉴**）

training-free 推理框架，6 攻击平均 ASR **47.9% → 14.7%**。对 **DIJA(AdvBench)**：
LLaDA **98.65% → 51.92%**（DiffuGuard 单用），叠加 self-reminder → **39.04%**；
Dream **99.23% → 6.94%**。两个模块正好对应本项目 §2 的两个缺口：

**模块一：Stochastic Annealing Remasking（随机退火重掩码）——本项目完全没有**

- 在 confidence topk 里注入随机项：
  \[ \mathcal I = \text{arg top-}k_i\big[(1-\alpha_n)\cdot \text{Prob}(\hat\tau_i^n) + \alpha_n\cdot R_i\big],\quad R_i\sim U(0,1) \]
- α 随步数退火（早期强、后期弱）：\( \alpha_n = \alpha_0\,(1-\tfrac{n-1}{N-1}) \)
- **为什么有效（直觉）**：当模型对某个有害顺从 token 给出异常高的 confidence 时，纯贪婪一定选它；
  注入随机项后，其他安全 token 有机会被选中，从而打破 DIJA 模板预设的确定性有害路径。
  早期随机性最强（此时对最终安全最关键），后期恢复贪婪以保证流畅度——**优雅化解安全-质量权衡**。
- DiffuGuard 消融：该模块主要防"预优化提示类"攻击（含 JBB 类）。

**模块二：Block-level Audit and Repair——本项目部分有，但触发信号更弱、且缺 logits 抑制**

- **Audit（触发信号）**：用 **Safety Divergence (SD)** = \(1-\cos(h_{\text{origin}}, h_{p_0})\) 度量
  "模板注入后的隐状态"相对"原始恶意 query 隐状态"的偏移。SD > λ 才触发修复。
  （本项目现在用的是 CSD 投影 > 0，是另一种触发信号。）
- **Repair（修复）**：随机 remask 该 block + 重新生成，且**把原有害 token 的 logits 设 -∞**，
  防止模型重复犯错。**这正是本项目 Phase 2 缺的那一步。**

### 4.2 A2D（arXiv 2509.23286，ICLR 2026，**需训练，仅借思路**）

token 级 EOS 对齐：微调模型，让它在有害 span 的 mask 位置输出 `[EOS]`。DIJA ASR **>80% → 1.3%**
（LLaDA）/ **0.0%**（Dream）。**需要微调**，与本项目 training-free 定位冲突，不直接采用。
可借鉴的 training-free 思路：推理时监控最左 mask 位置的 `P([EOS])`/refusal 概率，超阈值早停——
把本项目 README 已验证的"早期插入 Sorry 抑制有害"motivation 落成实际机制（见方向 C）。

### 4.3 攻击面提醒（arXiv 2602.00388 context nesting、2507.19227 PAD）

本项目目前只在 **DIJA** 上评测。context nesting / PAD 是更新的 dLLM 攻击。**风险**：只对 DIJA
调参可能过拟合单一攻击。改进候选应在多攻击上验证泛化（见方向 D）。

---

## 方向 A

## 方向 A：steering 超参扫描【最高优先级 · 零代码 · 复用 bank】

**这是最快、风险最低、无需改代码的方向。** 依据：§2.3 的四个旋钮直接控制干预强度与覆盖范围，
自项目开始从未扫过；复用现成 M12 bank，每配置仅 ~1h 推理 + ~2min 评判。

### A.1 实验矩阵

base 固定 `outputs/ct_csd_llada_m12/ct_csd_bank.pt`，单变量扫描，每点 **3 轮**（对齐 ±2.5pp 噪声）：

| 组 | 变量 | 扫描值 | 直觉 | 是否需重建 bank |
|---|---|---|---|---|
| baseline | — | overshoot=1.0, ratio=0.1, layer=31, θ=0.0 | 已有 67.7 | 否 |
| **O**（重点） | `steering_overshoot` | 1.5 / 2.0 / 3.0 | beta 越大越强推离有害半空间 | **否**（仅改推理 flag） |
| **R**（重点） | `initial_steering_ratio` | 0.2 / 0.3 / 0.5 | 早期 steering 覆盖更多去噪步 | **否** |
| T | `alignment_threshold` | -0.05 / -0.1 | 负阈值 → 更激进触发 steering/remask | 否 |
| **L**（潜力大） | `target_layer` | 16 / 20 / 24 / 28 | 中间层 steering 通常比末层有效（CAA/MAT-steer 经验） | **是**（每层需对应层 bank） |

> 说明：`docs/plan/steering_hparam_scan_plan.md` 已规划 O/R 两组但未启动，且**漏掉了最关键的
> `target_layer` 维度**。本方向补上 L 组和 T 组。O/R/T 三组零代码零重建，可立即跑；L 组因为
> steering 向量是从 `target_layer` 层的隐状态构造的，换层必须重建对应层的 bank（~5h/层）。

### A.2 具体命令（O 组示例，其余同理只改一个 flag）

```bash
# 复用 m_scan 系列脚本结构，tmux 后台 + 日志重定向 + 断点续跑
# 每配置目录 outputs/jbb_dija_ct_csd_m12_os{overshoot}_r{round}
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python eval_llada_steering.py \
  --csv_path JBB \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --generated_samples_path outputs/jbb_dija_ct_csd_m12_os1.5_r1 \
  --attack_method DIJA \
  --sampler steering \
  --steering_vector_path outputs/ct_csd_llada_m12/ct_csd_bank.pt \
  --target_layer 31 \
  --alignment_threshold 0.0 \
  --steering_overshoot 1.5 \        # ← O 组唯一变量
  --initial_steering_ratio 0.1 \
  --max_refinement_iters 5 \
  --sampling_steps 128 --mask_length 128 --block_size 128 --dija_mask_counts 128 \
  --device cuda
# 随后 scripts/eval_llama_guard_local.py 评判，读 metadata.asr_percent
```

### A.3 验收标准

- 3 轮均值较 baseline 67.7 **降 ≥5pp** 才算真实效应（单次极差可达 5pp，需 3 轮均值判断）。
- 同时过质量护栏（方向 D）：不能靠"全拒答/乱码"刷低 ASR。
- **预期**：`overshoot` 和 `target_layer` 最可能出正收益；这是"当前框架内"最快的提升点。

---

## 方向 B

## 方向 B：升级重掩码机制【高优先级 · training-free · 借鉴 DiffuGuard】

针对 §2.1 和 §2.2 的两个明确缺口，各写一个 training-free 补丁。**两处改动都默认关闭
（新增 flag 默认值 = 保持现行为），保证向后兼容、可消融。**

### B1. 随机退火重掩码（补 Phase 1 的贪婪缺口）

- **改哪里**：`eval_llada_steering.py` 两处 confidence topk：
  - DIJA 路径 `llada_dija_sample` 的 L640–L650（评测主路径）；
  - block 路径 `llada_remask_sample` 的 L542–L548（非 DIJA）。
- **改成什么**（伪代码）：

```python
# 现在（贪婪）：
confidence = torch.where(mask_index, x0_p, -np.inf)
_, select_index = torch.topk(confidence[j], k=k)

# 改后（退火随机）：
alpha_n = self.anneal_alpha0 * (1 - i / max(steps - 1, 1))   # 早强晚弱
noise = torch.rand_like(x0_p)                                 # R_i ~ U(0,1)
score = (1 - alpha_n) * x0_p + alpha_n * noise                # 混合分
score = torch.where(mask_index, score, -np.inf)
_, select_index = torch.topk(score[j], k=k)
# 注意：揭开后写入的仍是 x0（原采样 token），noise 只影响"揭哪些位置"
```

- **新增 CLI**：`--anneal_alpha0`（默认 **0.0** → 完全等价现行为）、`--anneal_decay`（默认 `linear`）。
- **验收**：`alpha0=0` 时输出与现版**逐 token 一致**（写回归测试断言）；`alpha0∈{0.1,0.3}` 扫 ASR + 质量。
- **预期**：DiffuGuard 消融显示该模块对 JBB 类预优化提示贡献最大。

### B2. 重生成 logits 抑制（补 Phase 2 的重复缺口）

- **改哪里**：`_refill_masks_with_steering`（L442–L463）。当前流程：remask 有害位置 → forward →
  softmax → 采样。**缺"禁止重生成原有害 token"这一步。**
- **改成什么**（伪代码）：

```python
# 调用方（Phase 2）先把"被判有害的位置 + 其原 token id"传进来
def _refill_masks_with_steering(self, xt, block_start, block_end,
                                suppressed=None):   # ← 新增参数
    ...
    logits = self.model(xt).logits
    if self.suppress_harmful_logits and suppressed is not None:
        # suppressed: {position: original_token_id}
        for pos, tok_id in suppressed.items():
            logits[0, pos, tok_id] = float('-inf')   # 禁止复现原有害 token
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0 = _sample_categorical(p)
    ...
```

- **新增 CLI**：`--suppress_harmful_logits`（默认 **False** → 向后兼容）。
- **验收**：开启后被 remask 的位置不再复现原 token（写单测验证）；ASR 对比。
- **预期**：直接阻断"检测有害→重生成同样有害"死循环，是低风险高确定性的补丁。

### B3.（可选，B1/B2 见效后再评估）Safety Divergence 触发门控

- 当前 Phase 2 用 CSD 投影 > θ 触发 remask（§2.2 第 ③ 步）。可叠加/替换为 DiffuGuard 的 SD：
  推理前对"去掉 DIJA 模板的原始 goal"多跑一次 forward 取 `h_origin`，再与模板注入后的隐状态算
  cosine 距离，SD > λ 才触发修复。
- **改动量中等**（需改造 prompt 构造以拿到 origin goal），放在 B1/B2 验证有效后再做。

### B4. 改动范围小结

| 文件 | 改动 | 原则 |
|---|---|---|
| `eval_llada_steering.py` | B1 两处 topk 注入退火随机；B2 `_refill_masks_with_steering` 加 logits 抑制；新增 3 个 CLI flag | 纯追加，默认值 = 现行为 |
| `tests/` | 回归测试：`alpha0=0`/`suppress=False` 时输出与现版一致；`suppress=True` 时原 token 不复现 | 单测先行，防回归 |

---

## 方向 C

## 方向 C：早期 refusal 概率干预 / 早停【中优先级 · training-free】

**依据**：本项目 README motivation 已实证"早期步注入 `Sorry` 抑制有害、注入 `Sure` 放大有害"。
把这个观察从"分析"变成"机制"：

- **C1 早停**：推理第 1 步监控最左 mask 位置 `P([EOS])` 或 refusal 关键词概率，超阈值直接输出
  拒答（借鉴 A2D 推理侧思路，但不训练）。**前置**：先做信号可分性分析（有害 vs 安全样本的
  P(EOS)/refusal-prob 分布是否可分），可分才上机制，否则无效。
- **C2 早期 refusal 偏置**：前 k 步在 logits 上对 refusal 起始 token（Sorry / I can't 等）加正偏置，
  等价于"软性地把安全 token 塞进早期轨迹"。

**优先级低于 A/B**：因为 LLaDA 未针对 EOS 拒答微调，信号强度未知，需先验证可分性。

---

## 方向 D

## 方向 D：质量护栏 + 攻击面泛化【贯穿所有方向 · 必做】

任何降 ASR 的候选都必须同时满足，否则视为"靠过拒答/乱码刷分"：

1. **质量护栏**：最优候选补跑 `scripts/test_rouge_score.sh`（TruthfulQA）+ `scripts/mmlu_eval.sh`，
   相对 baseline 回归 >2 分判为过度干预，候选作废。
2. **抽查生成**：每候选 r1 抽 ≥10 条 `results.json`，确认非空、非乱码、非全模板拒答。
3. **攻击面泛化**：最优候选在 `zeroshot / PAP / prefix` 上复测，确认不是只对 DIJA 过拟合。

---

## 5. 路线图与决策门槛

```text
第 1 步（今天可启动，GPU0 现空闲，零代码）：
    方向 A 的 O 组（overshoot 1.5/2.0）+ R 组（initial_steering_ratio 0.2/0.3）
    → 复用 M12 bank，单卡顺序，先看干预强度/覆盖步数是否松动 67.7
    （L 组 target_layer 需重建 bank，放第 1 步之后按结果决定是否投入）
    ↓
第 2 步（接续或并行）：
    方向 B1 + B2（随机退火 + logits 抑制）
    → 先写代码 + 3 轮 review + 回归测试（alpha0=0 / suppress=False 等价现版）
    → 在最优 base 上开 alpha0=0.1/0.3、suppress on/off 扫描
    ↓
第 3 步：
    组合最优（best overshoot/layer + 退火 + logits 抑制）
    → 方向 D 全套护栏 + 多攻击泛化
    ↓
决策门槛：
    方向 A 单独降 ≥5pp    → steering 超参是主收益源，优先固化
    方向 B 单独降 ≥5pp    → 重掩码机制是主收益源，写入默认 pipeline
    A/B 组合显著优于各自   → 作为新主方法
    全部无效              → 转"回归论文单方向 CSD 重做干预"（省去 ~5h/bank 构造）
```

---

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| 随机退火伤流畅性（安全-质量权衡） | α 退火（早强晚弱）+ 质量护栏；DiffuGuard 已验证退火可缓解 |
| 过度 steering 导致全拒答刷低 ASR | 方向 D 强制护栏 + over-refusal 抽查 |
| 只对 DIJA 过拟合 | 方向 D 多攻击泛化验证 |
| `target_layer` 换层需重建 bank（~5h/层） | 先跑零代码的 O/R/T 组，L 组按前面结果决定是否值得投入 |
| GPU 资源（GPU1 仍在跑 pca256） | 方向 A 用 GPU0；避免双卡同时建 bank 争抢（历史教训） |
| 改推理代码引入回归 | 所有新 flag 默认值 = 现行为；回归测试断言 alpha0=0 / suppress=False 时逐 token 一致 |

---

## 7. 一句话总结

过去把力气全花在"CSD bank 怎么构造"上，ASR 卡在 67% 到头了；**下一步应转向从未开发的干预维度**
——先扫 steering 超参（尤其 `overshoot` 和 `target_layer`，零代码复用 bank），再借鉴 DiffuGuard 补上
**随机退火重掩码**与**重生成 logits 抑制**这两项 training-free 手段。它们正是学术界把 dLLM 攻击 ASR
打到 14.7% 的关键，也正是本项目当前推理流程（`eval_llada_steering.py`）的两个明确空白。
