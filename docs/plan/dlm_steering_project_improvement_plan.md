# DLM Steering Remasking 项目改进计划

## 0. 文档目标

本文档用于整理当前项目在 `Stage 0–Stage 6` 范围内的下一步改进计划。重点不是重新设计一个完整新方法，而是给当前 `Category-aware CT-CSD` 流程补充两个可替换模块：

1. **Direction-selected Category-CT-CSD**：替换 token selection（token 选择）部分，解决 harmful response 中不是所有 token 都真正有害的问题。
2. **Feature preprocessing**：替换 clustering / routing（聚类 / 路由）时使用的特征表示，减少 hidden norm、高维噪声和格式 token 对聚类的影响。

本次完善后的文档同时承担三个作用：

```text
统一当前项目状态
明确后续方法主线
给出可执行的工程实施计划和验收门槛
```

整体原则是：

```text
少改主流程
模块可替换
实验可消融
结果可解释
不要把所有技巧堆到一个方法里
```

---

## 1. 当前项目状态总结

### 1.1 阶段状态总览

截至 `2026-06-28`，当前项目已经从单一 Global Sentence-CSD 推进到 Category-aware CT-CSD，并且 Stage 4 的 MIL probe 工程链路已经部分跑通。但从方法可靠性看，后续主线不应继续押在 MIL 上，而应优先做 direction-based token selection，并把 feature preprocessing 抽成独立 Stage 5 验证。

| 阶段 | 方法 | 当前状态 | 关键产物 / 结果 | 当前判断 |
|---|---|---|---|---|
| Stage 0 | Global Sentence-CSD | 已冻结 | JBB + DIJA + Llama-Guard：`ASR = 74.0%` | 作为历史全局向量基线 |
| Stage 1 | CT-CSD | 已完成 M16 主实验；M4/M8 已有 judge；M12 生成完成但 judge 等 GPU0 | M16：`outputs/ct_csd_llada_m16/ct_csd_bank.pt`，`ASR = 74.0%`；M4：`70.0%`；M8：`71.0%` | 多簇本身不保证提升，cluster 数仍需谨慎解释 |
| Stage 2 | Random-K-CSD / cluster 数选择 | 部分完成 | M4/M8/M16 有安全指标，M12 等待 Llama-Guard | 不能宣称已有正式 `M*`，当前 Stage 3–6 暂沿用 M16 |
| Stage 3 | Category-aware CT-CSD | 已完成 | `outputs/category_ct_csd_llada_m16/ct_csd_bank.pt`，JBB DIJA `ASR = 71.0%` | category 信息有价值，是当前最稳主线 |
| Stage 4A | Direction-selected Category-CT-CSD | 未实现 | 待新增 token selection 分支 | 下一步优先主线 |
| Stage 4B | MIL-selected CT-CSD / Category-CT-CSD | 工程链路部分完成，实验未闭环 | probe 训练完成；`probe_ct_csd_llada_m16_tau07` bank 已生成，retention `78.57%`；probe-category bank 仍在运行 | MIL 暂只作为对照，不作为主方法 |
| Stage 4C | Random top-ratio Category-CT-CSD | 未实现 | 待新增 random selection 分支 | Stage 4A 的 token 数量对照 |
| Stage 5 | Feature-preprocessed Category-CT-CSD | 未实现 | 待新增 route feature 预处理和 bank 字段 | 第二优先级 |
| Stage 6 | Direction-selected + Feature-preprocessed Category-CT-CSD | 未实现 | 待 Stage 4A / Stage 5 单独验证后组合 | 最终候选方法 |

### 1.2 Stage 4–6 阶段命名

```text
Stage 0：Global Sentence-CSD
Stage 1：CT-CSD
Stage 2：Random-K-CSD
Stage 3：Category-aware CT-CSD
Stage 4A：Direction-selected Category-CT-CSD
Stage 4B：MIL-selected Category-CT-CSD
Stage 4C：Random top-ratio Category-CT-CSD
Stage 5：Feature-preprocessed Category-CT-CSD
Stage 6：Direction-selected + Feature-preprocessed Category-CT-CSD
```

这里的 `A/B/C` 只用于 Stage 4 的 token selection 对照。Feature preprocessing 作为独立 Stage 5；组合验证放在 Stage 6。

| 标签 | 可读别名 | 角色 |
|---|---|---|
| Stage 4A | `S4-direction` | 先验证 direction selection |
| Stage 4B | `S4-mil-control` | MIL 对照，不作为主线 |
| Stage 4C | `S4-random-control` | random selection 对照 |
| Stage 5 | `S5-feature` | 独立验证 feature preprocessing |
| Stage 6 | `S6-combined` | 组合两个有效模块 |

### 1.3 当前观察

目前的结果说明：

```text
Global Sentence-CSD 效果有限
CT-CSD 单纯多簇不一定稳定提升
Category-aware CT-CSD 有一定效果，说明 category 信息有价值
MIL token probe 的训练指标过好，但不一定说明 token-level score 可靠
```

其中 Stage 4 的 MIL token probe（多实例学习 token 探针）存在一个重要问题：当前 safe response 主要是 refusal template（拒答模板），safe 样本数量和多样性不足，所以 probe 可能学到的是：

```text
harmful completion style
vs
refusal style
```

而不一定是真正的 harmful semantic token（有害语义 token）。

因此，后续不建议继续把 MIL 作为唯一主线，而是把 MIL 降级为对照方法，引入更稳的 direction-based token selection。

### 1.4 当前方法边界

当前项目的核心边界应保持为：

```text
离线阶段可以更换 token selection 和 clustering/routing feature
推理阶段尽量保持统一 bank 接口和统一 steering 公式
不引入 prompt category classifier
不引入 soft routing
不引入 per-cluster threshold
不把 MIL probe 带到推理阶段
不把 prompt token 放入候选 token 集合
```

因此，后续主方法可以命名为：

```text
Direction-selected Feature-preprocessed Category-CT-CSD
```

简称：

```text
DS-FP Category-CT-CSD
```

其中：

| 模块 | 作用 | 是否主线 |
|---|---|---|
| Category-aware clustering | 按 harmful category 分组后聚类 | 是，继承 Stage 3 |
| Direction-selected token selection | 只让高 harmful-direction 投影 token 进入聚类 | 是，Stage 4A |
| Feature preprocessing | 用 `Normalize(P(h-\mu))` 做聚类和路由特征 | 是，Stage 5 |
| MIL token probe | learned token selector | 否，作为 Stage 4B 对照 |
| Random top-ratio | 控制 token 数减少带来的影响 | 否，作为 Stage 4C 对照 |

---

## 2. 必须统一的基本原则

### 2.1 token 候选只来自 response 部分

当前数据构造形式是：

```text
prompt + harmful response
prompt + safe response
```

虽然模型 forward（前向传播）输入是 `prompt + response`，但所有用于 CSD、CT-CSD、MIL、Direction-selected 的 token 候选都应该只来自 response 部分。

也就是：

```text
forward 输入：prompt + response
hidden state 提取：只截取 response_start 之后
token selection：只在 response tokens 上做
clustering：只用 response token hidden states
safe mean：只用 safe response token hidden states
```

### 2.2 prompt 的作用

prompt token 不应该进入平均、聚类、MIL 或 direction selection。

但是 response token hidden state 是在 `prompt + response` 的完整上下文中计算得到的，因此它仍然包含 prompt-conditioned information（提示词条件信息）。这是合理的，因为安全风险本来就是在给定 prompt 条件下发生的。

可以在论文或报告中明确写：

> Prompt tokens are not used as clustering or selection candidates. However, response token hidden states are extracted from prompt-conditioned forward passes, so they still encode prompt context.

中文：

> prompt token 不作为聚类或筛选候选，但 response token hidden state 是在 prompt-conditioned 输入下得到的，因此仍然包含 prompt 上下文信息。

---

## 3. 当前 Stage 3 主流程

当前 `Category-aware CT-CSD` 的主流程可以概括为：

```text
prompt + harmful response / safe response
        ↓
提取 response token hidden states
        ↓
过滤 special / blank tokens
        ↓
所有 harmful response tokens 按 category 分组
        ↓
每个 category 内部做 KMeans 聚类
        ↓
构造 category-cluster local steering vectors
        ↓
推理时 route + threshold steering
```

现在要补充两个替换点：

| 替换点 | 当前做法 | 新替换方法 |
|---|---|---|
| A：token selection | 所有 harmful response tokens 进入聚类 | Direction-selected token selection |
| B：clustering/routing feature | 使用 `unit(h)` 做聚类和路由 | 使用 `Normalize(P(h-\mu))` 做聚类和路由 |

---

## 4. 替换模块 A：Direction-selected Category-CT-CSD

### 4.1 模块定位

Direction-selected Category-CT-CSD 不改变整个推理公式，也不改变 category-aware clustering 的主体。它只替换当前流程中的 token selection 部分。

原来：

```text
所有 harmful response tokens 直接进入聚类
```

替换后：

```text
harmful response tokens
        ↓
用 category-level coarse CSD direction 打分
        ↓
每条 response 选 top 30%，最多 32 个 token
        ↓
选中的 tokens 再进入 category-aware clustering
```

插入位置：

```text
response hidden state 提取之后
KMeans 聚类之前
```

---

### 4.2 Coarse direction 选择

使用方案 B：**Category-level CSD**。

对每个 category \(c\)，构造一个 coarse direction（粗方向）：

$$
v_c^{coarse}
=
\mu_c^{harm}
-
\mu^{safe}
$$

其中：

$$
\mu_c^{harm}
=
\frac{1}{|\mathcal H_c|}
\sum_{h\in \mathcal H_c} h
$$

$$
\mu^{safe}
=
\frac{1}{|\mathcal S|}
\sum_{h\in \mathcal S} h
$$

含义：

| 符号 | 含义 |
|---|---|
| \(c\) | harmful category（有害类别） |
| \(\mathcal H_c\) | 类别 \(c\) 下的 harmful response token hidden states |
| \(\mathcal S\) | safe response token hidden states |
| \(v_c^{coarse}\) | 类别 \(c\) 的粗 CSD 方向 |

如果某个 category 样本太少，可以 fallback（回退）到全局方向：

$$
v_c^{coarse}
=
v_{\text{global}}
$$

---

### 4.3 Token scoring

对第 \(i\) 条 harmful response，假设它属于类别 \(c_i\)，第 \(j\) 个 response token hidden state 为：

$$
h_{ij}^{harm}
$$

用类别粗方向打分：

$$
s_{ij}
=
\left\langle
h_{ij}^{harm},
\hat v_{c_i}^{coarse}
\right\rangle
$$

其中：

$$
\hat v_{c_i}^{coarse}
=
\frac{v_{c_i}^{coarse}}
{\|v_{c_i}^{coarse}\|_2+\epsilon}
$$

直觉：

> 如果某个 response token hidden state 在该类别 harmful direction 上投影更大，它更可能携带 harmful signal。

---

### 4.4 Token selection 默认配置

暂时固定：

```text
selection_ratio = 0.3
max_selected_tokens = 32
coarse_direction_type = category
```

对每条 harmful response 内部排序，保留分数最高的 \(M_i\) 个 token：

$$
M_i
=
\min
\left(
32,
\left\lceil 0.3T_i \right\rceil
\right)
$$

其中：

| 符号 | 含义 |
|---|---|
| \(T_i\) | 第 \(i\) 条 harmful response 的有效 response token 数 |
| \(0.3T_i\) | 保留 top 30% token |
| \(32\) | 每条 response 最多保留 32 个 token |
| \(M_i\) | 最终保留 token 数 |

例如：

```text
如果 T_i = 128，则 M_i = min(32, ceil(38.4)) = 32
如果 T_i = 50，则 M_i = min(32, ceil(15)) = 15
```

---

### 4.5 基础过滤

在 direction scoring 和 top-ratio selection 之前，先做基础过滤。

建议过滤：

```text
special token
padding token
空白 token
纯标点
换行
列表编号
重复格式符号
```

这一步不是新方法贡献，只是为了避免 cluster 学到明显无意义的格式 token。

---

### 4.6 最终进入聚类的集合

原 Stage 3 中第 \(c\) 类进入聚类的集合为：

$$
\mathcal H_c
=
\{h_{ij}^{harm}\mid c_i=c\}
$$

Direction-selected 后变为：

$$
\mathcal H_c^{select}
=
\left\{
h_{ij}^{harm}
\mid
c_i=c,\ 
j\in \operatorname{TopM_i}(s_i)
\right\}
$$

然后继续类别内聚类：

$$
\mathcal H_c^{select}
=
C_{c,1}
\cup
C_{c,2}
\cup
\cdots
\cup
C_{c,K_c}
$$

每个 cluster 的 harmful center 为：

$$
\mu_{c,k}^{harm}
=
\frac{1}{|C_{c,k}|}
\sum_{h\in C_{c,k}}h
$$

local steering vector（局部引导向量）为：

$$
v_{c,k}
=
\mu_{c,k}^{harm}
-
\mu^{safe}
$$

归一化：

$$
\hat v_{c,k}
=
\frac{v_{c,k}}
{\|v_{c,k}\|_2+\epsilon}
$$

---

### 4.7 代码实现建议

新增 token selection mode（token 选择模式）：

```text
--token_selection all / direction_top_ratio / mil_top_ratio / random_top_ratio
--selection_ratio 0.3
--max_selected_tokens 32
--coarse_direction_type category
--coarse_direction_path path/to/coarse_direction.pt
```

建议新增接口：

```python
def select_harmful_response_tokens(
    h_tokens,
    token_ids,
    sample,
    args,
):
    ...
```

其中 `h_tokens` 必须是已经截取出的 response token hidden states，不包含 prompt tokens。

伪代码：

```python
h_tokens = extract_response_hidden_states(...)
h_tokens, token_ids = filter_response_tokens(...)

if args.token_selection == "direction_top_ratio":
    direction = get_category_coarse_direction(sample["semantic_category"])
    scores = h_tokens.float() @ direction.float()
    h_tokens, token_ids = top_ratio_select(
        h_tokens=h_tokens,
        token_ids=token_ids,
        scores=scores,
        ratio=0.3,
        max_tokens=32,
    )
```

---

## 5. 替换模块 B：Feature preprocessing

### 5.1 模块定位

Feature preprocessing 替换的是当前 clustering / routing 时使用的 feature 表示。

当前实现大致等价于：

$$
z = \operatorname{Normalize}(h)
$$

也就是使用 `unit(h)` 做 KMeans 聚类和推理时的 routing。

新替换方法：

$$
z
=
\operatorname{Normalize}
\left(
P(h-\mu)
\right)
$$

其中：

| 符号 | 含义 |
|---|---|
| \(h\) | 原始 response token hidden state |
| \(\mu\) | 训练集 response token hidden state 均值 |
| \(P\) | PCA 投影矩阵 |
| \(z\) | 用于 clustering / routing 的特征 |

---

### 5.2 它替换什么

替换：

```text
unit(h) 用于 KMeans 聚类和 route
```

为：

```text
Normalize(P(h - μ)) 用于 KMeans 聚类和 route
```

也就是：

```text
原始 hidden state h
        ↓
center：h - μ
        ↓
PCA：P(h - μ)
        ↓
L2 normalize
        ↓
z-space feature，用于聚类和路由
```

---

### 5.3 它不替换什么

Feature preprocessing 不改变 steering vector 的构造空间。

steering vector 仍然在原始 hidden space 中构造：

$$
v_{c,k}
=
\mu_{c,k}^{harm}
-
\mu^{safe}
$$

推理时 hidden state intervention（隐藏状态干预）也仍然在原始 hidden space 中执行：

$$
h'
=
h
-
\beta
[
\langle h,\hat v_{\hat c,\hat k}\rangle-\theta
]_+
\hat v_{\hat c,\hat k}
$$

因此要严格区分：

| 空间 | 用途 |
|---|---|
| \(z\)-space | clustering / routing |
| \(h\)-space | steering vector construction / hidden intervention |

---

### 5.4 推荐预处理版本

建议先做三个版本：

```text
l2_only
center_l2
center_pca128_l2
```

其中：

| 版本 | 含义 |
|---|---|
| `l2_only` | 当前默认做法，相当于 `Normalize(h)` |
| `center_l2` | 使用 `Normalize(h-\mu)` |
| `center_pca128_l2` | 使用 `Normalize(P_{128}(h-\mu))` |

推荐主实验先使用：

```text
center_pca128_l2
```

---

### 5.5 PCA / mean 的拟合范围

PCA 和 mean 的拟合也必须只使用 response token hidden states，不使用 prompt token。

推荐拟合数据：

```text
harmful selected response tokens + safe response tokens
```

也就是：

```text
用于拟合 μ 和 PCA：
    harmful selected response token hidden states
    safe response token hidden states

用于聚类：
    harmful selected response token hidden states

用于构造方向：
    原始 h-space 下的 harmful cluster mean - global safe mean
```

这样可以让 PCA 同时看到 harmful 和 safe 两侧的 response 分布，但不混入 prompt token。

---

### 5.6 bank 文件需要新增字段

建议 bank state 中新增：

```python
state["preprocess"] = {
    "mode": "center_pca128_l2",
    "mean": mean,
    "pca_components": components,
    "pca_dim": 128,
}
```

同时保存：

```python
state["route_centers"] = route_centers
state["raw_centers"] = raw_centers
state["vectors"] = vectors
state["vectors_unit"] = vectors_unit
```

其中：

| 字段 | 用途 |
|---|---|
| `route_centers` | z-space 中的中心，用于推理 route |
| `raw_centers` | 原始 h-space 中的中心，用于诊断或构造方向 |
| `vectors` | 原始 h-space 中的 local steering vectors |
| `vectors_unit` | 归一化后的 steering vectors |

---

### 5.7 推理时 route 逻辑修改

当前：

```python
h_unit = normalize(hidden)
sim = h_unit @ centers_unit.T
route = sim.argmax(dim=-1)
```

替换为：

```python
z = transform_route_features(hidden, preprocess)
sim = z @ route_centers.T
route = sim.argmax(dim=-1)
```

但 steering 仍然使用原始 hidden：

```python
v = vectors_unit[route]
score = hidden @ v
alpha = beta * clamp(score - theta, min=0)
hidden_new = hidden - alpha * v
```

---

## 6. 改进后的整体流程

### 6.1 Stage 3：原始 Category-aware CT-CSD

```text
prompt + harmful response
        ↓
提取 response token hidden states
        ↓
所有 harmful response tokens 按 category 聚类
        ↓
构造 local steering vectors
```

### 6.2 Stage 4A：加入 Direction-selected token selection

```text
prompt + harmful response
        ↓
提取 response token hidden states
        ↓
category-level coarse direction 打分
        ↓
每条 response 选 top 30%，最多 32 个
        ↓
selected tokens 按 category 聚类
        ↓
构造 local steering vectors
```

### 6.3 Stage 5：加入 Feature preprocessing

```text
prompt + harmful response
        ↓
提取 response token hidden states
        ↓
所有 harmful response tokens
        ↓
z = Normalize(P(h - μ))
        ↓
用 z 做 KMeans 聚类
        ↓
用原始 h-space 构造 local steering vectors
```

### 6.4 Stage 6：两个替换模块组合

```text
prompt + harmful response
        ↓
提取 response token hidden states
        ↓
category-level coarse direction 打分
        ↓
每条 response 选 top 30%，最多 32 个
        ↓
z = Normalize(P(h - μ))
        ↓
selected tokens 按 category 聚类
        ↓
用原始 h-space 构造 local steering vectors
```

---

## 7. 推理阶段保持统一

无论使用 Stage 3、Stage 4A、Stage 5 还是 Stage 6，推理阶段都保持：

```text
route
↓
alignment
↓
threshold steering
```

### 7.1 Routing

如果没有 feature preprocessing：

$$
(\hat c,\hat k)
=
\arg\max_{c,k}
\operatorname{cos}
\left(
h,
\mu_{c,k}^{harm}
\right)
$$

如果使用 feature preprocessing：

$$
(\hat c,\hat k)
=
\arg\max_{c,k}
\operatorname{cos}
\left(
z(h),
z_{c,k}
\right)
$$

### 7.2 Alignment

$$
s(h)
=
\langle h,\hat v_{\hat c,\hat k}\rangle
$$

### 7.3 Threshold steering

$$
\alpha(h)
=
\beta[s(h)-\theta]_+
$$

其中：

$$
[x]_+
=
\max(x,0)
$$

### 7.4 Hidden update

$$
h'
=
h
-
\alpha(h)\hat v_{\hat c,\hat k}
$$

完整写法：

$$
h'
=
h
-
\beta
\left[
\langle h,\hat v_{\hat c,\hat k}\rangle-\theta
\right]_+
\hat v_{\hat c,\hat k}
$$

### 7.5 关键说明

每个 hidden state 都会被 route 到一个最近的 category-cluster center，但不是每个 token 都会被 steering。只有：

$$
s(h) > \theta
$$

时才真正引导。

---

## 8. 实验设计

### 8.1 主实验矩阵

| 实验 | token selection | feature preprocessing | 目的 |
|---|---|---|---|
| Stage 3 | all | l2_only | 当前 Category-aware CT-CSD baseline |
| Stage 4A | direction_top_ratio | l2_only | 验证 direction-based token selection |
| Stage 4C | random_top_ratio | l2_only | 排除 token 数量减少带来的影响 |
| Stage 5 | all | center_pca128_l2 | 验证 feature preprocessing |
| Stage 6 | direction_top_ratio | center_pca128_l2 | 验证两个替换模块的组合效果 |
| Stage 4B | mil_top_ratio | l2_only | MIL 对照 |

### 8.2 第一轮推荐只跑

```text
Stage 3：all + l2_only
Stage 4A：direction_top_ratio + l2_only
Stage 4C：random_top_ratio + l2_only
Stage 5：all + center_pca128_l2
Stage 6：direction_top_ratio + center_pca128_l2
```

### 8.3 固定参数

```text
target_layer = 31
selection_ratio = 0.3
max_selected_tokens = 32
coarse_direction_type = category
cluster_budget = 当前 Stage3 设置
beta = 当前最优
theta = 当前最优
generation settings = 不变
```

### 8.4 后续消融参数

```text
selection_ratio = 0.2 / 0.3 / 0.5
max_selected_tokens = 16 / 32 / 64
feature_preprocess = l2_only / center_l2 / center_pca128_l2 / center_pca256_l2
cluster_budget = 16 / 32 / 64
coarse_direction_type = global / category
```

---

## 9. 诊断指标

### 9.1 Token selection 诊断

需要记录：

```text
每条 response 平均保留 token 数
每个 category 保留 token 数
是否有 category 被筛空
选中 token 的平均 score
未选 token 的平均 score
选中 token 文本样例
```

### 9.2 Cluster 诊断

需要记录：

```text
cluster size
nearest tokens
cluster category purity
小 cluster 数量
是否聚到标点 / 模板词
```

### 9.3 Routing 诊断

需要记录：

```text
route histogram
routing margin
每个 category-cluster 被使用次数
```

其中 routing margin 可以定义为：

$$
m(h)
=
r_1(h)-r_2(h)
$$

其中 \(r_1(h)\) 是最近 cluster 的相似度，\(r_2(h)\) 是第二近 cluster 的相似度。

### 9.4 Steering 诊断

需要记录：

```text
steering activation rate
mean alignment score
mean alpha
active token count
remask token count
```

其中最关键的是：

```text
steering activation rate
```

因为不是所有 route 到 cluster 的 token 都真正被 steering。

### 9.5 评测指标

```text
ASR
refusal rate
over-refusal
runtime
```

---

## 10. 风险与应对

### 10.1 Direction selection 可能选错 token

风险：

```text
coarse direction 不够准，top-ratio 可能选到模板词或无关 token
```

应对：

```text
加入 random_top_ratio 对照
检查 selected token examples
比较 selection_ratio = 0.2 / 0.3 / 0.5
```

### 10.2 Coarse direction 本身不准

风险：

```text
某些 category 样本少，category-level coarse direction 不稳定
```

应对：

```text
小 category fallback 到 global direction
对比 global coarse vs category coarse
```

### 10.3 Feature preprocessing 可能破坏路由

风险：

```text
PCA 后的 z-space 不一定比原始 unit(h) 更适合 route
```

应对：

```text
保留 l2_only baseline
只在 clustering / routing 使用 z-space
steering 始终使用原始 h-space
```

### 10.4 小 category token 太少

风险：

```text
direction selection 后小类别 token 更少，聚类不稳定
```

应对：

```text
设置 min category tokens
小类 fallback
cluster budget 自适应分配
必要时降低该类 K_c
```

### 10.5 MIL 暂时不稳定

风险：

```text
safe 样本少，MIL 可能学到 refusal style
```

应对：

```text
MIL 作为 Stage 4B 对照
不作为主线
后续补充 safe alternative responses 后再重新评估
```

---

## 11. 最终推荐路线

### 11.1 第一阶段：验证 Direction-selected

对比：

```text
Stage 3 vs Stage 4A vs Stage 4C
```

目标：

```text
确认 direction-based token selection 是否优于 all-token 和 random-token
```

如果 Stage 4A 优于 Stage 4C，说明 token selection 的提升不是简单来自 token 数变少，而是来自 direction-based selection。

### 11.2 第二阶段：验证 Feature preprocessing

对比：

```text
Stage 3 vs Stage 5
Stage 4A vs Stage 6
```

目标：

```text
确认 center_pca128_l2 是否改善 clustering / routing
```

### 11.3 第三阶段：组合最佳方法

最终候选：

```text
Direction-selected + center_pca128_l2
```

也就是：

```text
Stage 6
```

### 11.4 第四阶段：再考虑 MIL

只有在补充更多 safe alternative responses 后，再重新评估：

```text
MIL-selected Category-CT-CSD
```

否则当前 MIL 只作为对照，不作为主线。

---

## 12. 具体实施计划

### 12.1 实施总原则

后续实施按“先小闭环、再全量、最后评估”的顺序推进。每个新模块都必须先在 smoke 数据上验证 bank 可生成、metadata 可读、旧 bank 兼容，再跑全量 JBB DIJA 评测。

```text
先保持 Stage 3 baseline 不变
先单独验证 Direction-selected
再单独验证 Feature preprocessing
最后组合两个模块
MIL 只作为横向对照
不在推理阶段引入新模型或新分类器
不规划 git commit / push
```

### 12.2 文件级改动范围

| 文件 | 改动内容 | 原则 |
|---|---|---|
| `utils/make_ct_csd_llada.py` | 新增 token selection、direction coarse direction、random top-ratio、feature preprocessing、bank metadata、诊断输出 | 主要改动点 |
| `utils/ct_csd_bank.py` | 读取可选 `preprocess` / `route_centers`，在 routing 前做同样的 feature transform | 保持旧 `ct_csd_v1` bank 兼容 |
| `eval_llada_steering.py` | 原则上不改；只通过 `CTCSDBank` 读取新版 bank | 推理入口稳定 |
| `tests/test_make_ct_csd_llada.py` | 覆盖 selection、coarse direction、preprocess fit/transform、metadata | 单元测试先行 |
| `tests/test_ct_csd_bank.py` | 覆盖旧 bank 兼容、新 route feature routing、diagnostics | 防止推理回归 |
| `docs/stage4_to_stage6_direction_feature_metrics.md` | 汇总 `S4-direction` / `S4-random-control` / `S5-feature` / `S6-combined` 指标 | 结果落盘 |

### 12.3 Milestone 0：冻结当前状态

目标：

```text
把 Stage 3 作为后续主 baseline
把 MIL 运行状态记为对照，不阻塞新主线
把 M16 标为当前沿用参数，而不是正式 M*
```

执行项：

```text
确认 Stage 3 bank、generation、judge、diagnostics 路径存在
确认 Stage 4 MIL probe_ct_csd bank 已生成但 probe_category 仍未闭环
确认 m12 judge 仍等待 GPU0，不把 Stage 2 M* 写死
```

验收：

```text
文档中不再把 MIL 写成唯一后续主线
文档中明确 Stage 4A / Stage 4B / Stage 4C / Stage 5 / Stage 6 的角色
所有后续实验都默认对比 Stage 3 Category-aware CT-CSD M16
```

### 12.4 Milestone 1：统一 token selection 接口

目标：

```text
让 all / direction_top_ratio / random_top_ratio / mil_top_ratio 共用同一 selection 接口
确保 selection 只作用于 harmful response tokens
safe response tokens 仍只用于 safe mean 和 preprocess 拟合
```

建议新增 CLI：

```text
--token_selection all / direction_top_ratio / random_top_ratio / mil_top_ratio
--selection_ratio 0.3
--max_selected_tokens 32
--coarse_direction_type category / global
--min_coarse_tokens 1024
--feature_preprocess l2_only / center_l2 / center_pca128_l2 / center_pca256_l2
--pca_dim 128
```

建议新增函数：

```python
def top_ratio_select(h_tokens, token_ids, scores, ratio, max_tokens):
    ...


def select_harmful_response_tokens(h_tokens, token_ids, sample, args, selection_state):
    ...
```

接入位置：

```text
response hidden state 提取之后
filter_response_tokens 之后
KMeans partial_fit / predict 之前
cluster sum accumulate 之前
```

验收：

```text
token_selection=all 时，新旧 category_ct_csd 输出 metadata 兼容
direction/random/mil 三种模式都写入 state["config"]["token_selection"]
空 selection 必须有 fallback，不允许某条样本导致整个 category 静默消失
```

### 12.5 Milestone 2：实现 Stage 4A Direction-selected Category-CT-CSD

目标：

```text
用 category-level coarse CSD direction 筛掉低价值 harmful tokens
验证提升是否超过 random top-ratio 对照
```

实现步骤：

```text
1. 在全量 harmful response tokens 上统计 per-category harmful mean。
2. 在 safe response tokens 上统计 sample-balanced global safe mean。
3. 构造 v_c^{coarse} = mu_c^{harm} - mu^{safe}。
4. category token 数低于 min_coarse_tokens 时 fallback 到 global coarse direction。
5. 对每条 harmful response 内 token 打分 s = h @ unit(v_c^{coarse})。
6. 保留 min(max_selected_tokens, ceil(selection_ratio * T_i)) 个最高分 token。
7. selected tokens 进入原 Stage 3 category-aware KMeans。
8. steering vectors 仍用原始 h-space 的 selected harmful cluster mean - global safe mean 构造。
```

输出目录建议：

```text
outputs/category_ct_csd_direction_r03_m16_smoke/
outputs/category_ct_csd_direction_r03_m16/
outputs/jbb_dija_category_ct_csd_direction_r03_m16/
```

必须新增诊断：

```text
token_selection_summary.json
direction_selected_tokens.md
cluster_token_top_terms.md
ct_csd_bank_summary.json
```

验收：

```text
probe_empty_samples 等价指标为 0，或有明确 fallback 计数
每个 category 的 selected token 数 > 对应该 category cluster 数
direction_selected_tokens.md 中不能主要是空白、标点、列表编号或模板词
Stage 4A 的 ASR 优于 Stage 3，或者 activation_rate / over-refusal 有明确改善
```

### 12.6 Milestone 3：实现 Stage 4C Random top-ratio 对照

目标：

```text
排除 Stage 4A 的收益只是来自 token 数减少
```

实现步骤：

```text
1. 复用 top_ratio_select 的保留数量 M_i。
2. 用固定 seed 在每条 response 内随机采样 M_i 个 token。
3. 保持 category clustering、safe mean、steering、eval 参数完全不变。
```

输出目录建议：

```text
outputs/category_ct_csd_random_r03_m16/
outputs/jbb_dija_category_ct_csd_random_r03_m16/
```

验收：

```text
Stage 4A 优于 Stage 4C，才能说明 direction score 有效
如果 Stage 4A 与 Stage 4C 接近，只能说 token pruning 有效，不能声称 semantic selection 有效
```

### 12.7 Milestone 4：实现 Stage 5 Feature preprocessing

目标：

```text
只替换 clustering / routing feature，不改变 steering vector 所在空间
```

实现步骤：

```text
1. 新增 fit_route_preprocess，支持 l2_only、center_l2、center_pca128_l2、center_pca256_l2。
2. PCA / mean 只拟合 response token hidden states。
3. KMeans partial_fit 和 predict 使用 z = Normalize(P(h - mu))。
4. cluster raw center 和 steering vector 仍在原始 h-space 构造。
5. bank 中同时保存 route_centers、raw_centers、vectors、vectors_unit、preprocess。
6. CTCSDBank.route 在新版 bank 上先 transform hidden，再和 route_centers 做相似度。
7. 旧 bank 没有 preprocess / route_centers 时，继续使用现有 centers_unit 路由。
```

bank 字段建议：

```python
state["preprocess"] = {
    "mode": "center_pca128_l2",
    "mean": mean,
    "pca_components": components,
    "pca_dim": 128,
}
state["route_centers"] = route_centers
state["raw_centers"] = raw_centers
state["vectors"] = vectors
state["vectors_unit"] = vectors_unit
```

输出目录建议：

```text
outputs/category_ct_csd_fp_center_pca128_m16/
outputs/jbb_dija_category_ct_csd_fp_center_pca128_m16/
```

验收：

```text
旧 ct_csd_bank.pt 和旧 category_ct_csd bank 仍能由 CTCSDBank 加载
feature_preprocess=l2_only 时 routing 行为和旧实现保持一致
center_pca128_l2 的 route histogram 不出现单簇塌缩
Stage 5 至少在 cluster 诊断或 ASR 上优于 Stage 3，否则不进入默认组合
```

### 12.8 Milestone 5：实现 Stage 6 组合方法

目标：

```text
验证 Direction-selected token selection 和 center_pca128_l2 是否互补
```

组合配置：

```text
token_selection = direction_top_ratio
selection_ratio = 0.3
max_selected_tokens = 32
coarse_direction_type = category
feature_preprocess = center_pca128_l2
pca_dim = 128
num_total_clusters = 16
target_layer = 31
```

输出目录建议：

```text
outputs/category_ct_csd_direction_fp_center_pca128_r03_m16/
outputs/jbb_dija_category_ct_csd_direction_fp_center_pca128_r03_m16/
```

验收：

```text
Stage 6 优于 Stage 3
Stage 6 不显著差于 Stage 4A 和 Stage 5 中的最佳单模块
route / active 诊断没有出现异常单簇主导
refusal rate、over-refusal、空回复比例没有明显恶化
```

### 12.9 推荐实验矩阵和决策门槛

第一轮只跑以下 5 个点：

| 实验 | token selection | feature preprocessing | 目的 | 是否必须 |
|---|---|---|---|---|
| Stage 3 | all | l2_only | 已有 baseline | 是 |
| Stage 4A | direction_top_ratio | l2_only | 验证 direction selection | 是 |
| Stage 4C | random_top_ratio | l2_only | token 数控制 | 是 |
| Stage 5 | all | center_pca128_l2 | 验证 route feature | 是 |
| Stage 6 | direction_top_ratio | center_pca128_l2 | 组合候选 | 是 |

决策规则：

```text
如果 Stage 4A <= Stage 4C：
    不把 direction selection 写成有效贡献，只保留 token pruning 观察。

如果 Stage 5 route histogram 单簇塌缩：
    不继续跑 Stage 6 的 PCA 组合，先回退 center_l2。

如果 Stage 6 最好：
    主方法采用 DS-FP Category-CT-CSD。

如果 Stage 4A 最好：
    主方法采用 Direction-selected Category-CT-CSD，feature preprocessing 降级为负结果。

如果 Stage 3 仍最好：
    保留 Stage 3 为最终工程方法，本计划两个模块写成后续探索。
```

### 12.10 固定评测口径

除非新增文档明确说明，否则 Stage 4–6 的主线实验都沿用：

```text
model_path = /dev/shm/LLaDA-8B-Instruct
target_layer = 31
num_total_clusters = 16
category_key = semantic_category
max_response_len = 128
max_total_len = 2048
attack_method = DIJA
batch_size = 32
sampling_steps = 128
mask_length = 128
block_size = 128
dija_mask_counts = 128
remasking = low_confidence
sampler = steering
remdm_number = 4
cfg = 0
self_reminder = False
steering_overshoot = 1.0
alignment_threshold = 0.0
initial_steering_ratio = 0.1
max_refinement_iters = 5
judge = /dev/shm/Llama-Guard-4-12B
```

### 12.11 最小验证命令

代码改动后先跑单测：

```bash
python -m unittest tests.test_make_ct_csd_llada tests.test_ct_csd_bank
```

smoke bank 构造建议：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python utils/make_ct_csd_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json .worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json \
  --refusals_txt utils/refusals.txt \
  --output_dir outputs/category_ct_csd_direction_r03_m16_smoke \
  --target_layer 31 \
  --max_samples 16 \
  --method category_ct_csd \
  --category_key semantic_category \
  --num_total_clusters 16 \
  --token_selection direction_top_ratio \
  --selection_ratio 0.3 \
  --max_selected_tokens 32 \
  --coarse_direction_type category \
  --feature_preprocess l2_only \
  --device cuda \
  --seed 42
```

全量 bank 构造建议：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python utils/make_ct_csd_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json .worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json \
  --refusals_txt utils/refusals.txt \
  --output_dir outputs/category_ct_csd_direction_r03_m16 \
  --target_layer 31 \
  --method category_ct_csd \
  --category_key semantic_category \
  --num_total_clusters 16 \
  --token_selection direction_top_ratio \
  --selection_ratio 0.3 \
  --max_selected_tokens 32 \
  --coarse_direction_type category \
  --feature_preprocess l2_only \
  --device cuda \
  --seed 42
```

评估命令沿用 Stage 3 的 JBB DIJA 口径，只替换：

```text
--steering_vector_path outputs/category_ct_csd_direction_r03_m16/ct_csd_bank.pt
--generated_samples_path outputs/jbb_dija_category_ct_csd_direction_r03_m16
```

### 12.12 文档产物

每轮实验完成后，至少更新或新增：

```text
docs/stage4_to_stage6_direction_feature_metrics.md
docs/stage4_token_selection_diagnostics.md
docs/stage5_feature_preprocess_route_analysis.md
```

其中 `docs/stage4_to_stage6_direction_feature_metrics.md` 必须包含：

```text
ASR / unsafe_count / total_samples
refusal rate / over-refusal / 空回复比例
steering activation rate
route histogram
selected token retention ratio
per-category retention ratio
cluster top terms
runtime
结论：保留、组合、回退或放弃哪个模块
```

---

## 13. 报告 / 论文叙事

### 13.1 问题

原始 Category-aware CT-CSD 使用所有 harmful response tokens 聚类，但 harmful response 中大量 token 是功能词、模板词、标点或格式 token。直接聚类会让 local steering vectors 受到这些低价值 token 的干扰。

### 13.2 替换方法一：Direction-selected token selection

使用 category-level coarse CSD direction 对 response tokens 打分，每条 harmful response 只保留投影最高的 top 30%，最多 32 个 token。该方法不依赖额外 MIL probe，也不需要大量 safe 数据。

### 13.3 替换方法二：Feature preprocessing

用 `center + PCA + L2` 替换原始 `unit(h)` 表示，用于 clustering 和 routing。注意 steering vector 仍然在原始 hidden space 中构造和执行。

### 13.4 实验逻辑

```text
Stage 4A：验证 token selection
Stage 4C：random selection 对照
Stage 5：验证 feature preprocessing
Stage 6：验证两个替换模块组合效果
Stage 4B：MIL 对照
```

### 13.5 预期结论

```text
如果 Stage 4A > Stage 3：
说明只使用 direction-selected tokens 比 all tokens 更干净。

如果 Stage 4A > Stage 4C：
说明提升来自 direction-based selection，而不是 token 数减少。

如果 Stage 5 > Stage 3：
说明 feature preprocessing 改善了 clustering / routing。

如果 Stage 6 最好：
说明 direction-selected token selection 和 feature preprocessing 可以互补。
```

---

## 14. 最终一句话总结

本改进计划不重写原有 Category-aware CT-CSD，而是在现有流程中加入两个可替换模块：第一，使用 category-level coarse CSD direction 对 harmful response tokens 进行 direction-selected token selection，每条 response 保留 \(M_i=\min(32,\lceil0.3T_i\rceil)\) 个高分 token，减少模板词和功能词对聚类的污染；第二，使用 `center + PCA + L2` 的 feature preprocessing 替换原始 `unit(h)` 路由特征，提高 clustering / routing 的稳定性。推理阶段仍保持原有 route + threshold steering 公式不变。
