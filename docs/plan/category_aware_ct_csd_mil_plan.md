# Category-aware CT-CSD with MIL Token Selection 完整方案

> 版本：v1.0，三轮 review（审阅）后定稿  
> 适用项目：DLM Steering Remasking，目前优先落地到 `LLaDA`，随后迁移到 `Dream`  
> 核心目标：在不显著增加方法复杂度的前提下，将原始单一 CSD（Contrastive Steering Direction，对比引导方向）升级为类别感知的局部 CSD 向量组，并用 MIL token probe（多实例学习 token 探针）作为可替换的 token selection（token 选择）模块。

---

## 0. 三轮 review 后的定稿结论

### Review 1：方法主线收敛

早期方案里曾把 `Token-CSD` 单独列成核心方法，但这不够准确。对于 DLM（Diffusion Language Model，扩散语言模型）当前实现来说，sentence representation（句子表示）本来就是由 response token hidden states（回答 token 隐藏状态）平均得到的。

因此，最终不把 `Token-CSD` 作为核心贡献。真正的核心改动是：

```text
原始 Global Sentence-CSD：
response token hidden states
→ 先在每条 response 内平均
→ 再跨样本平均
→ 得到一个全局 steering vector

新方法 CT-CSD：
response token hidden states
→ 不先平均
→ 先聚类
→ 再在每个 cluster 内平均
→ 得到多个局部 steering vectors
```

也就是说，新方法的重点不是“token 表示 vs 句子表示”，而是：

> **将 averaging（平均）操作从聚类之前推迟到聚类之后。**

---

### Review 2：复杂度控制

最终方案明确删掉以下复杂模块：

| 模块 | 处理方式 | 原因 |
|---|---|---|
| safe token clustering（安全 token 聚类） | 不做 | 会引入配对噪声，难解释 |
| harmful-safe cluster 笛卡尔积 | 不做 | 会产生大量语义不对应的随机方向 |
| soft routing（软路由） | 不做 | 增加温度、权重等额外变量 |
| per-cluster threshold（逐簇阈值） | 不做 | 会和向量效果混在一起，影响消融解释 |
| prompt category classifier（提示类别分类器） | 不做 | 推理时不新增分类器 |
| diffusion-aware bank（扩散时间感知向量库） | 暂不做 | 放到后续扩展，不进入第一版主线 |
| PCA（主成分分析降维） | 不作为方法 | 如需使用，只作为工程降维备选 |

最终只保留：

```text
1. harmful token hidden states 聚类
2. category-aware clustering（类别感知聚类）
3. global safe mean（全局安全均值）
4. hard routing（硬路由）
5. 原论文 threshold-gated steering（阈值门控引导）公式
6. MIL token probe 作为 token selection 替换模块
```

---

### Review 3：工程落地检查

当前仓库已有两个关键位置：

1. `utils/make_csd_llada.py` 负责构造 CSD 向量；当前逻辑会提取每层 response token hidden states，然后先对每条 response 求均值。
2. `eval_llada_steering.py` 负责推理；当前逻辑加载单个 `steering_vector`，在 `_per_token_alignment` 和 `_build_adaptive_steering_hook` 中使用这个全局向量。

最终方案只需要替换两个工程点：

```text
离线构造：
从“输出一个 layer_i vector”
替换为“输出多个 category-cluster vectors + harmful centers”

推理阶段：
从“所有 hidden states 用同一个 global vector”
替换为“每个 hidden state 先 route 到最近 harmful center，再用对应 local vector”
```

同时保留原始 `steering_vectors.pt` 加载路径，方便 baseline（基线）对照。

---

## 1. 背景与当前问题

### 1.1 当前 Global Sentence-CSD 的构造方式

当前项目中的 CSD（Contrastive Steering Direction，对比引导方向）构造逻辑可以概括为：

```text
prompt + harmful response
→ 提取 target layer 或所有 layers 的 response token hidden states
→ 每条 response 内部求平均
→ 跨 harmful samples 求平均
→ 得到 harmful mean

prompt + safe refusal
→ 提取 target layer 或所有 layers 的 response token hidden states
→ 每条 response 内部求平均
→ 跨 safe samples 求平均
→ 得到 safe mean

harmful mean - safe mean
→ 得到一个 global steering vector
```

公式为：

$$
\bar h_i^{harm}
=
\frac{1}{T_i^{harm}}
\sum_{j=1}^{T_i^{harm}}
h_{ij}^{harm}
$$

$$
\bar h_i^{safe}
=
\frac{1}{T_i^{safe}}
\sum_{j=1}^{T_i^{safe}}
h_{ij}^{safe}
$$

$$
v_{global}
=
\mathbb{E}_i[\bar h_i^{harm}]
-
\mathbb{E}_i[\bar h_i^{safe}]
$$

其中：

| 符号 | 含义 |
|---|---|
| $h_{ij}^{harm}$ | 第 $i$ 条 harmful response（有害回答）中第 $j$ 个 response token 的 hidden state（隐藏状态） |
| $h_{ij}^{safe}$ | 第 $i$ 条 safe response（安全回答）中第 $j$ 个 response token 的 hidden state |
| $\bar h_i^{harm}$ | 第 $i$ 条 harmful response 的句子级 hidden representation（隐藏表示） |
| $v_{global}$ | 单一全局 CSD steering vector（引导向量） |

---

### 1.2 当前推理阶段的使用方式

当前推理中，模型会在 target layer（目标层）的 hidden state 上计算 alignment（对齐分数）：

$$
s(h)
=
\langle h, \hat v_{global}\rangle
$$

如果超过 alignment threshold（对齐阈值） $\theta$，就进行 steering（引导）：

$$
\alpha(h)
=
\beta[s(h)-\theta]_+
$$

$$
h'
=
h-
\alpha(h)\hat v_{global}
$$

其中：

$$
[x]_+ = \max(x,0)
$$

所以当前方法已经不是所有 token 都强制引导，而是：

```text
如果 s(h) <= θ：
    不引导

如果 s(h) > θ：
    沿 global harmful direction 减去一部分
```

---

### 1.3 当前方法的问题

当前方法的主要问题不是“没有 token hidden states”，而是这些 token hidden states 在构造 CSD 时被过早平均了。

这会带来三个问题：

#### 问题 1：token-level harmful signal（token 级有害信号）被抹平

一条 harmful response 中，不同 token 的作用不同：

```text
Here is a detailed guide to ...
```

其中 `Here`、`is`、标点、换行可能主要是模板或语法 token，而真正表达 harmful semantic signal（有害语义信号）的可能是动作、对象、材料、步骤、参数等 token。

如果直接对整条 response 的 token hidden states 求平均，这些局部有害信息会被模板 token 和普通 token 稀释。

#### 问题 2：不同 harmful category（有害类别）被混成一个方向

HarmBench 数据中通常包含 `semantic_category`、`functional_category` 等类别字段。不同类别的 harmful response 可能对应不同 hidden state region（隐藏状态区域）。

例如：

```text
cybersecurity misuse（网络安全滥用）
chemical / biological misuse（化学/生物滥用）
fraud / deception（欺诈/欺骗）
violence（暴力）
privacy（隐私侵犯）
```

如果全部平均成一个方向，最终的 global vector 可能变成多个风险类别的混合方向。

#### 问题 3：一个 global vector 难以覆盖局部表示结构

当前推理中所有 hidden states 都使用同一个 $\hat v_{global}$。如果 harmfulness（有害性）在 hidden space（隐藏空间）中不是单一线性方向，而是多个局部区域，那么单一方向可能不稳定。

当前项目里的探索结果也显示，继续只调一个 global vector 的 `beta` 和 `threshold`，收益空间可能有限。

---

## 2. 方法总览

最终方法主线为：

```text
Global Sentence-CSD
↓
CT-CSD
↓
Category-aware CT-CSD
↓
Probe-Category-aware CT-CSD
```

其中：

| 方法 | 作用 |
|---|---|
| Global Sentence-CSD | 当前 baseline（基线） |
| CT-CSD | 验证“先聚类再平均”是否优于“先平均” |
| Category-aware CT-CSD | 验证 category（类别）信息是否能让聚类更干净 |
| Probe-Category-aware CT-CSD | 验证 MIL token probe（多实例学习 token 探针）筛选 token 是否进一步提升效果 |

最重要的是，MIL token probe 不作为一个复杂的新主方法，而是作为 token selection（token 选择）替换模块：

```text
原来：所有 harmful response tokens 都进入聚类
替换后：只有 MIL token probe 打高分的 harmful tokens 进入聚类
```

---

## 3. Core Method 1：CT-CSD

CT-CSD 全称为：

> Clustered Token-level Contrastive Steering Direction（聚类 token 级对比引导方向）

### 3.1 方法动机

原始 Global Sentence-CSD 是：

```text
先平均 → 再构造方向
```

CT-CSD 改成：

```text
先聚类 → 再平均构造方向
```

也就是保留 harmful response 中的 token hidden states，不先压缩为句子表示，而是先看 hidden state 分布中是否存在多个局部 cluster（聚类簇）。

---

### 3.2 离线构造流程

#### Step 1：收集 harmful token hidden states

对训练集中的 harmful responses，提取 target layer 的 response token hidden states：

$$
\mathcal{H}
=
\{h_{ij}^{harm}\}
$$

其中 $h_{ij}^{harm}\in\mathbb{R}^D$。

第一版建议只使用一个 target layer，例如当前 LLaDA 实验中的：

```text
target_layer = 31
```

这样可以减少存储和调试成本。

---

#### Step 2：过滤明显无效 token

这里不是做复杂 token 标注，只做基础过滤：

```text
过滤 special tokens（特殊 token）
过滤 padding tokens（填充 token）
过滤空白 token
可选过滤纯标点 token
```

注意：这里不能声称剩下的 token 都是“有害 token”。更准确的说法是：

> 它们是 harmful response distribution（有害回答分布）中的 token hidden states。

---

#### Step 3：对 harmful token hidden states 聚类

对 $\mathcal{H}$ 做 KMeans（K 均值聚类）：

$$
\mathcal{H}
=
C_1\cup C_2\cup \cdots \cup C_K
$$

其中 $K$ 是唯一核心超参数之一。

建议初始取值：

```text
K = 4, 8, 16
```

用于聚类的特征可以使用 L2-normalized hidden state（L2 归一化隐藏状态）：

$$
\tilde h
=
\frac{h}{\|h\|_2+\epsilon}
$$

但真正构造 steering vector 时仍然回到原始 hidden space（隐藏空间）中计算均值。

---

#### Step 4：构造 harmful cluster center

对每个 cluster：

$$
\mu_k^{harm}
=
\frac{1}{|C_k|}
\sum_{h\in C_k}h
$$

其中 $\mu_k^{harm}$ 是第 $k$ 个 harmful cluster center（有害聚类中心）。

这个 center 有两个作用：

```text
1. 推理时作为 prototype（原型）做 routing（路由）
2. 离线时用于和 safe mean 构造 local steering vector
```

---

#### Step 5：构造 global safe mean

为了不引入 safe clustering（安全聚类）和 safe pairing（安全配对）的复杂性，safe side（安全侧）只使用一个 global safe mean（全局安全均值）。

建议沿用原项目的 sample-balanced mean（样本均衡平均）口径：

$$
\bar h_i^{safe}
=
\frac{1}{T_i^{safe}}
\sum_{j=1}^{T_i^{safe}}h_{ij}^{safe}
$$

$$
\mu^{safe}
=
\mathbb{E}_i[\bar h_i^{safe}]
$$

这样可以避免某些较长 safe refusal（安全拒答）因为 token 多而占更大权重。

---

#### Step 6：构造 local steering vectors

每个 harmful cluster 构造一个局部方向：

$$
v_k
=
\mu_k^{harm}-\mu^{safe}
$$

归一化：

$$
\hat v_k
=
\frac{v_k}{\|v_k\|_2+\epsilon}
$$

直觉是：

```text
μ_safe → μ_k^harm 的方向：第 k 类局部有害方向
推理时减去这个方向：把 hidden state 从该局部有害方向往安全侧拉回
```

---

### 3.3 CT-CSD 推理流程

对于当前 hidden state $h$，先找最近的 harmful cluster center：

$$
\hat k
=
\arg\max_k
\operatorname{cos}(h,\mu_k^{harm})
$$

选中对应的局部方向：

$$
\hat v_{\hat k}
$$

然后计算 harmful alignment（有害对齐分数）：

$$
s(h)
=
\langle h,\hat v_{\hat k}\rangle
$$

只有当 $s(h)>\theta$ 时才执行引导：

$$
\alpha(h)
=
\beta[s(h)-\theta]_+
$$

$$
h'
=
h-
\alpha(h)\hat v_{\hat k}
$$

也就是：

$$
h'
=
h-
\beta
\left[
\langle h,\hat v_{\hat k}\rangle-\theta
\right]_+
\hat v_{\hat k}
$$

这里要强调：

> routing（路由）只负责选择候选方向，threshold（阈值）才决定是否真正干预。

---

### 3.4 CT-CSD 替换了什么

| Global Sentence-CSD | CT-CSD |
|---|---|
| response 内先求平均 | 不先求平均，保留 token hidden states |
| 一个 global steering vector | 多个 local steering vectors |
| 所有 hidden states 用同一个方向 | 每个 hidden state 路由到最近 cluster 的方向 |
| 先平均再构造方向 | 先聚类再平均构造方向 |
| 推理时只计算一个 alignment | 推理时先 route，再计算对应 local alignment |

---

## 4. Core Method 2：Category-aware CT-CSD

Category-aware CT-CSD 是最终主方法。

### 4.1 方法动机

CT-CSD 虽然保留了 token-level hidden state distribution（token 级隐藏状态分布），但如果所有 harmful tokens 混在一起聚类，KMeans 仍可能把不同 harmful category 混到同一个 cluster。

例如一个 cluster 里可能同时出现：

```text
password / weapon / chemical / exploit
```

这种 cluster 的语义不干净，构造出来的 direction 也难解释。

因此，Category-aware CT-CSD 加入一个简单约束：

> 先按 harmful category 分组，再在每个 category 内部聚类。

注意：category 只用于离线分组，不在推理时训练额外分类器。

---

### 4.2 category 字段选择

针对 HarmBench 数据，优先级建议为：

```text
semantic_category
→ functional_category
→ category
→ unknown
```

针对 JBB（JailBreakBench，越狱基准）或其他数据，如果只有 `category` 字段，就直接使用 `category`。

为了避免字段不一致，构造脚本中建议提供：

```text
--category_key semantic_category
```

如果字段缺失，自动 fallback（回退）到其他字段。

---

### 4.3 离线构造流程

#### hidden state 截取口径

Category-aware CT-CSD 的模型前向输入仍然是完整的：

```text
prompt tokens + response tokens
```

这样 response hidden state 会带有 prompt 条件上下文。但进入后续 token 计数、KMeans 聚类和 cluster 累加的 hidden states 只取 response 对应的部分：

```text
hidden[:, response_start:, :]
```

截取 response 段之后，还要过滤 special token 和空白 token。也就是说：

```text
用于聚类：harmful response 的有效 token hidden states
不用于聚类：prompt token hidden states
不用于聚类：完整序列的所有 token hidden states
```

safe anchor 使用同样口径：只取 safe refusal response 段的有效 token hidden states，并先对每条 safe refusal response 求均值，再参与全局 safe mean 计算。

#### Step 1：按 category 分组

第 $i$ 条 harmful sample 的类别为：

$$
c_i\in\mathcal{C}
$$

类别 $c$ 下的 harmful token hidden states 为：

$$
\mathcal{H}_c
=
\{h_{ij}^{harm}\mid c_i=c\}
$$

---

#### Step 2：类别内部聚类

对每个类别内部做 KMeans：

$$
\mathcal{H}_c
=
C_{c,1}\cup C_{c,2}\cup\cdots\cup C_{c,K_c}
$$

其中 $K_c$ 是类别 $c$ 内的 cluster 数。

为保证消融公平，建议控制 total vector budget（总向量预算）：

```text
M = 总 cluster / vector 数
Stage 2 先从 M ∈ {4, 8, 12, 16} 中选择默认 M*
```

然后按每个 category 的 token 数分配：

$$
K_c
\approx
\max\left(1,
\operatorname{round}
\left(
M\cdot \frac{|\mathcal{H}_c|}{\sum_{c'}|\mathcal{H}_{c'}|}
\right)
\right)
$$

最后做一次调整，使：

$$
\sum_c K_c=M
$$

如果实现上想更简单，第一版也可以使用：

```text
每个 category 固定 K_c = 1 或 2
```

但正式对比 CT-CSD 时，最好让总向量数尽量一致。

---

#### Step 3：构造 category-cluster harmful center

$$
\mu_{c,k}^{harm}
=
\frac{1}{|C_{c,k}|}
\sum_{h\in C_{c,k}}h
$$

---

#### Step 4：构造 category-cluster steering vector

safe side 仍然使用同一个 global safe mean：

$$
\mu^{safe}
=
\mathbb{E}_i[\bar h_i^{safe}]
$$

每个 category-cluster 的方向为：

$$
v_{c,k}
=
\mu_{c,k}^{harm}-\mu^{safe}
$$

归一化：

$$
\hat v_{c,k}
=
\frac{v_{c,k}}{\|v_{c,k}\|_2+\epsilon}
$$

---

### 4.4 推理流程

推理时不需要先判断 prompt 属于哪个类别。

对于当前 hidden state $h$，直接在所有 category-cluster centers 中找最近的一个：

$$
(\hat c,\hat k)
=
\arg\max_{c,k}
\operatorname{cos}
\left(
h,\mu_{c,k}^{harm}
\right)
$$

这一步只是选择候选方向：

$$
\hat v_{\hat c,\hat k}
$$

然后计算 alignment：

$$
s(h)
=
\langle h,\hat v_{\hat c,\hat k}\rangle
$$

通过 threshold 控制是否真正引导：

$$
\alpha(h)
=
\beta[s(h)-\theta]_+
$$

$$
h'
=
h-\alpha(h)\hat v_{\hat c,\hat k}
$$

即：

$$
h'
=
h-
\beta
\left[
\langle h,\hat v_{\hat c,\hat k}\rangle-\theta
\right]_+
\hat v_{\hat c,\hat k}
$$

---

### 4.5 为什么不是所有 token 都会被引导

每个 hidden state 都会被 route（路由）到一个最近 center，但这不表示每个 hidden state 都有害。

路由只回答：

```text
如果这个 hidden state 需要使用某个 harmful direction，应该使用哪一个？
```

真正决定是否引导的是：

$$
s(h)>\theta
$$

分段写法为：

$$
h'
=
\begin{cases}
h, & s(h)\le \theta \\
h-\beta(s(h)-\theta)\hat v_{\hat c,\hat k}, & s(h)>\theta
\end{cases}
$$

因此：

```text
s(h) <= θ：alpha = 0，不改变 hidden state
s(h) > θ：alpha > 0，才执行 steering
```

这保留了原项目中 adaptive steering（自适应引导）的最小干预思想。

---

### 4.6 Category-aware CT-CSD 替换了什么

| CT-CSD | Category-aware CT-CSD |
|---|---|
| 所有 harmful tokens 混合聚类 | 先按 category 分组 |
| cluster 可能混合多个风险类型 | cluster 更语义一致 |
| 方向解释性较弱 | 方向可对应 category-cluster |
| 推理时找全局 cluster | 推理时找所有 category-cluster 中最近 center |
| 不使用类别元数据 | 使用类别元数据作为离线聚类约束 |

---

## 5. MIL Token Probe 替换模块

MIL token probe（Multiple Instance Learning token probe，多实例学习 token 探针）是可替换模块，不单独作为主方法。

### 5.1 模块定位

它替换的是 token selection（token 选择）步骤。

原始 CT-CSD / Category-aware CT-CSD：

```text
所有 harmful response tokens 都进入聚类
```

加入 MIL token probe 后：

```text
先训练一个弱监督 token probe
→ 给每个 token hidden state 打 harmfulness score
→ 只保留高分 harmful tokens
→ 再进入 CT-CSD 或 Category-aware CT-CSD 聚类
```

---

### 5.2 为什么需要 MIL token probe

harmful response 中不是每个 token 都真正表达 harmful semantic signal。

例如：

```text
Here is a detailed guide to ...
```

这里的 `Here`、`is`、`a`、标点和换行可能只是模板或功能 token。如果它们大量进入聚类，cluster 可能学到：

```text
句首模板
列表编号
换行格式
标点
拒答/回答风格
```

而不是有害语义。

MIL token probe 的作用是：

> 用 response-level label（回答级标签）弱监督地估计 token-level harmfulness score（token 级有害性分数），从而筛掉低贡献 token。

---

### 5.3 训练数据组织

每条 response 看作一个 bag（包），response 中的 token hidden states 看作 instances（实例）。

```text
bag = 一条 response
instance = 一个 token hidden state
bag label = response-level harmful/safe label
```

标签定义：

```text
harmful response: y_i = 1
safe response:    y_i = 0
```

可以使用的数据：

```text
prompt + harmful response
prompt + safe refusal
可选：benign prompt + benign response
```

第一版为了和当前项目最小对齐，可以只使用 harmful responses 和 safe refusals。

---

### 5.4 probe 形式

使用一个轻量 linear probe（线性探针）：

$$
s_{ij}
=
w^\top h_{ij}+b
$$

其中：

| 符号 | 含义 |
|---|---|
| $h_{ij}$ | 第 $i$ 条 response 的第 $j$ 个 token hidden state |
| $s_{ij}$ | token-level harmful logit（token 级有害 logit） |
| $w,b$ | 线性 probe 参数 |

---

### 5.5 top-q pooling

由于没有人工 token-level label（token 级标签），不能直接对每个 token 计算 loss（损失）。

因此用 top-q pooling（前 q 个分数池化）把 token scores 聚合成 response-level score：

$$
S_i
=
\frac{1}{q_i}
\sum_{j\in \operatorname{TopQ}(s_i)}
s_{ij}
$$

其中 $q_i$ 可以设置为：

```text
q_i = max(1, ceil(r * T_i))
r = 0.1 或 0.2
```

也可以固定：

```text
q = 8
```

第一版建议：

```text
q_i = max(1, ceil(0.1 * T_i))
```

直觉是：一条 harmful response 不需要所有 token 都表达有害语义，只要少数关键 token 分数高，就足以使 response 被判为 harmful。

---

### 5.6 训练目标

使用 response-level label 训练：

$$
\mathcal L
=
\operatorname{BCEWithLogitsLoss}(S_i,y_i)
$$

训练完成后，对每个 token 计算 harmfulness probability（有害性概率）：

$$
p_{ij}
=
\sigma(s_{ij})
$$

---

### 5.7 替换到 CT-CSD

原始 CT-CSD 使用：

$$
\mathcal H
=
\{h_{ij}^{harm}\}
$$

Probe-CT-CSD 使用：

$$
\mathcal H^{probe}
=
\{h_{ij}^{harm}\mid p_{ij}\ge \tau_p\}
$$

然后再执行普通 CT-CSD 的聚类和向量构造。

建议阈值：

```text
τ_p = 0.5, 0.7, 0.9
```

第一版推荐主结果使用：

```text
τ_p = 0.7
```

---

### 5.8 替换到 Category-aware CT-CSD

原始 Category-aware CT-CSD：

$$
\mathcal H_c
=
\{h_{ij}^{harm}\mid c_i=c\}
$$

Probe-Category-aware CT-CSD：

$$
\mathcal H_c^{probe}
=
\{h_{ij}^{harm}\mid c_i=c,\ p_{ij}\ge \tau_p\}
$$

然后在每个 category 内部聚类。

---

### 5.9 MIL 替换部分总结

| 原方法 | MIL 替换后 |
|---|---|
| 所有 harmful response tokens 进入聚类 | 只有高 harmfulness score tokens 进入聚类 |
| 容易聚到模板词、标点、换行 | 更偏向 harmful semantic signal |
| 没有 token selection | 增加弱监督 token selection |
| 聚类噪声较大 | 聚类更干净 |
| 推理公式不变 | 推理公式仍然只做 routing + threshold steering |

重要的是：

> MIL token probe 只影响离线构造向量时选哪些 token，不改变推理时的 steering 公式。

---

## 6. 最终方法定义

### 6.1 主方法：Category-aware CT-CSD

主方法流程：

```text
1. 提取 harmful / safe response token hidden states
2. 计算 sample-balanced global safe mean
3. 对 harmful tokens 按 category 分组
4. 每个 category 内部做 KMeans 聚类
5. 每个 category-cluster 构造一个 local steering vector
6. 推理时 hidden state route 到最近 harmful center
7. 只有 alignment 超过 θ 才执行 steering
```

---

### 6.2 增强方法：Probe-Category-aware CT-CSD

增强方法只替换第 3 步之前的 token selection：

```text
1. 训练 MIL token probe
2. 对 harmful tokens 计算 harmfulness score
3. 只保留 p_ij >= τ_p 的 token
4. 再按 category 分组并聚类
```

推理阶段完全不变。

---

### 6.3 最终推理公式

路由：

$$
(\hat c,\hat k)
=
\arg\max_{c,k}
\operatorname{cos}
\left(
h,\mu_{c,k}^{harm}
\right)
$$

计算 alignment：

$$
s(h)
=
\langle h,\hat v_{\hat c,\hat k}\rangle
$$

计算引导强度：

$$
\alpha(h)
=
\beta[s(h)-\theta]_+
$$

更新 hidden state：

$$
h'
=
h-
\alpha(h)
\hat v_{\hat c,\hat k}
$$

合并为：

$$
h'
=
h-
\beta
\left[
\langle h,\hat v_{\hat c,\hat k}\rangle-\theta
\right]_+
\hat v_{\hat c,\hat k}
$$

---

## 7. 离线构造文件格式

建议新向量文件保存为 `ct_csd_bank.pt`，格式如下：

```python
{
    "format": "category_ct_csd_v1",
    "model_family": "llada",
    "target_layer": 31,
    "safe_anchor_type": "sample_balanced_global_safe_mean",

    "safe_mean": Tensor[D],

    "centers": Tensor[M, D],
    "centers_unit": Tensor[M, D],

    "vectors": Tensor[M, D],
    "vectors_unit": Tensor[M, D],

    "center_categories": list[str],
    "center_category_ids": Tensor[M],
    "cluster_ids": Tensor[M],
    "cluster_sizes": Tensor[M],

    "categories": list[str],

    "config": {
        "method": "category_ct_csd",
        "num_total_clusters": M,
        "cluster_feature": "l2_normalized_hidden",
        "category_key": "semantic_category",
        "max_response_len": 128,
        "exclude_special_tokens": True,
        "exclude_punctuation": True
    },

    "mil": {
        "enabled": False,
        "probe_path": None,
        "probe_threshold": None,
        "top_q_ratio": None
    }
}
```

如果使用 MIL token probe：

```python
"mil": {
    "enabled": True,
    "probe_path": ".../mil_token_probe.pt",
    "probe_threshold": 0.7,
    "top_q_ratio": 0.1
}
```

说明：

```text
centers 用于 routing
vectors_unit 用于 steering
safe_mean 只用于记录和复现，不在推理时重新计算
cluster_sizes 用于诊断，不参与推理公式
```

---

## 8. 代码实现计划

### 8.1 新增文件

第一版建议尽量少加文件：

```text
utils/ct_csd_bank.py
utils/make_ct_csd_llada.py
utils/train_mil_token_probe_llada.py
```

其中：

| 文件 | 作用 |
|---|---|
| `utils/ct_csd_bank.py` | 加载 bank，完成 routing、alignment、steering |
| `utils/make_ct_csd_llada.py` | 构造 CT-CSD / Category-aware CT-CSD / Probe variants |
| `utils/train_mil_token_probe_llada.py` | 训练 MIL token probe |

Dream 版本后续迁移：

```text
utils/make_ct_csd_dream.py
utils/train_mil_token_probe_dream.py
```

---

### 8.2 `utils/make_ct_csd_llada.py`

核心参数：

```bash
MSTAR=8  # 示例；实际填写 Stage 2 选出的 M*

python utils/make_ct_csd_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json data/harmbench_csd_train.json \
  --refusals_txt utils/refusals.txt \
  --output_dir "outputs/category_ct_csd_llada_m${MSTAR}" \
  --target_layer 31 \
  --max_response_len 128 \
  --method category_ct_csd \
  --num_total_clusters "${MSTAR}" \
  --category_key semantic_category
```

支持的 `--method`：

```text
global_sentence_csd      # 兼容旧 baseline，可选
ct_csd                   # 全局 token 聚类
random_k_csd             # 随机分组对照
category_ct_csd          # 类别感知聚类
probe_ct_csd             # MIL 替换 CT-CSD token selection
probe_category_ct_csd    # MIL 替换 Category-aware CT-CSD token selection
```

重要实现变化：

当前 `make_csd_llada.py` 中的：

```python
resp = h[0, response_start:, :]
out[i] = resp.mean(dim=0).float().cpu()
```

需要替换为 target layer token states 提取：

```python
resp = h[0, response_start:, :]  # [resp_len, D]
return resp.float().cpu()
```

第一版不要同时保存所有层，只保存：

```text
target_layer = 31
```

---

### 8.3 `utils/ct_csd_bank.py`

建议提供一个简单类：

```python
class CTCSDBank:
    @classmethod
    def load(cls, path, device, dtype=torch.float32):
        ...

    def route(self, hidden):
        ...

    def alignment(self, hidden):
        ...

    def steer(self, hidden, beta, theta):
        ...
```

核心逻辑：

```python
# hidden: [N, D]
h_unit = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-8)
sim = h_unit @ centers_unit.T
idx = sim.argmax(dim=-1)

v = vectors_unit[idx]
score = (hidden * v).sum(dim=-1)
alpha = beta * (score - theta).clamp(min=0.0)
hidden_new = hidden - alpha.unsqueeze(-1) * v
```

注意：

```text
route 使用 centers_unit
steering 使用 vectors_unit
alignment threshold 仍然使用全局 θ
不使用 per-cluster threshold
```

---

### 8.4 修改 `eval_llada_steering.py`

当前加载逻辑：

```python
vectors = torch.load(steering_vector_path, weights_only=True)
key = f'layer_{target_layer}'
self.steering_vector = vectors[key]
```

建议改成兼容两种格式：

```python
obj = torch.load(steering_vector_path, weights_only=True)

if isinstance(obj, dict) and obj.get("format") == "category_ct_csd_v1":
    self.steering_bank = CTCSDBank.from_state_dict(obj, device=self.device)
    self.steering_vector = None
else:
    self.steering_vector = obj[f"layer_{target_layer}"]
    self.steering_bank = None
```

---

### 8.5 替换 `_per_token_alignment`

当前：

```python
sv = self.steering_vector.to(block_hidden.device, dtype=block_hidden.dtype)
sv_unit = sv / (sv.norm() + 1e-8)
return (block_hidden * sv_unit).sum(dim=-1)
```

新逻辑：

```python
if self.steering_bank is not None:
    return self.steering_bank.alignment(block_hidden)
else:
    # 保持原始 global vector 路径
    ...
```

`steering_bank.alignment(block_hidden)` 内部会对每个 token：

```text
route 到最近 harmful center
取对应 local vector
计算 local alignment
```

---

### 8.6 替换 `_build_adaptive_steering_hook`

当前所有 mask positions（掩码位置）都使用同一个 `sv_unit`。

新逻辑：

```python
if self.steering_bank is not None:
    hidden[_mask_index] = self.steering_bank.steer(
        masked_h,
        beta=_beta,
        theta=_theta,
    ).to(hidden.dtype)
else:
    # 保持原始 global vector 路径
    ...
```

仍然保持：

```text
alpha = beta * clamp(score - theta, min=0)
```

所以不是所有 token 都会被真正修改。

---

### 8.7 替换 DIJA 分支 alignment

当前 DIJA 分支中有类似逻辑：

```python
alignment = (full_hidden * sv_unit).sum(dim=-1)
harmful_mask = (alignment > self.alignment_threshold) & original_mask_index
```

新逻辑：

```python
if self.steering_bank is not None:
    alignment = self.steering_bank.alignment(full_hidden)
else:
    alignment = (full_hidden * sv_unit).sum(dim=-1)

harmful_mask = (alignment > self.alignment_threshold) & original_mask_index
```

---

### 8.8 Dream 迁移

Dream 的代码结构与 LLaDA 类似，主要差异在 transformer layer 获取方式。

因此建议：

```text
先完成 LLaDA 版本
确认 CTCSDBank 接口稳定
再迁移 Dream
```

Dream 迁移时只需要替换：

```text
1. CSD 构造脚本中的 block 获取逻辑
2. eval_dream_steering.py 中的 bank 加载、alignment 和 hook 路径
```

---

## 9. 实验设计

### 9.1 主实验矩阵

| 方法 | 目的 |
|---|---|
| Global Sentence-CSD | 当前 baseline |
| CT-CSD | 验证“先聚类再平均”是否有效 |
| Random-K-CSD | 排除“只是向量数量变多”的影响 |
| Category-aware CT-CSD | 验证 category 信息是否有效 |
| Probe-CT-CSD | 验证 MIL token selection 是否有效 |
| Probe-Category-aware CT-CSD | 验证最终增强版是否最好 |

---

### 9.2 不作为主线的可选诊断

| 方法 | 作用 |
|---|---|
| Token-weighted Global CSD | 只检查 token 等权平均与 sentence 等权平均是否有差异，不作为核心贡献 |

说明：

```text
Token-weighted Global CSD 与原始 Sentence-CSD 只是在样本权重上略有差异。
如果 response 长度接近，它们几乎等价。
所以它不进入主线。
```

---

### 9.3 固定参数

为了让实验只比较向量构造方式，第一轮固定：

```text
target_layer = 31
steering_overshoot / beta = 当前项目最优或默认值，例如 1.0
alignment_threshold / theta = 当前项目最优或默认值，例如 0.0
initial_steering_ratio = 0.1
max_refinement_iters = 5
sampling_steps = 128
mask_length = 128
block_size = 128
remasking = low_confidence
```

不要第一轮同时扫太多 `beta` 和 `theta`，否则会掩盖方法本身的差异。

---

### 9.4 变化参数

第一轮只变化：

```text
method ∈ {
  global_sentence_csd,
  ct_csd,
  random_k_csd,
  category_ct_csd,
  probe_ct_csd,
  probe_category_ct_csd
}

num_total_clusters M 先在 Stage 1/2 中从 {4, 8, 12, 16} 选择
Stage 3 及之后默认继承 Stage 2 选出的 M*
probe_threshold τ_p ∈ {0.5, 0.7, 0.9}
```

推荐最小运行矩阵：

```text
Global Sentence-CSD
CT-CSD, M=M*
Random-K-CSD, M=M*
Category-aware CT-CSD, M=M*
Probe-CT-CSD, M=M*, τ_p=0.7
Probe-Category-aware CT-CSD, M=M*, τ_p=0.7
```

这里的 `M*` 不是预设值，而是 Stage 2 根据 `ASR`、`unsafe_count`、routing / activation 诊断和 cluster 稳定性选出的默认簇数。`M=16` 只作为候选点保留，不能直接当成后续阶段的默认结论。

---

### 9.5 评测指标

安全指标：

```text
ASR（Attack Success Rate，攻击成功率）↓
unsafe_count（不安全样本数）↓
```

质量指标：

```text
refusal rate（拒答率）
over-refusal（过度拒答）
平均输出长度
空回复比例
```

干预指标：

```text
steering activation rate（引导触发率）
平均被 steering 的 token 数
平均 remask token 数
平均 refinement iterations（重生成迭代次数）
```

效率指标：

```text
推理耗时
额外显存
bank route 耗时
```

---

## 10. 消融实验

### 10.1 Global Sentence-CSD vs CT-CSD

验证：

```text
先平均
vs
先聚类再平均
```

如果 CT-CSD 更好，说明 token hidden state distribution 中确实存在有用的局部结构。

---

### 10.2 CT-CSD vs Random-K-CSD

Random-K-CSD 做法：

```text
把 harmful tokens 随机分成 K 组
每组构造一个 vector
推理时仍然 route 到最近 random group center
```

如果：

```text
CT-CSD > Random-K-CSD
```

说明 KMeans 聚类真的捕捉到了有效结构，而不是“向量数量变多”导致的偶然提升。

注意：这个对照必须在相同 `M` 下比较。Stage 2 的默认做法是先从 CT-CSD 的簇数消融中选出 `M*`，再比较：

```text
CT-CSD, M=M*
Random-K-CSD, M=M*
```

如果 Random-K-CSD 和 CT-CSD 指标接近，需要增加固定随机种子的重复实验，例如 `seed = 42, 43, 44`，避免把随机划分波动误判成方法差异。

---

### 10.3 CT-CSD vs Category-aware CT-CSD

验证：

```text
category 信息是否让 cluster 更语义一致
```

如果 Category-aware CT-CSD 更好，说明 harmful category 对 hidden state direction 构造有帮助。

---

### 10.4 CT-CSD vs Probe-CT-CSD

验证：

```text
MIL token selection 是否能清理全局聚类噪声
```

---

### 10.5 Category-aware CT-CSD vs Probe-Category-aware CT-CSD

验证：

```text
在类别内聚类基础上，MIL token probe 是否进一步提升效果
```

这是最关键的最终对比之一。

---

### 10.6 K / M 值消融

第一轮推荐：

```text
M = 4, 8, 12, 16
```

观察：

```text
M 太小：局部方向不足
M 太大：cluster 变小，方向不稳定，推理路由噪声增加
```

选择默认簇数 `M*` 时使用以下规则：

```text
1. 主指标优先：优先选择 unsafe_count / ASR 更低的 M。
2. 平局容忍：如果 unsafe_count 只差 1 个样本以内，认为安全指标近似持平。
3. 诊断过滤：剔除出现明显异常的 M，例如空簇、极小簇占比过高、route_count 极端集中、activation_rate 接近 0 或异常过高。
4. 简洁优先：在主指标近似持平且诊断都可接受时，选择更小的 M，降低推理路由成本和后续 category 分配复杂度。
5. 保守回退：如果所有候选都和 Global Sentence-CSD 持平，选择“诊断最稳定且最小”的 M，而不是默认回到 16。
```

如果后续论文消融需要更完整曲线，可以再补充 `M = 32`。但 Stage 2 的执行目标是选一个可继承到 Stage 3/4 的默认 `M*`，不是一开始扩大成大规模超参搜索。

---

### 10.7 MIL threshold 消融

推荐：

```text
τ_p = 0.5, 0.7, 0.9
```

观察：

```text
τ_p 太低：仍有大量模板 token 进入聚类
τ_p 太高：保留 token 太少，cluster 不稳定
```

---

## 11. 诊断分析

### 11.1 cluster 高频 token 检查

对每个 cluster 输出最近 center 的 token 文本和高频 token。

检查 cluster 是否主要是：

```text
真正 harmful semantic tokens
还是
Here / is / first / step / punctuation / newline / numbering
```

---

### 11.2 cluster category 分布

对于 CT-CSD，统计每个 cluster 里的 category 分布。

如果一个 cluster 混合多个 category，说明 category-aware clustering 有必要。

---

### 11.3 MIL 高分 token 检查

输出 MIL token probe 最高分 token：

```text
sample_id
category
token_text
p_ij
```

人工检查高分 token 是否合理。

---

### 11.4 route histogram

推理时统计：

```text
每个 category-cluster 被 route 的次数
每个 category-cluster 真正触发 steering 的次数
```

如果某些 cluster 永远不用，可能说明：

```text
cluster 不稳定
center 不在推理分布附近
M 设置过大
```

---

### 11.5 steering activation rate

必须区分：

```text
route_count：被路由到该 center 的 token 数
active_count：s(h) > θ，真正被 steering 的 token 数
```

因为：

> 每个 token 都会 route，但不是每个 token 都会 steering。

---

### 11.6 failure case 分析

失败类型：

| 失败类型 | 可能原因 | 对应处理 |
|---|---|---|
| ASR 没降 | 向量方向不有效；threshold 太高 | 检查 alignment 分布，降低 theta 或换 layer |
| 输出质量差 | threshold 太低；steering 太强 | 降低 beta，提高 theta |
| 过度拒答 | safe mean 过于拒答模板化 | 增加 safe responses 多样性 |
| cluster 学到模板词 | harmful tokens 噪声大 | 使用 MIL token probe 或过滤模板 token |
| Category-aware 无提升 | category 太粗或数据不均衡 | 合并小类，检查 category 分布 |
| Probe 无提升 | probe 没学到有效 token | 检查高分 token，调整 top-q 和 threshold |

---

## 12. 风险与控制

### 12.1 safe mean 可能学到拒答风格

当前 safe response 主要来自 refusal templates（拒答模板），global safe mean 可能包含“拒答风格”。

第一版仍然保留 global safe mean，因为它最简单、最可控。但需要在文档里承认：

```text
safe mean 可能不完全等价于安全语义，也可能包含 refusal style。
```

后续扩展可以增加：

```text
更丰富的 safe response
同 prompt 的安全替代回答
非拒答式安全回答
```

但不进入第一版主方法。

---

### 12.2 category 数据不均衡

如果某些 category 样本太少，类别内聚类会不稳定。

控制方式：

```text
小类别合并到 other
每类至少保留一定 token 数
K_c 最小为 1，过小类别不再细分
记录 cluster_sizes
```

---

### 12.3 cluster 太小导致方向不稳定

控制方式：

```text
限制 M 不要太大
丢弃或合并小 cluster
输出 cluster size 诊断
```

第一版建议：

```text
M 不超过 32
```

---

### 12.4 MIL token probe 可能学不到真正 harmful token

因为 probe 只有 response-level label，没有 token-level annotation（token 级标注）。

控制方式：

```text
MIL 只作为替换模块
必须保留 no-probe 对照
检查高分 token
不要直接把 MIL 作为主方法前提
```

---

### 12.5 推理阶段路由错误

每个 hidden state 都会被 route 到最近 harmful center，即使它是安全 token。

控制方式：

```text
保留 alignment threshold θ
只有 s(h) > θ 才 steering
统计 route_count 和 active_count
```

这也是为什么不需要额外 gate（门控）模块。

---

## 13. 论文/报告叙事

### 13.1 方法动机写法

可以写成：

> Existing CSD methods construct a single global steering direction from sentence-level averaged hidden states. However, in diffusion language models, the sentence representation is obtained by averaging token hidden states, which may prematurely collapse heterogeneous token-level harmful signals. We propose CT-CSD, which delays the averaging operation: harmful token hidden states are first clustered, and each cluster is then contrasted against a global safe anchor to construct a local steering direction.

中文版本：

> 现有 CSD 方法通常先将一条回答中的 token hidden states 平均成句子表示，再构造单一全局引导方向。但对于扩散语言模型来说，这种过早平均可能会抹平不同 token 的局部有害信号。我们提出 CT-CSD：推迟平均操作，先对 harmful response 中的 token hidden states 聚类，再在每个 cluster 内部求均值，并与全局安全均值构造局部对比引导方向。

---

### 13.2 贡献点

建议贡献写成四点：

```text
贡献 1：
提出 CT-CSD，将 CSD 构造从“先平均再构造单方向”改为“先聚类再构造多局部方向”。

贡献 2：
提出 Category-aware CT-CSD，利用 harmful category 作为离线聚类约束，使局部方向更语义一致。

贡献 3：
引入 MIL token probe 作为 token selection 替换模块，在没有 token-level annotation 的情况下过滤模板 token 和低贡献 token。

贡献 4：
推理阶段保持原始 threshold-gated steering 公式，只将单一 global vector 替换为 route 得到的 local vector，保证方法简单可解释。
```

---

### 13.3 实验验证逻辑

```text
Global Sentence-CSD
↓
CT-CSD
验证：先聚类再平均是否有效

CT-CSD
↓
Random-K-CSD
验证：提升是否来自真实聚类结构，而不是向量数量增加

CT-CSD
↓
Category-aware CT-CSD
验证：category 信息是否有效

Category-aware CT-CSD
↓
Probe-Category-aware CT-CSD
验证：MIL token selection 是否有效
```

---

## 14. Stage 拆分落地计划

本节把前面的方案拆成可独立验收的 stage。拆分原则是：

```text
每个 stage 只引入一个新增变量
每个 stage 都保留上一阶段可复现结果
每个 stage 都有明确代码交付物、实验对照和退出标准
```

总依赖关系为：

```text
Stage 0：冻结 baseline 与数据口径
↓
Stage 1：CT-CSD bank 最小闭环
↓
Stage 2：默认簇数选择与 Random-K-CSD 对照
↓
Stage 3：Category-aware CT-CSD 主方法
↓
Stage 4：MIL token probe 替换模块
↓
Stage 5：Dream 迁移
↓
Stage 6：消融、诊断与论文叙事固化
```

| Stage | 核心问题 | 新增变量 | 主要产物 |
|---|---|---|---|
| Stage 0 | 当前 baseline 是否可复现 | 无 | baseline 配置、数据字段检查、评测口径 |
| Stage 1 | 先聚类再平均是否有效 | CT-CSD local vectors | `ct_csd_bank.py`、`make_ct_csd_llada.py`、bank 推理路径 |
| Stage 2 | 默认簇数是多少，提升是否只是因为向量数量变多 | M 选择、random grouping | `stage2_m_selection.md`、`random_k_csd` 构造分支 |
| Stage 3 | category 信息是否有帮助 | category-aware clustering | `category_ct_csd` 构造分支 |
| Stage 4 | MIL token selection 是否有帮助 | MIL token probe | `train_mil_token_probe_llada.py`、probe variants |
| Stage 5 | 方法是否能迁移到 Dream | Dream 适配 | Dream 构造脚本与推理路径 |
| Stage 6 | 结果是否能支撑论文叙事 | 无新方法 | 消融表、诊断图、失败分析、报告文本 |

---

### Stage 0：冻结 baseline 与数据口径

目标：先确认原始 Global Sentence-CSD 可以稳定复现，避免后续把数据、参数或评测脚本差异误当成方法收益。

对应内容：

```text
第 1 节：当前 Global Sentence-CSD 构造与推理方式
第 8.4-8.7 节：eval_llada_steering.py 中的现有 global vector 路径
第 9.3 节：固定参数
第 17 节：定稿检查清单
```

实现范围：

```text
不新增方法
不改推理公式
只记录当前可跑通的 baseline 命令、参数和输出目录
检查 harmful_json 中 category 字段是否存在
检查 safe refusal 数据与当前 make_csd_llada.py 的口径
```

固定项：

```text
target_layer = 31
sampling_steps = 128
mask_length = 128
block_size = 128
remasking = low_confidence
initial_steering_ratio = 0.1
max_refinement_iters = 5
```

验收标准：

```text
Global Sentence-CSD 可以完整构造 steering_vectors.pt
eval_llada_steering.py 可以用旧格式 steering_vectors.pt 完整评测
记录 ASR、unsafe_count、refusal rate、平均输出长度、耗时
后续所有 stage 都使用同一套评测配置作为对照
```

输出物：

```text
outputs/global_sentence_csd_llada/
outputs/jbb_dija_global_sentence_csd/
baseline_metrics.md 或实验日志
```

---

### Stage 1：CT-CSD bank 最小闭环

目标：先只验证“先聚类再平均”是否优于“先平均再构造单方向”，跑通多向量 bank、hard routing 和 threshold-gated steering 的完整路径。

对应内容：

```text
第 3 节：CT-CSD
第 7 节：离线构造文件格式
第 8.1-8.7 节：LLaDA 代码实现计划
第 10.1 节：Global Sentence-CSD vs CT-CSD
第 11.4-11.5 节：route histogram 与 steering activation rate
```

实现范围：

```text
新增 utils/ct_csd_bank.py
新增 utils/make_ct_csd_llada.py
在 eval_llada_steering.py 中兼容加载 ct_csd_bank.pt
保留旧 steering_vectors.pt 加载路径
实现 method=ct_csd
```

本阶段只做：

```text
收集 harmful response token hidden states
过滤 special / padding / 空白 token
KMeans 聚类 harmful token hidden states
构造 global safe mean
构造 local steering vectors
推理时 route 到最近 center
仍使用全局 beta 和 theta
```

本阶段不做：

```text
不使用 category 字段
不训练 MIL token probe
不做 safe token clustering
不做 soft routing
不做 per-cluster threshold
不迁移 Dream
```

推荐参数：

```text
method = ct_csd
num_total_clusters M = 16
target_layer = 31
cluster_feature = l2_normalized_hidden
```

最小实验：

```text
Global Sentence-CSD
CT-CSD, M=16
```

诊断必须记录：

```text
每个 cluster 的 cluster_size
每个 cluster 的 route_count
每个 cluster 的 active_count
整体 steering activation rate
推理阶段 bank route 耗时
```

验收标准：

```text
ct_csd_bank.pt 可以被 eval_llada_steering.py 正常加载
旧 global vector 路径仍然可用
CT-CSD 推理中每个 token 都能 route，但只有 s(h) > theta 时被 steering
CT-CSD 至少完成一组与 Global Sentence-CSD 的同配置对比
```

输出物：

```text
utils/ct_csd_bank.py
utils/make_ct_csd_llada.py
outputs/ct_csd_llada_m16/ct_csd_bank.pt
outputs/jbb_dija_ct_csd_m16/
stage1_ct_csd_metrics.md
```

---

### Stage 2：默认簇数选择与 Random-K-CSD 对照

目标：先用 CT-CSD 簇数消融选出后续阶段默认簇数 `M*`，再用同一 `M*` 下的 Random-K-CSD 排除“只是向量数量变多所以效果更好”的解释。

对应内容：

```text
第 9.1 节：主实验矩阵
第 10.2 节：CT-CSD vs Random-K-CSD
第 10.6 节：K / M 值消融
```

实现范围：

```text
在 utils/make_ct_csd_llada.py 中新增 method=random_k_csd
支持 --num_total_clusters 传入 Stage 2 选出的 M*
复用 Stage 1 的 ct_csd_bank.pt 格式
复用 Stage 1 的 CTCSDBank 推理逻辑
在 bank config 中记录 method、num_total_clusters、seed 和 grouping 策略
```

本阶段分两步执行。

第一步：完成 CT-CSD 簇数消融，选出 `M*`。

```text
候选 M = 4, 8, 12, 16
当前 M=16 已完成
M=4 / 8 / 12 按同一口径补齐 bank、生成、judge 和 diagnostics
```

选择 `M*` 的口径：

```text
主指标：unsafe_count / ASR 越低越好
辅助诊断：cluster_size 分布、route_count 分布、active_count 分布、activation_rate、route_time_sec
平局规则：unsafe_count 相差 <= 1 时，优先选择诊断更稳定且更小的 M
异常剔除：出现空簇、极小簇过多、route 极端集中、activation_rate 接近 0 或异常过高时，不作为默认 M*
```

第二步：在固定 `M*` 后做 Random-K-CSD 对照。

```text
CT-CSD：KMeans 分组 harmful tokens
Random-K-CSD：随机分组 harmful tokens
```

必须保持一致：

```text
vector 数量 M 一致，均为 M*
safe mean 构造方式一致
routing 方式一致
steering 公式一致
beta / theta / target_layer 一致
评测数据与采样参数一致
random_seed 固定；如结果接近，再补 seed 重复
```

推荐参数：

```text
method = random_k_csd
num_total_clusters M = M*
random_seed = 42
可选重复 random_seed = 43, 44
```

最小实验：

```text
CT-CSD, M=M*
Random-K-CSD, M=M*, seed=42
```

如果 `Random-K-CSD, seed=42` 与 CT-CSD 的 `unsafe_count` 相差不超过 1，需要补跑：

```text
Random-K-CSD, M=M*, seed=43
Random-K-CSD, M=M*, seed=44
```

如果 Random-K 多 seed 平均仍接近 CT-CSD，则 Stage 2 结论应写成“暂不能证明 KMeans 结构优于随机分组”，不能在论文叙事中强 claim clustering 贡献。

如果 `M*` 的选择本身不稳定，补充同预算矩阵：

```text
CT-CSD, M=8 / 12 / 16
Random-K-CSD, M=8 / 12 / 16
```

`M=4` 只在它被选为 `M*` 或者表现异常好时进入 Random-K 矩阵，避免不必要扩大实验量。

诊断必须记录：

```text
每个 M 的 unsafe_count / ASR / total_samples
每个 M 的 cluster_sizes 摘要：min、p10、median、max
每个 M 的 route_count / active_count / activation_rate
每个 M 的 route_time_sec
Random-K 每个 seed 的同样诊断
最终 M* 的选择理由
```

验收标准：

```text
完成 CT-CSD 的 M=4/8/12/16 同口径簇数消融，或明确记录未完成点位的原因
基于指标和诊断选出 M*，且后续 Stage 3/4 默认继承 M*
Random-K-CSD 可以生成与 CT-CSD 同结构的 ct_csd_bank.pt
Random-K-CSD 可以走同一套 eval_llada_steering.py 推理路径
同一个 M* 下完成 CT-CSD vs Random-K-CSD 对照
只有当 CT-CSD 明确优于 Random-K-CSD，才声称 KMeans 聚类结构本身有贡献
```

输出物：

```text
outputs/ct_csd_llada_m{M}/ct_csd_bank.pt
outputs/jbb_dija_ct_csd_m{M}/
outputs/random_k_csd_llada_m{MSTAR}_seed42/ct_csd_bank.pt
outputs/jbb_dija_random_k_csd_m{MSTAR}_seed42/
stage2_m_selection.md
stage2_random_k_metrics.md
```

---

### Stage 3：Category-aware CT-CSD 主方法

目标：验证 harmful category 是否能让 cluster 更语义一致，并形成第一版主方法。

对应内容：

```text
第 4 节：Category-aware CT-CSD
第 6.1 节：主方法定义
第 7 节：bank 文件格式
第 10.3 节：CT-CSD vs Category-aware CT-CSD
第 11.2 节：cluster category 分布
第 12.2 节：category 数据不均衡
```

实现范围：

```text
在 utils/make_ct_csd_llada.py 中新增 method=category_ct_csd
支持 --category_key semantic_category
实现 category 字段 fallback
按 category 分组后在类别内部聚类
按 token 数分配每类 K_c，并保证总向量数为 M
在 bank 中保存 center_categories、center_category_ids、cluster_ids、categories
```

本阶段只新增：

```text
category-aware offline clustering
category-cluster metadata
category 分布诊断
```

本阶段不做：

```text
不训练 prompt category classifier
不在推理时预测 prompt 类别
不改变 routing 公式
不改变 threshold-gated steering 公式
不加入 MIL token probe
```

推荐参数：

```text
method = category_ct_csd
category_key = semantic_category
num_total_clusters M = M*
每个 category 至少 K_c = 1
过小 category 合并到 other 或只保留 1 个 cluster
```

最小实验：

```text
CT-CSD, M=M*
Category-aware CT-CSD, M=M*
```

诊断必须记录：

```text
每个 category 的 token 数
每个 category 分到的 K_c
每个 category-cluster 的 cluster_size
CT-CSD cluster 的 category 混合程度
Category-aware CT-CSD 的 route_count / active_count
```

验收标准：

```text
category_ct_csd 生成的 bank 与 Stage 1 bank 加载接口一致
推理阶段不需要输入 prompt category
总 vector 数 M 与 CT-CSD 对照保持一致
完成 CT-CSD vs Category-aware CT-CSD 的同配置对比
```

输出物：

```text
outputs/category_ct_csd_llada_m{MSTAR}/ct_csd_bank.pt
outputs/jbb_dija_category_ct_csd_m{MSTAR}/
stage3_category_ct_csd_metrics.md
cluster_category_distribution.md
```

---

### Stage 4：MIL token probe 替换模块

目标：验证弱监督 token selection 是否能减少模板 token、标点、换行等噪声进入聚类。

对应内容：

```text
第 5 节：MIL Token Probe 替换模块
第 6.2 节：Probe-Category-aware CT-CSD
第 10.4 节：CT-CSD vs Probe-CT-CSD
第 10.5 节：Category-aware CT-CSD vs Probe-Category-aware CT-CSD
第 10.7 节：MIL threshold 消融
第 11.3 节：MIL 高分 token 检查
第 12.4 节：MIL token probe 风险控制
```

实现范围：

```text
新增 utils/train_mil_token_probe_llada.py
在 utils/make_ct_csd_llada.py 中支持 method=probe_ct_csd
在 utils/make_ct_csd_llada.py 中支持 method=probe_category_ct_csd
bank 的 mil 字段记录 probe_path、probe_threshold、top_q_ratio
```

训练设置：

```text
bag = 一条 response
instance = token hidden state
bag label = harmful / safe
probe = linear probe
pooling = top-q pooling
loss = BCEWithLogitsLoss
```

推荐参数：

```text
top_q_ratio = 0.1
probe_threshold tau_p = 0.7
target_layer = 31
num_total_clusters M = M*
```

本阶段只影响离线 token selection：

```text
原始 CT-CSD：所有 harmful tokens 进入聚类
Probe-CT-CSD：p_ij >= tau_p 的 harmful tokens 进入聚类

原始 Category-aware CT-CSD：每类所有 harmful tokens 进入聚类
Probe-Category-aware CT-CSD：每类 p_ij >= tau_p 的 harmful tokens 进入聚类
```

本阶段不做：

```text
不把 MIL 作为唯一主方法
不改变推理公式
不在推理时运行 probe
不取消 no-probe 对照
```

最小实验：

```text
CT-CSD, M=M*
Probe-CT-CSD, M=M*, tau_p=0.7
Category-aware CT-CSD, M=M*
Probe-Category-aware CT-CSD, M=M*, tau_p=0.7
```

阈值消融：

```text
tau_p = 0.5, 0.7, 0.9
```

诊断必须记录：

```text
MIL token probe 训练 loss / validation 指标
每条 harmful response 保留 token 比例
每个 category 保留 token 数
MIL 高分 token 文本样例
probe 前后 cluster 高频 token 对比
```

验收标准：

```text
probe 可以独立训练并保存
probe variants 可以生成与前面 stage 同结构的 ct_csd_bank.pt
推理阶段仍只加载 bank，不额外运行 probe
完成 no-probe vs probe 的同配置对照
```

输出物：

```text
utils/train_mil_token_probe_llada.py
outputs/mil_token_probe_llada.pt
outputs/probe_ct_csd_llada_m{MSTAR}_tau07/ct_csd_bank.pt
outputs/probe_category_ct_csd_llada_m{MSTAR}_tau07/ct_csd_bank.pt
outputs/jbb_dija_probe_category_ct_csd_m{MSTAR}_tau07/
stage4_mil_probe_metrics.md
mil_high_score_tokens.md
```

---

### Stage 5：迁移到 Dream

目标：在 LLaDA 路径稳定后，把同一套 bank 抽象迁移到 Dream，验证方法不是只绑定一个实现。

对应内容：

```text
第 8.8 节：Dream 迁移
第 9.3-9.5 节：实验设置与评测指标
```

进入条件：

```text
Stage 1-4 的 CTCSDBank 接口稳定
LLaDA 上已经确定主结果方法
Category-aware CT-CSD 或 Probe-Category-aware CT-CSD 至少有一组稳定结果
```

实现范围：

```text
新增 utils/make_ct_csd_dream.py
可选新增 utils/train_mil_token_probe_dream.py
修改 eval_dream_steering.py 支持 ct_csd_bank.pt
尽量复用 utils/ct_csd_bank.py
```

只允许改模型相关适配：

```text
Dream transformer block 获取方式
Dream response token hidden states 提取方式
Dream eval hook 接入位置
```

必须保持一致：

```text
bank state_dict 格式一致
routing 逻辑一致
alignment 逻辑一致
steering 公式一致
实验参数尽量与 LLaDA 对齐
```

最小实验：

```text
Dream Global Sentence-CSD
Dream Category-aware CT-CSD, M=M*
可选：Dream Probe-Category-aware CT-CSD, M=M*, tau_p=0.7
```

验收标准：

```text
Dream 可以加载同格式 ct_csd_bank.pt
Dream 旧 global vector baseline 仍然可用
Dream 上至少完成一组主方法 vs baseline 对比
```

输出物：

```text
utils/make_ct_csd_dream.py
utils/train_mil_token_probe_dream.py
outputs/category_ct_csd_dream_m{MSTAR}/ct_csd_bank.pt
outputs/jbb_dija_dream_category_ct_csd_m{MSTAR}/
stage5_dream_metrics.md
```

---

### Stage 6：消融、诊断与论文叙事固化

目标：把前面 stage 的结果整理成能支撑论文或报告的证据链。

对应内容：

```text
第 9 节：实验设计
第 10 节：消融实验
第 11 节：诊断分析
第 12 节：风险与控制
第 13 节：论文/报告叙事
第 16 节：最小实验命令模板
第 17 节：定稿检查清单
```

主实验矩阵：

```text
Global Sentence-CSD
CT-CSD, M=M*
Random-K-CSD, M=M*
Category-aware CT-CSD, M=M*
Probe-CT-CSD, M=M*, tau_p=0.7
Probe-Category-aware CT-CSD, M=M*, tau_p=0.7
```

消融矩阵：

```text
M = 4, 8, 12, 16
tau_p = 0.5, 0.7, 0.9
```

诊断矩阵：

```text
cluster 高频 token 检查
cluster category 分布
MIL 高分 token 检查
route histogram
steering activation rate
failure case 分析
```

指标表至少包含：

```text
ASR
unsafe_count
refusal rate
over-refusal
平均输出长度
空回复比例
steering activation rate
平均被 steering 的 token 数
平均 remask token 数
平均 refinement iterations
推理耗时
额外显存
bank route 耗时
```

验收标准：

```text
每个贡献点都有对应对照实验
每个新增模块都有 no-module 对照
失败 case 有原因归因，不只给最终分数
论文叙事能从 Global Sentence-CSD 逐步推进到最终方法
```

输出物：

```text
stage6_main_results.md
stage6_ablation_results.md
stage6_diagnostics.md
stage6_failure_cases.md
paper_storyline.md
```

---

### 14.1 Stage 间决策规则

每个 stage 结束后按以下规则决定是否进入下一阶段：

| 判断项 | 通过条件 | 不通过时处理 |
|---|---|---|
| 工程闭环 | 构造、加载、推理、评测都跑通 | 先修工程，不进入新方法 |
| baseline 兼容 | 旧 global vector 路径仍可跑 | 回退兼容逻辑 |
| 指标记录 | 安全、质量、干预、效率指标都有日志 | 补评测日志 |
| 消融解释 | 新 stage 只比上一阶段多一个变量 | 拆掉混入的额外变量 |
| 诊断可解释 | route / active / cluster 统计能解释结果 | 补诊断输出 |

如果某个 stage 没有带来提升，也不直接删除。它仍然可以作为负结果或诊断结果保留，但不能继续把它当作后续 stage 的前提。

---

## 15. 最终一句话总结

最终方案是：

> **Category-aware CT-CSD 保留 harmful response 的 token hidden states，不做过早 sentence-level averaging；先按 harmful category 分组，再在类别内部聚类；每个 category-cluster 与全局 safe mean 构造一个局部 steering vector；推理时每个 hidden state 路由到最近 harmful center，但只有当它在对应 harmful direction 上的 alignment 超过阈值 $\theta$ 时才真正执行 steering。MIL token probe 作为 token selection 替换模块，用于在离线构造阶段筛选更可能表达有害语义的 token。**

---

## 16. 附：最小实验命令模板

### 16.1 构造 Category-aware CT-CSD bank

```bash
MSTAR=8  # 示例；实际填写 Stage 2 选出的 M*

python utils/make_ct_csd_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json data/harmbench_csd_train.json \
  --refusals_txt utils/refusals.txt \
  --output_dir "outputs/category_ct_csd_llada_m${MSTAR}" \
  --target_layer 31 \
  --method category_ct_csd \
  --num_total_clusters "${MSTAR}" \
  --category_key semantic_category \
  --max_response_len 128
```

### 16.2 训练 MIL token probe

```bash
python utils/train_mil_token_probe_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json data/harmbench_csd_train.json \
  --refusals_txt utils/refusals.txt \
  --output_path outputs/mil_token_probe_llada.pt \
  --target_layer 31 \
  --top_q_ratio 0.1 \
  --max_response_len 128
```

### 16.3 构造 Probe-Category-aware CT-CSD bank

```bash
MSTAR=8  # 示例；实际填写 Stage 2 选出的 M*

python utils/make_ct_csd_llada.py \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --harmful_json data/harmbench_csd_train.json \
  --refusals_txt utils/refusals.txt \
  --output_dir "outputs/probe_category_ct_csd_llada_m${MSTAR}" \
  --target_layer 31 \
  --method probe_category_ct_csd \
  --mil_probe_path outputs/mil_token_probe_llada.pt \
  --probe_threshold 0.7 \
  --num_total_clusters "${MSTAR}" \
  --category_key semantic_category \
  --max_response_len 128
```

### 16.4 推理评估

```bash
MSTAR=8  # 示例；实际填写 Stage 2 选出的 M*

python eval_llada_steering.py \
  --csv_path JBB \
  --attack_method DIJA \
  --model_path /dev/shm/LLaDA-8B-Instruct \
  --generated_samples_path "outputs/jbb_dija_probe_category_ct_csd_m${MSTAR}" \
  --batch_size 32 \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --dija_mask_counts 128 \
  --remasking low_confidence \
  --sampler steering \
  --remdm_number 4 \
  --cfg 0 \
  --device cuda:0 \
  --self_reminder False \
  --steering_vector_path "outputs/probe_category_ct_csd_llada_m${MSTAR}/ct_csd_bank.pt" \
  --steering_overshoot 1.0 \
  --target_layer 31 \
  --alignment_threshold 0.0 \
  --max_refinement_iters 5 \
  --initial_steering_ratio 0.1
```

---

## 17. 定稿检查清单

- [x] 不把 Token-CSD 作为核心方法。
- [x] 明确核心是“先平均 vs 先聚类再平均”。
- [x] 保留 category 信息，但只用于离线聚类。
- [x] 不引入 safe clustering、soft routing、per-cluster threshold。
- [x] MIL token probe 作为 token selection 替换模块。
- [x] 推理时每个 token 先 route，但只有 $s(h)>\theta$ 才 steering。
- [x] 保留原始 global vector baseline 路径。
- [x] 消融实验能分别验证 clustering、category、MIL 的贡献。
- [x] 实现计划与当前 `make_csd_llada.py` 和 `eval_llada_steering.py` 对齐。
