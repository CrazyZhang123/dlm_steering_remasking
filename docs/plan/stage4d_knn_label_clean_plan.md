# Stage 4D：KNN 标签去噪（Edited Nearest Neighbors）token selection 实施计划

> 状态：**已过 2 轮 review（方案一致性 + 代码落地与测试），Critical/Important 已闭环**
> 创建日期：2026-06-29
> 所属轴：Stage 4 token selection（与 4A direction / 4B MIL / 4C random 并列）；可读别名 **`S4-knn`**
> 关联文档：`docs/plan/dlm_steering_project_improvement_plan.md`、`docs/plan/category_aware_ct_csd_mil_plan.md`、`docs/stage4_mil_token_probe_execution_plan.md`、`docs/category_aware_ct_csd_stage_progress.md`

> **§0 前置依赖（开工前必办）**：`improvement_plan.md` 的 Stage 命名表（1.2 节）与实验矩阵（8.1 节）目前只登记了 4A/4B/4C，**未登记 4D**。开工前须在 improvement_plan 追加一行
> `Stage 4D | KNN-selected Category-CT-CSD（knn_label_clean）| 别名 S4-knn | ENN 标签去噪，独立对照`，
> 使两份文档对 Stage 4 子方法集的定义一致；否则本 Stage 缺上游背书，review 无法判定边界。

---

## 1. 背景与动机

### 1.1 当前 token 标签的问题

当前 CT-CSD 构造中，token 标签是 **response 级弱标签**直接继承给每个 token：

- 有害回复（`harmful_json` 的 `response`）里**每个 token 标为有害**；
- 安全拒绝回复（`refusal`）里**每个 token 标为无害**。

随后用这些 token 的第 `target_layer`（默认 31）层 hidden state 计算 steering 向量：
`v_{c,k} = mean(cluster 内有害 token) − mean(全部安全 token)`（见 `utils/make_ct_csd_llada.py:1088` `accumulate_cluster_sums` 与 `:813` `build_bank_state_from_cluster_sums`）。

问题：一条有害回复里包含大量**中性功能词 / 连接词**（“我”“的”“你”“首先”等）。它们仅因身处有害回复而被打上“有害”标签，本身并不携带有害语义。把它们喂进 cluster center，会**稀释有害方向、引入噪声**。

### 1.2 师兄提出的方法：KNN 标签去噪（ENN）

对每个有害 token，在**全数据集 token 表示空间**中找其最近的 K 个邻居；若邻居标签不一致（有害邻居占比过低），则认为该 token 的“有害”标签不可信，**剔除，不参与 steering 向量计算**。

这是机器学习中的经典思想 **Edited Nearest Neighbors（ENN）标签去噪**：*若一个样本的标签与其多数近邻不一致，则视其为噪声样本并剔除*。

### 1.3 与现有 token selection 方法的关系

本方法与 Stage 4A（direction-selected）目标一致——**只保留判别性强的有害 token**——但筛选信号不同：

| Stage | 方法 | 筛选信号 | 现状 |
|---|---|---|---|
| 4A | direction_top_ratio | 与 coarse CSD 方向的投影分数 top 30% | 主线，待实现 |
| 4B | mil_probe_threshold | MIL probe 分数 ≥ 0.7 | 对照，已部分跑通（ASR 71%，未降） |
| 4C | random_top_ratio | 随机同数量 | token 数对照，待实现 |
| **4D** | **knn_label_clean** | **近邻标签一致性（ENN）** | **本计划新增** |

**预期管理**：当前最低 ASR ≈ 70~71%，且 4B MIL probe 做完 ASR 与 no-probe 持平（`docs/stage4_mil_token_probe_progress.md`）。Stage 4D 不保证降 ASR，其价值在于：提供一种**无需训练、无需预先方向**的 token 去噪信号，作为 4A/4B 的独立对照。**必须与 4C（random 同数量）对照**，才能证明收益（若有）来自“去噪选得准”而非“token 变少”。

---

## 2. 方法定义

### 2.1 形式化

设全数据集经基础过滤（去 special / blank token，见 `make_ct_csd_llada.py:66` `keep_response_token`）后：

- 有害 token 池：`H = {h_1, ..., h_{N_h}}`，每个 `h ∈ R^d`，标签 1；
- 安全 token 池：`S = {s_1, ..., s_{N_s}}`，每个 `s ∈ R^d`，标签 0；
- 合并池 `X = H ∪ S`，`X ∈ R^{N×d}`（`N = N_h + N_s`），标签向量 `y ∈ {0,1}^N`。

对每个有害 token `h_i`：

1. 在 `X` 中查其最近的 `K` 个邻居（**排除自身**），度量为 `knn_metric`（默认 cosine）；
2. 计算有害邻居占比 `r_i = (邻居中 y=1 的个数) / K`；
3. **保留判据**：`r_i ≥ knn_keep_ratio` 则保留，否则剔除。

只有保留下来的有害 token 进入后续 KMeans 路由与 cluster 累加。**安全 token 池 `S` 不做删除**（见 §2.2）。

### 2.2 不变量：只清洗有害侧

与 Stage 4B execution plan 一致（`docs/stage4_mil_token_probe_execution_plan.md:97`），token selection **只影响有害 token 集合**：

- `safe_mean` 仍由**全部**安全 token 的 sample-balanced 均值计算，保持固定；
- 安全 token 仅作为 KNN 池中的“无害类参考”，参与邻居投票，但**自身不被删除**；
- 推理公式（route → alignment → 阈值 steering，见 `utils/ct_csd_bank.py:169` `steer`）**完全不变**。

这样 Stage 4D 与 Stage 3 / 4A / 4B / 4C 严格可比：唯一变量是“有害侧 token 子集”。

---

## 3. 架构设计

### 3.1 关键事实：现有流程已是两遍扫描

`make_ct_csd_llada.py` 的离线构造已是 pass1 + pass2：

- **pass 1**：`fit_minibatch_kmeans`（`:949`）/ `fit_category_minibatch_kmeans`（`:1006`）——遍历有害样本训练 KMeans + 累加 `safe_mean`；
- **pass 2**：`accumulate_cluster_sums`（`:1088`）/ `accumulate_category_cluster_sums`（`:1142`）——再遍历一遍，把 token 分配到 cluster 累加 `cluster_sums`。

两遍都通过统一入口 `iter_valid_sample_tokens`（`:903`）取筛选后的 token；现有 per-sample 筛选（direction / random / probe）都塞在该入口内。

### 3.2 全局 KNN 为何不能塞进 per-sample 入口

`select_harmful_response_tokens`（`:524`）每次只看到**单条样本**的有害 token，拿不到全数据集，也拿不到安全 token 池——无法构建全局邻域。因此全局 KNN 必须**在 pass 1 之前增加一个 pass 0 预扫描**，先把全局池建好、把每个有害 token 的“保留/剔除”结论算出来并缓存；pass 1 / pass 2 的 per-sample 入口只做“查缓存、应用 mask”。

### 3.3 三遍结构（仅 `knn_label_clean` 模式启用 pass 0）

```
fit_route_preprocess（现有，无条件运行，:1360）
        └─ 内部调 select_harmful_response_tokens；此时 KNN 缓存未就绪 → 等效 "all"（见 §3.6）
pass 0（新增，仅 knn_label_clean，紧接 fit_route_preprocess 之后、pass1 之前）：build_knn_keep_masks(...)
        ├─ 有害池循环：仅 forward 有害侧（build_sequence + extract + 基础过滤），收集 H（向量 + sample_index + position + token_id）
        ├─ 安全池循环（独立，choose_aux_refusal 抽 refusal）：收集 S（向量 + 标签 0）
        ├─ 拼接 X=[H;S], y，归一化后构建 KNN 索引（faiss / sklearn）
        ├─ 对每个有害 token 查 K 近邻（按 index 排除自身），算 r_i，得 keep 标记
        └─ 缓存 args._knn_keep_masks: dict[sample_index -> BoolTensor]
pass 1：fit_(category_)minibatch_kmeans —— iter_valid_sample_tokens 内查缓存过滤
pass 2：accumulate_(category_)cluster_sums —— 同上
```

### 3.4 对齐与确定性（Critical，已据 review C1 修正）

pass 0 与 pass 1/2 必须对同一 `sample_index` 得到**完全相同顺序、相同长度**的有害 token 序列，缓存的 `BoolTensor` 才能按位对齐。

**风险点**：`extract_sample_response_tokens`（`:609`）的长度门控是 `if len(ids_h) > max_total_len or len(ids_s) > max_total_len: return None`（`:624`），其中 `ids_s` 依赖 refusal。若 pass 0 与 pass 1/2 对同一样本选到长度不同的 refusal，可能一侧跳过、一侧不跳过，破坏“所有有害样本都经 KNN 清洗”的语义。

**修正方案**：pass 0 的**有害池收集与安全侧完全解耦**——有害侧只调 `build_sequence(tokenizer, prompt, harmful_resp, args.max_response_len)` + `extract_target_layer_tokens` + `filter_response_tokens`，**只检查 `len(ids_h) > max_total_len`**，不 forward 安全侧、不抽 refusal。安全池在**独立循环**里收集（少几条不影响有害侧对齐）。这样有害 token 的 skip/not-skip 判定与 refusal 彻底无关，对齐变为无条件成立。

对齐保证来源：

- forward 在 `@torch.no_grad()` 下确定性；
- 基础过滤 `filter_response_tokens`（`:87`）只依赖 token id，确定性；
- 有害侧 `build_sequence` 仅依赖 `prompt + harmful_resp + max_response_len`，三遍一致。

key 采用 `sample_index`（`args._sample_index`），value 为该样本过滤后有害 token 的 `BoolTensor`，长度等于 `select_harmful_response_tokens` 入口处 `hidden.shape[0]`。应用时**断言长度一致**，不一致立即报错（防止 silent 错位）。

### 3.5 RNG 隔离（Critical，已据 review I4/I3 修正）

已有测试 `tests/test_make_ct_csd_llada.py:379`
`test_extra_direction_and_preprocess_passes_do_not_consume_global_refusal_rng`
要求：**任何额外的 pass 不得消耗全局 `random` 状态**，否则 pass1/pass2 的 `refusal = random.choice(refusals)` 配对漂移，破坏与其它 Stage 的可比性。

因此 pass 0 安全池抽 refusal 时，**复用现有 `choose_aux_refusal(refusals, args, idx, pass_salt=47)`**（`:467`，per-sample 确定性、内部 `random.Random(seed + pass_salt*1_000_003 + idx)`，已被 coarse direction `pass_salt=41`、route preprocess `pass_salt=53` 使用）。`47` 为未占用的新 salt。**不得手建 `random.Random` 顺序抽取**（输出会依赖样本遍历顺序），更不得调用全局 `random.*`。其它随机源同样固定：KMeans `random_state=args.seed`（现有）、KNN 无随机性、安全池下采样若有也用局部 generator。Stage 4D 需新增对应回归测试（见 §6，复用 `:379` 的 mock 模式）。

### 3.6 预处理阶段等效 `all`（Important，据 review C2/I1）

`fit_route_preprocess`（`:714`/调用点 `:1360`）**无条件**在 pass1 前运行，内部 `:766` 会调 `select_harmful_response_tokens`。由于 pass 0 安排在其**之后**，预处理阶段 `args._knn_keep_masks` 尚不存在。因此 `select_harmful_response_tokens` 的 `knn_label_clean` 分支约定：**当缓存属性不存在（`getattr(args, "_knn_keep_masks", None) is None`）时直接返回原 token，等效 `all`**。这与 `direction_top_ratio` 模式下“预处理用全 token 计算 mean/PCA”的现有行为一致，是预期语义，不是 bug。**无需修改 `:531` 的 early-return 集合**，只需新增一个独立的 `knn_label_clean` 分支（放在 `:558` 的 `raise` 之前）。

---

## 4. 接口设计

### 4.1 CLI 参数（`make_ct_csd_llada.py` argparse，紧跟 `--token_selection`（`:1301`）之后追加）

- 扩展 `TOKEN_SELECTION_CHOICES`（`:475`）：新增 `"knn_label_clean"`。
- 新增参数：

| 参数 | 默认 | help 说明 |
|---|---|---|
| `--knn_k` | `6` | 近邻数 K（消融 4 / 6 / 8 / 10） |
| `--knn_keep_ratio` | `0.5` | 有害邻居占比阈值，`r_i ≥` 此值保留（消融 0.5 / 0.6 / 0.7） |
| `--knn_metric` | `cosine` | `cosine` / `euclidean` |
| `--knn_backend` | `auto` | `auto` / `faiss` / `sklearn`；auto 优先 faiss，缺失回退 sklearn |
| `--knn_safe_pool_cap` | `0` | `0 = 不限（用全部安全 token）`；正整数 = 安全池下采样上限，控制 pass0 内存/时间 |

> **命名说明（据 review I1）**：`TOKEN_SELECTION_CHOICES` 现有项用 `_top_ratio` / `_threshold` 后缀表达“截断式选择”。本方法不是 top-ratio 截断，而是基于邻居标签一致性的 **ENN 标签去噪**，保留量由一致性判据动态决定，故用 `_label_clean` 后缀表意，刻意区别于截断式选项。
> 不引入与 4A/4C 重复的 `selection_ratio`/`max_selected_tokens`——保持 KISS / YAGNI。

### 4.2 数据结构

- 全局池：`X: torch.Tensor [N, d] (float32)`、`y: torch.Tensor [N] (int8)`；
- 索引映射：`harmful_index -> (sample_index, position, token_id)`，用于把每个有害 token 的 keep 结论写回对应样本，并支持诊断 decode（§4.4，据 review M4）；
- 缓存：`args._knn_keep_masks: dict[int, torch.BoolTensor]`。

### 4.3 配置落盘

`state["config"]` 增加 `token_selection="knn_label_clean"`、`knn_k`、`knn_keep_ratio`、`knn_metric`、`knn_backend`、`knn_safe_pool_cap`、`knn_retention_ratio`（实际保留比例，便于横比），并记录诊断统计（见 §4.4），与现有 `:1443` 配置写出风格一致。

### 4.4 诊断产物

写出 `knn_label_clean_summary.json`（仿 `mil_token_selection_summary.json`，`:246`）：

- `total_harmful_tokens`、`kept_harmful_tokens`、`retention`（保留率）；
- `removed_top_terms`：被剔除最多的 token 文本 top-N（预期含大量功能词，用于人工 sanity check）；
- `kept_top_terms`：保留 token top-N；
- 每参数组合一份，便于和 4A/4C retention 对照。

---

## 5. 算法伪码（函数签名以代码为准）

```python
TOKEN_SELECTION_CHOICES = (..., "knn_label_clean")  # :475
KNN_PASS_SALT = 47                                   # §3.5，未占用（41=direction, 53=preprocess）

def build_knn_keep_masks(model, tokenizer, harmful, refusals, args, device) -> dict[int, BoolTensor]:
    # ---- (1) 有害池：与安全侧彻底解耦，只 forward 有害侧（§3.4 C1） ----
    H, h_meta = [], []                               # h_meta[i] = (sample_index, position, token_id)
    for idx, sample in enumerate(harmful):
        prompt, harmful_resp = sample["prompt"], sample["response"]
        ids_h, rs_h, response_ids_h = build_sequence(tokenizer, prompt, harmful_resp, args.max_response_len)  # :313
        if len(ids_h) > args.max_total_len:          # 只检查有害侧，不涉及 refusal
            continue
        h_tokens = extract_target_layer_tokens(model, ids_h.unsqueeze(0), rs_h, args.target_layer, device)    # :441
        h_tokens, h_token_ids = filter_response_tokens(tokenizer, response_ids_h, h_tokens)                    # :87
        for pos in range(h_tokens.shape[0]):
            H.append(h_tokens[pos]); h_meta.append((idx, pos, int(h_token_ids[pos])))

    # ---- (2) 安全池：独立循环，choose_aux_refusal 保持 RNG 隔离（§3.5 I3） ----
    S = []
    for idx, sample in enumerate(harmful):
        refusal = choose_aux_refusal(refusals, args, idx, pass_salt=KNN_PASS_SALT)  # :467，禁用全局 random
        ids_s, rs_s, response_ids_s = build_sequence(tokenizer, sample["prompt"], refusal, args.max_response_len)
        if len(ids_s) > args.max_total_len:
            continue
        s_tokens = extract_target_layer_tokens(model, ids_s.unsqueeze(0), rs_s, args.target_layer, device)
        s_tokens, _ = filter_response_tokens(tokenizer, response_ids_s, s_tokens)
        if args.knn_safe_pool_cap > 0 and s_tokens.shape[0] > 0:
            s_tokens = subsample(s_tokens, args.knn_safe_pool_cap, generator=local_gen(args.seed + idx))
        S.append(s_tokens)

    # ---- (3) 建索引 + 查近邻（I6 归一化统一；I2 按 index 排除自身） ----
    X = torch.cat([torch.stack(H)] + S, dim=0).float()
    n_h = len(H); y = torch.cat([torch.ones(n_h), torch.zeros(X.shape[0]-n_h)]).to(torch.int8)
    X = l2_normalize(X)                              # cosine: 归一化后内积/欧氏均单调等价
    # faiss 后端：IndexFlatIP，取 top-k 最大内积(=最相似)
    # sklearn 后端：NearestNeighbors(metric="euclidean", algorithm="brute")，取 top-k 最小欧氏(=最相似)
    #   —— 切勿用 sklearn metric="cosine" + 未归一化向量（数值口径与 faiss IP 不一致）
    index = build_index(X, backend=args.knn_backend)            # auto: faiss 优先，缺失回退 sklearn
    nbr_idx = index.query(X[:n_h], k=args.knn_k + 1)            # +1 用于排除自身
    keep = torch.zeros(n_h, dtype=torch.bool)
    for i in range(n_h):
        nb = [j for j in nbr_idx[i] if j != i][:args.knn_k]     # 按 query index == neighbor index 排除自身
        r_i = float((y[nb] == 1).float().mean())
        keep[i] = (r_i >= args.knn_keep_ratio)

    # ---- (4) 回写为按 position 有序的 BoolTensor（M1） ----
    pairs = defaultdict(list)
    for i, (idx, pos, _tok) in enumerate(h_meta):
        pairs[idx].append((pos, bool(keep[i])))
    masks = {idx: torch.tensor([v for _p, v in sorted(p)], dtype=torch.bool) for idx, p in pairs.items()}
    write_knn_diagnostics(args, tokenizer, h_meta, keep)        # §4.4，用 token_id decode top-terms
    return masks

# select_harmful_response_tokens(:524)：在 :558 raise 之前新增独立分支
if mode == "knn_label_clean":
    masks = getattr(args, "_knn_keep_masks", None)
    if masks is None:                                # 预处理阶段缓存未就绪 → 等效 all（§3.6 C2）
        return hidden, token_ids
    mask = masks.get(args._sample_index)
    if mask is None:                                 # 该样本 pass0 被跳过（罕见，I4）
        warn(f"[knn] sample {args._sample_index} 无 mask，保守保留全部 token")
        return hidden, token_ids
    assert mask.shape[0] == hidden.shape[0], f"KNN mask 与 token 序列错位: {mask.shape[0]} vs {hidden.shape[0]}"
    return hidden[mask], token_ids[mask.cpu()]

# main()：fit_route_preprocess(:1360) 之后、pass1(fit_*_minibatch_kmeans) 之前
if args.token_selection == "knn_label_clean":
    args._knn_keep_masks = build_knn_keep_masks(model, tokenizer, harmful, refusals, args, device)
```

---

## 6. 测试计划（`tests/test_make_ct_csd_llada.py`）

仿现有 token selection 测试（`:203` direction、`:228` random）新增：

1. `test_knn_label_clean_removes_boundary_tokens` —— 构造小数据集：3 个明显有害向量聚一簇、3 个明显安全向量聚一簇、1 个夹在中间的有害 token；`knn_k=6, knn_keep_ratio=0.5`，断言边界 token 被剔除、纯有害 token 保留。
2. `test_knn_label_clean_keep_ratio_threshold` —— 同一 token 在 `keep_ratio=0.5` 保留、`0.7` 剔除，验证阈值语义。
3. `test_knn_keep_masks_align_across_passes_without_assertion`（**据 review I5 强化为集成测试**）—— mock 2~3 个样本，实际跑 `build_knn_keep_masks` + `iter_valid_sample_tokens`（后者内部调 `select_harmful_response_tokens` 应用 mask），断言 `mask.shape[0] == hidden.shape[0]` 不触发、mask 被正确应用、跨 pass token 序列一致。
4. `test_knn_pass_does_not_consume_global_refusal_rng` —— **回归测试**：完全复用 `:379` 的 mock 模式（`patch` 全局 `random.choice` 使其 `side_effect=AssertionError`），断言 `build_knn_keep_masks` 不触碰全局 `random`。
5. `test_knn_backend_auto_falls_back_to_sklearn` —— 分两层（据 review M3）：(a) 必测——monkeypatch faiss 缺失，断言回退 sklearn 不崩溃且返回有效 mask；(b) 可选——`pytest.importorskip("faiss")` 下对比两后端输出一致。
6. `test_knn_label_clean_equivalent_to_all_when_cache_missing`（**据 review C2**）—— `args._knn_keep_masks` 不存在时，`select_harmful_response_tokens(mode="knn_label_clean")` 返回原 token（验证预处理阶段等效 all、不触发 `:558` raise）。
7. `test_main_writes_knn_ct_csd_bank_with_cli_metadata` —— end-to-end：`token_selection=knn_label_clean` 跑通，bank `config` 含全部 knn 参数 + `knn_retention_ratio`，诊断 json 写出（仿 `:556` / `:1051`）。

要求：`pytest tests/test_make_ct_csd_llada.py -q` 全绿；`pytest tests/test_ct_csd_bank.py -q` 回归通过。

---

## 7. 性能与降级

### 7.1 规模估算（需先实测有害样本数）

设有害样本 `R` 条、过滤后每条约 `T` 个有害 token：`N_h ≈ R·T`；安全池同量级，`N ≈ 2·R·T`，`d = 4096`。

- 内存：`N×d×4B`。若 `N = 5e4` → 约 0.8 GB（可放内存）；`N = 2e5` → 约 3.3 GB。
- 查询：`N_h` 次 K 近邻。
  - **faiss**（`IndexFlatIP`，cosine 用归一化向量）：`N=5e4` 量级秒级~十几秒。
  - **sklearn `NearestNeighbors(algorithm="brute")`**：`5e4×5e4×4096` 量级，约分钟级，可接受。

### 7.2 降级与优化（按需，先不过度设计）

- `--knn_backend auto`：faiss 优先，缺失回退 sklearn brute（保证可跑）；
- `--knn_safe_pool_cap`：安全池过大时下采样，控成本；
- 若 `N` 过大（>5e5）再考虑 PCA 降维加速——**YAGNI，先实测再决定**，不在首版实现。

---

## 8. 实验与评测口径

遵循 `AGENTS.md` 实验/评测约定：

- **生成**：沿用 `AGENTS.md` / `improvement_plan` §12.10 固定口径——JBB 与 DIJA 两个攻击方式**各 100 条分别评测**（`attack_method ∈ {zeroshot/JBB, DIJA}`），不混算；长任务一律 `tmux` 后台、日志重定向到输出目录。
- **评判**：本地 Llama-Guard，核心指标 `unsafe_count` / `ASR`。
- **构造命令**（示例，`l2_only` 预处理与 4A/4B/4C 对齐，便于横比）：

```bash
tmux new -s s4d_knn
CUDA_VISIBLE_DEVICES=0 python utils/make_ct_csd_llada.py \
  --token_selection knn_label_clean \
  --knn_k 6 --knn_keep_ratio 0.5 --knn_metric cosine --knn_backend auto \
  --num_total_clusters 16 --feature_preprocess l2_only \
  --category_aware ...  2>&1 | tee outputs/s4d_knn/build.log
```

- **对照表**（据 review I2 补入 4B）：Stage 3（baseline）vs 4A（direction）vs **4B（MIL，已有数据，参考行）** vs 4C（random）vs **4D（knn）**，固定 M=16、`l2_only`、同一评测集。4B 无需重跑，直接引用已有 ASR。
- **消融**：`knn_k ∈ {4,6,8,10}`、`knn_keep_ratio ∈ {0.5,0.6,0.7}`，先粗扫 `k=6` × `ratio∈{0.5,0.7}` 两点定位。

指标记录到 `docs/stage4d_knn_label_clean_metrics.md`，进度同步到 `docs/category_aware_ct_csd_stage_progress.md`。

---

## 9. 验收标准（退出条件）

1. **代码**：`token_selection=knn_label_clean` 跑通 global 与 category-aware 两条路径；§6 全部单测 + ct_csd_bank 回归通过。
2. **不变量**：safe_mean 与推理公式不变；RNG 隔离回归测试通过；mask 对齐断言生效。
3. **诊断**：`knn_label_clean_summary.json` 写出，retention 合理（被剔除 token top-terms 以功能词为主，符合预期）。
4. **实验**：完成 4D 与 3 / 4A / 4C 在 M=16 / `l2_only` 同口径对照，`unsafe_count` / `ASR` 记入 metrics 文档。
5. **结论**：明确回答“KNN 去噪相对 random 同数量是否有净收益”，无论正负都记录归档。

---

## 10. 风险与回退

| 风险 | 影响 | 缓解 |
|---|---|---|
| pass 0 多一遍 forward，GPU 时间 +50% | 构造变慢 | 仅 knn 模式启用；`tmux` 后台；可后续缓存全局池复用 |
| mask 与 token 序列错位 | 静默污染 steering 向量 | §3.4 断言长度一致，错位即报错 |
| 误消耗全局 refusal RNG | 破坏与其它 Stage 可比性 | §3.5 独立 RNG + §6 回归测试 |
| ASR 不降甚至升 | 方法无收益 | 与 4C 对照即可解释；负结果照常归档，不强行上主线 |
| N 过大导致 OOM / 超时 | 跑不动 | `--knn_safe_pool_cap` 下采样；faiss 优先；必要时 PCA（二期） |

**回退**：本方法是**纯增量**——不改推理、不改 Stage 3/4A/4B/4C 任何路径，仅新增一个 `token_selection` 选项与一个条件触发的 pass 0。若效果不佳，保留代码作为对照，不影响既有主线。

---

## 11. 分步实施清单

> **实现进度（2026-06-29）**：代码项 **S1/S2/S3/S4/S5 已实现**，纯 KNN 增量改动于 `utils/make_ct_csd_llada.py` + `tests/test_make_ct_csd_llada.py`，通过**代码 2 轮 review（无 Critical；闭环 2 个 Important：safe 池空 degenerate 告警、e2e main 测试）**，全量 **44 项单测 + ct_csd_bank 11 项回归全绿**。剩余 **S0a 注册、S0b 实测规模、S6 实验、S7 文档**待执行（需 GPU + 模型）。

- [ ] **S0a 注册 4D**（§0 前置）：在 `improvement_plan.md` Stage 命名表 + 实验矩阵登记 Stage 4D / 别名 `S4-knn`。
- [ ] **S0b 实测规模**：统计有害样本数 `R`、平均过滤后 token 数 `T`，估 `N` 与内存/时间，决定 backend 与是否需要 cap。
- [ ] **S1 接口**：扩 `TOKEN_SELECTION_CHOICES`；紧跟 `--token_selection` 追加 `--knn_*` 参数；config 落盘字段（含 `knn_safe_pool_cap`、`knn_retention_ratio`）。
- [ ] **S2 pass 0**：实现 `build_knn_keep_masks`（有害/安全分离收集、`choose_aux_refusal(pass_salt=47)`、对齐 meta 含 token_id、faiss/sklearn 归一化统一、诊断收集）。
- [ ] **S3 入口分支**：`select_harmful_response_tokens` 在 `:558` raise 前加 `knn_label_clean` 分支（缓存缺失等效 all + 缺 mask warning + 长度断言）；`main` 在 `fit_route_preprocess` 之后、pass1 之前条件触发 pass 0。
- [ ] **S4 诊断**：写 `knn_label_clean_summary.json`。
- [ ] **S5 测试**：§6 全部单测 + 回归，跑绿。
- [ ] **S6 实验**：粗扫 `k=6 × ratio∈{0.5,0.7}`，与 3/4A/4C 对照；扩消融。
- [ ] **S7 文档**：metrics + 进度文档更新；3 轮 review 闭环（方案一致性 / 代码质量与测试 / 验收产物）。

---

## 12. 已知问题：KNN 池类别不平衡导致去噪失效（2026-06-29 smoke test）

50 样本最小构造跑通（method=ct_csd, l2_only, M=16, k=6, keep_ratio=0.5），但 `retention = 6062/6074 = 99.8%`，几乎不去噪。

诊断数据（`outputs/s4d_knn_smoke/knn_label_clean_summary.json`）：

| 量 | 值 |
|---|---|
| 有害 token | 6074（T_harmful ≈ 121/条，接近 `max_response_len=128`） |
| 安全 token | 518（T_refusal ≈ 10.4/条） |
| 有害 : 安全 | ≈ **11.7 : 1** |
| removed_top_terms | `'m / 't / Not / I / am …`（refusal 措辞，方向对但仅 12 个） |
| kept_top_terms | `a(370) / the(244) / to(153) / and(135) …`（功能词本应剔除却保留） |

**根因**：安全池每个有害样本只配 **1 条 refusal**（`build_knn_keep_masks` 安全循环），有害池每样本 1 条 response。两者都 ∝ `max_samples`，故
`有害:安全 = (N·T_harmful)/(N·T_refusal) = T_harmful/T_refusal`，**与 max_samples 无关**。ENN 投票被多数类（有害）主导：功能词的 k 近邻大概率仍是有害 token → 保留。

**调大 max_samples 无效**（重要结论）：有害/安全 token **等比例放大**，比例恒定 ≈11.7:1；且 `refusals.txt` 仅 20 条模板，放大安全池只是堆**重复 token**，有效多样性不增。

**解决方向**：

1. **per-class 加权投票**（首选，§13）——治本，与池大小无关；
2. 平衡采样（下采样有害 / 上采样安全到 1:1）——丢信息 / 堆重复；
3. 每样本取多条 refusal——受限于 20 条模板；
4. 调小 `max_response_len` 截短有害——丢有害语义。

---

## 13. Stage 4D-balanced：per-class 加权投票设计

**核心**：ENN 投票时按邻居所属类的**全局池大小归一化票权**，消除「实例数」偏置。

设 token i 的 k 近邻中有害邻居数 `|H_i|`、安全邻居数 `|S_i|`，有害池大小 `N_h`、安全池大小 `N_s`：

```
有害得分 w_h = |H_i| / N_h
安全得分 w_s = |S_i| / N_s
归一化有害占比 r_i = w_h / (w_h + w_s)      （w_h + w_s = 0 时保守保留）
保留 iff r_i >= knn_keep_ratio
```

直觉：`N_h ≈ 11.7·N_s` 时，每个有害邻居票权仅安全邻居的 `1/11.7`。功能词若 6 邻居中 5 有害 1 安全 → `r = (5/N_h) / (5/N_h + 1/N_s) = 5/(5+11.7) ≈ 0.30 < 0.5` → **剔除**（标准投票下 `5/6=0.83` 会保留）。

**接口**：新增 `--knn_balanced`（flag，默认 `False`，保持现有 ENN 行为向后兼容）。`knn_keep_decisions` 加 `balanced: bool` 参数；config 落盘 `knn_balanced`。

**测试**：构造**不平衡** toy 池（多有害 + 少安全，含落在安全簇附近的"功能词"有害 token），断言 `balanced=False` 保留该 token、`balanced=True` 剔除该 token；并复跑既有 7+1 用例确保 `balanced=False` 行为不变。

**与对照**：balanced 仍须与 4C random 同口径对比，证明收益来自"按邻域去噪"而非单纯减 token。

---

## 附录 A：关键代码锚点

| 锚点 | 位置 |
|---|---|
| `TOKEN_SELECTION_CHOICES` | `utils/make_ct_csd_llada.py:475` |
| `select_harmful_response_tokens`（per-sample 筛选分发） | `:524` |
| `extract_sample_response_tokens`（取有害/安全 token） | `:609` |
| `extract_target_layer_tokens`（layer hook，`:440` 为 `@torch.no_grad()` 装饰器） | `:441` |
| `filter_response_tokens` / `keep_response_token`（基础过滤） | `:87` / `:66` |
| `build_sequence`（构建 prompt+response 序列） | `:313` |
| `choose_aux_refusal`（per-sample 确定性 refusal） | `:467` |
| `fit_route_preprocess`（无条件预处理 pass，内部调 select） | `:714` / 调用点 `:1360` |
| `iter_valid_sample_tokens`（统一入口） | `:903` |
| `fit_minibatch_kmeans` / `fit_category_minibatch_kmeans`（pass1） | `:949` / `:1006` |
| `accumulate_cluster_sums` / `accumulate_category_cluster_sums`（pass2） | `:1088` / `:1142` |
| `build_bank_state_from_cluster_sums`（v = center − safe_mean） | `:813` |
| `steer`（推理，不变） | `utils/ct_csd_bank.py:169` |
| RNG 隔离回归测试样板 | `tests/test_make_ct_csd_llada.py:379` |
| token selection 单测样板 | `tests/test_make_ct_csd_llada.py:203` / `:228` |
