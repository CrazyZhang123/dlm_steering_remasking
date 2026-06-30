# Stage 3 Category-aware CT-CSD 实施计划

> **给 agentic workers：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐项实施本计划。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标：** 在现有 LLaDA CT-CSD 最小闭环上新增 `category_ct_csd` 构造分支，验证 harmful category 约束是否能让 cluster 更稳定、更可解释，并保持推理阶段继续只加载同一种 bank。

**架构：** Stage 3 只改离线构造和诊断，不改 routing / steering 公式。`utils/make_ct_csd_llada.py` 增加 category 解析、按 token 数分配每类 cluster budget、类别内 MiniBatchKMeans、category metadata 写入；`utils/ct_csd_bank.py` 只增强可选诊断字段，仍接受 `format == "ct_csd_v1"`，让 `eval_llada_steering.py` 不需要 prompt category。

**hidden state 口径：** Stage 3 构造时模型前向输入仍是完整的 `prompt + response`，让 response hidden state 带有 prompt 条件上下文；但用于 category token 计数、KMeans fit、cluster accumulate 的 hidden states 只截取 response 段，即 `hidden[:, response_start:, :]`。截取后继续过滤 special token 和空白 token。prompt token 不进入聚类，完整序列 token 也不整体进入聚类。safe anchor 同理只使用 safe refusal response 段的有效 token hidden states，并对每条 refusal response 求均值后进入全局 safe mean。

**技术栈：** Python 3.13, PyTorch, scikit-learn `MiniBatchKMeans`, `unittest`, existing `eval_llada_steering.py`, existing `CTCSDBank`

---

## 文件结构

**修改**
- `utils/make_ct_csd_llada.py`
  - 新增 `category_ct_csd` CLI method。
  - 新增 category fallback、category token count 估计、cluster budget 分配、类别内 KMeans fit / accumulate。
  - 对 `ct_csd` 和 `category_ct_csd` 都写出 cluster/category 诊断摘要。
- `utils/ct_csd_bank.py`
  - 读取可选 `center_categories` / `categories`。
  - `diagnostics()` 输出 per-category route / active 统计；routing 和 steering 逻辑不变。
- `tests/test_make_ct_csd_llada.py`
  - 增加 category helper、budget 分配、category bank state、CLI metadata 测试。
- `tests/test_ct_csd_bank.py`
  - 增加可选 category metadata 的诊断聚合测试。

**不修改**
- `eval_llada_steering.py`
  - Stage 3 不需要修改。它已经通过 `format == "ct_csd_v1"` 加载 `CTCSDBank`。

**输出物**
- `outputs/category_ct_csd_llada_m{MSTAR}/ct_csd_bank.pt`
- `outputs/category_ct_csd_llada_m{MSTAR}/ct_csd_bank_summary.json`
- `outputs/category_ct_csd_llada_m{MSTAR}/cluster_category_distribution.md`
- `outputs/jbb_dija_category_ct_csd_m{MSTAR}/results.json`
- `outputs/jbb_dija_category_ct_csd_m{MSTAR}/ct_csd_diagnostics.json`
- `docs/stage3_category_ct_csd_metrics.md`

**重要约束**
- 本计划不包含 `git commit` / `git push` 步骤。AGENTS.md 明确要求未主动请求时不要规划或执行 git 提交与分支操作。
- Stage 3 不训练 MIL token probe，不新增 prompt category classifier，不在推理时预测 prompt 类别。
- Stage 3 必须继承 Stage 2 选出的 `M*`。如果 Stage 2 还没有最终 `M*`，只能把 `MSTAR=8` 标注为 smoke 参数，不能写成正式结论。

## 任务 1：补充 category helper 失败测试

**文件：**
- 修改：`tests/test_make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：为 category fallback 和 cluster 分配写失败测试**

把以下测试追加到 `tests/test_make_ct_csd_llada.py` 的 `BuilderHelpersTest` 中：

```python
    def test_resolve_category_prefers_requested_key_then_fallbacks(self):
        sample = {
            "semantic_category": " illegal ",
            "functional_category": "standard",
            "category": "fallback",
        }

        self.assertEqual(builder.resolve_category(sample, "semantic_category"), "illegal")
        self.assertEqual(builder.resolve_category(sample, "missing_key"), "illegal")
        self.assertEqual(
            builder.resolve_category({"functional_category": " contextual "}, "semantic_category"),
            "contextual",
        )
        self.assertEqual(builder.resolve_category({}, "semantic_category"), "unknown")

    def test_count_response_tokens_by_category_uses_filtered_response_tokens(self):
        args = SimpleNamespace(max_response_len=128, category_key="semantic_category")
        harmful = [
            {"response": "harmful1", "semantic_category": "illegal"},
            {"response": "harmful2", "semantic_category": "fraud"},
            {"response": "   ", "semantic_category": "ignored"},
        ]

        counts = builder.count_response_tokens_by_category(_SequenceTokenizer(), harmful, args)

        self.assertEqual(counts, {"illegal": 2, "fraud": 2})

    def test_make_category_cluster_plan_allocates_total_budget_and_merges_tail(self):
        plan = builder.make_category_cluster_plan(
            {
                "illegal": 100,
                "cyber": 60,
                "fraud": 30,
                "privacy": 10,
            },
            num_total_clusters=3,
        )

        self.assertEqual(plan["categories"], ["cyber", "illegal", "other"])
        self.assertEqual(plan["category_token_counts"], {"cyber": 60, "illegal": 100, "other": 40})
        self.assertEqual(sum(plan["category_cluster_counts"].values()), 3)
        self.assertEqual(plan["category_cluster_counts"], {"cyber": 1, "illegal": 1, "other": 1})
        self.assertEqual(plan["raw_to_center_category"]["fraud"], "other")
        self.assertEqual(plan["raw_to_center_category"]["privacy"], "other")

    def test_make_category_cluster_plan_distributes_remaining_budget_by_fraction(self):
        plan = builder.make_category_cluster_plan(
            {
                "a": 70,
                "b": 20,
                "c": 10,
            },
            num_total_clusters=5,
        )

        self.assertEqual(plan["categories"], ["a", "b", "c"])
        self.assertEqual(plan["category_cluster_counts"], {"a": 3, "b": 1, "c": 1})
        self.assertEqual(sum(plan["category_cluster_counts"].values()), 5)
```

- [ ] **步骤 2：运行 helper 测试，确认失败**

运行：

```bash
python -m unittest "tests/test_make_ct_csd_llada.py" -v
```

预期：

```text
ERROR: AttributeError: module 'utils.make_ct_csd_llada' has no attribute 'resolve_category'
```

## 任务 2：实现 category 纯函数

**文件：**
- 修改：`utils/make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：在 `load_harmful_data` 后添加 helper**

把以下代码添加到 `utils/make_ct_csd_llada.py` 的 `load_harmful_data` 后：

```python
CATEGORY_FALLBACK_KEYS = ("semantic_category", "functional_category", "category")


def resolve_category(sample: dict, category_key: str | None = None) -> str:
    keys: list[str] = []
    if category_key:
        keys.append(category_key)
    keys.extend(key for key in CATEGORY_FALLBACK_KEYS if key not in keys)

    for key in keys:
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "unknown"


def count_response_tokens_by_category(tokenizer, harmful: list[dict], args) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in harmful:
        response = str(sample.get("response", "")).strip()
        if not response:
            continue
        response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if len(response_ids) > args.max_response_len:
            response_ids = response_ids[: args.max_response_len]
        kept = sum(1 for token_id in response_ids if keep_response_token(tokenizer, int(token_id)))
        if kept <= 0:
            continue
        category = resolve_category(sample, getattr(args, "category_key", None))
        counts[category] = counts.get(category, 0) + int(kept)
    return counts


def make_category_cluster_plan(
    category_token_counts: dict[str, int],
    num_total_clusters: int,
) -> dict:
    if num_total_clusters <= 0:
        raise ValueError("num_total_clusters must be positive")

    positive = {
        str(category): int(count)
        for category, count in category_token_counts.items()
        if int(count) > 0
    }
    if not positive:
        raise RuntimeError("No category has usable harmful response tokens.")

    ranked = sorted(positive.items(), key=lambda item: (-item[1], item[0]))
    raw_to_center_category: dict[str, str] = {}

    if len(ranked) > num_total_clusters:
        if num_total_clusters == 1:
            kept: list[tuple[str, int]] = []
            tail = ranked
        else:
            kept = ranked[: num_total_clusters - 1]
            tail = ranked[num_total_clusters - 1 :]
        collapsed: dict[str, int] = {category: count for category, count in kept}
        other_count = sum(count for _category, count in tail)
        collapsed["other"] = collapsed.get("other", 0) + other_count
        for category, _count in kept:
            raw_to_center_category[category] = category
        for category, _count in tail:
            raw_to_center_category[category] = "other"
    else:
        collapsed = dict(ranked)
        raw_to_center_category = {category: category for category in collapsed}

    categories = sorted(collapsed)
    cluster_counts = {category: 1 for category in categories}
    remaining = num_total_clusters - len(categories)
    if remaining < 0:
        raise RuntimeError(
            f"Collapsed category count {len(categories)} exceeds cluster budget {num_total_clusters}."
        )

    if remaining:
        total = float(sum(collapsed.values()))
        weighted = []
        for category in categories:
            exact = remaining * (collapsed[category] / total)
            base = int(exact)
            cluster_counts[category] += base
            weighted.append((exact - base, collapsed[category], category))
        leftover = num_total_clusters - sum(cluster_counts.values())
        for _fraction, _count, category in sorted(weighted, key=lambda item: (-item[0], -item[1], item[2]))[:leftover]:
            cluster_counts[category] += 1

    return {
        "categories": categories,
        "category_token_counts": {category: int(collapsed[category]) for category in categories},
        "category_cluster_counts": {category: int(cluster_counts[category]) for category in categories},
        "raw_to_center_category": raw_to_center_category,
    }
```

- [ ] **步骤 2：运行 helper 测试，确认通过**

运行：

```bash
python -m unittest "tests/test_make_ct_csd_llada.py" -v
```

预期：

```text
OK
```

## 任务 3：补充 category bank state 测试

**文件：**
- 修改：`tests/test_make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：写 category metadata 失败测试**

把以下测试追加到 `BuilderHelpersTest`：

```python
    def test_build_state_from_cluster_sums_adds_category_metadata(self):
        safe_mean = torch.tensor([0.5, 0.5])
        cluster_sums = torch.tensor([[3.0, 0.0], [0.0, 6.0], [8.0, 0.0]])
        cluster_counts = torch.tensor([3, 2, 4])
        category_plan = {
            "categories": ["cyber", "illegal"],
            "category_token_counts": {"cyber": 2, "illegal": 7},
            "category_cluster_counts": {"cyber": 1, "illegal": 2},
            "raw_to_center_category": {"cyber": "cyber", "illegal": "illegal"},
        }

        state = builder.build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=31,
            max_response_len=128,
            num_total_clusters=3,
            method="category_ct_csd",
            category_key="semantic_category",
            center_categories=["cyber", "illegal", "illegal"],
            center_cluster_ids=[0, 0, 1],
            category_plan=category_plan,
        )

        self.assertEqual(state["format"], "ct_csd_v1")
        self.assertEqual(state["config"]["method"], "category_ct_csd")
        self.assertEqual(state["config"]["category_key"], "semantic_category")
        self.assertEqual(state["categories"], ["cyber", "illegal"])
        self.assertEqual(state["center_categories"], ["cyber", "illegal", "illegal"])
        self.assertEqual(state["center_category_ids"].tolist(), [0, 1, 1])
        self.assertEqual(state["cluster_ids"].tolist(), [0, 0, 1])
        self.assertEqual(state["global_cluster_ids"].tolist(), [0, 1, 2])
        self.assertEqual(state["config"]["category_token_counts"], {"cyber": 2, "illegal": 7})
        self.assertEqual(state["config"]["category_cluster_counts"], {"cyber": 1, "illegal": 2})
```

- [ ] **步骤 2：运行聚焦测试，确认失败**

运行：

```bash
python -m unittest "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_build_state_from_cluster_sums_adds_category_metadata" -v
```

预期：

```text
ERROR: TypeError，因为 build_bank_state_from_cluster_sums 还不接受 method/category metadata 参数
```

## 任务 4：扩展 bank state builder

**文件：**
- 修改：`utils/make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：替换 `build_bank_state_from_cluster_sums` 的签名和函数体**

用以下代码替换现有 `build_bank_state_from_cluster_sums`：

```python
def build_bank_state_from_cluster_sums(
    safe_mean: torch.Tensor,
    cluster_sums: torch.Tensor,
    cluster_counts: torch.Tensor,
    target_layer: int,
    max_response_len: int,
    num_total_clusters: int,
    method: str = "ct_csd",
    category_key: str | None = None,
    center_categories: list[str] | None = None,
    center_cluster_ids: list[int] | None = None,
    category_plan: dict | None = None,
) -> dict:
    if torch.any(cluster_counts <= 0):
        raise RuntimeError(f"Empty CT-CSD cluster detected: counts={cluster_counts.tolist()}")
    centers = cluster_sums / cluster_counts.unsqueeze(-1).float()
    vectors = centers - safe_mean.unsqueeze(0)
    state = {
        "format": "ct_csd_v1",
        "model_family": "llada",
        "target_layer": int(target_layer),
        "safe_anchor_type": "sample_balanced_global_safe_mean",
        "safe_mean": safe_mean.float().cpu(),
        "centers": centers.float().cpu(),
        "centers_unit": unit(centers).float().cpu(),
        "vectors": vectors.float().cpu(),
        "vectors_unit": unit(vectors).float().cpu(),
        "cluster_ids": torch.arange(num_total_clusters, dtype=torch.long),
        "global_cluster_ids": torch.arange(num_total_clusters, dtype=torch.long),
        "cluster_sizes": cluster_counts.long().cpu(),
        "config": {
            "method": method,
            "num_total_clusters": int(num_total_clusters),
            "cluster_feature": "l2_normalized_hidden",
            "category_key": category_key,
            "max_response_len": int(max_response_len),
            "exclude_special_tokens": True,
            "exclude_blank_tokens": True,
            "exclude_punctuation": False,
        },
        "mil": {
            "enabled": False,
            "probe_path": None,
            "probe_threshold": None,
            "top_q_ratio": None,
        },
    }

    if center_categories is not None:
        if len(center_categories) != num_total_clusters:
            raise ValueError("center_categories length must match num_total_clusters")
        categories = list(category_plan["categories"]) if category_plan is not None else sorted(set(center_categories))
        category_to_id = {category: idx for idx, category in enumerate(categories)}
        state["categories"] = categories
        state["center_categories"] = list(center_categories)
        state["center_category_ids"] = torch.tensor(
            [category_to_id[category] for category in center_categories],
            dtype=torch.long,
        )
        if center_cluster_ids is None:
            raise ValueError("center_cluster_ids is required when center_categories is provided")
        if len(center_cluster_ids) != num_total_clusters:
            raise ValueError("center_cluster_ids length must match num_total_clusters")
        state["cluster_ids"] = torch.tensor(center_cluster_ids, dtype=torch.long)
        state["config"]["category_token_counts"] = dict(category_plan["category_token_counts"])
        state["config"]["category_cluster_counts"] = dict(category_plan["category_cluster_counts"])
        state["config"]["raw_to_center_category"] = dict(category_plan["raw_to_center_category"])

    return state
```

- [ ] **步骤 2：运行 state builder 测试**

运行：

```bash
python -m unittest "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_build_state_from_cluster_sums_uses_global_safe_mean" "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_build_state_from_cluster_sums_adds_category_metadata" -v
```

预期：

```text
OK
```

## 任务 5：补充 category KMeans 流程测试

**文件：**
- 修改：`tests/test_make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：添加 fake KMeans 和 category flow 测试**

把以下 fake class 放到 `_NearestAxisKMeans` 附近：

```python
class _RecordingCategoryKMeans:
    def __init__(self, category):
        self.category = category
        self.partial_fit_batches = []

    def partial_fit(self, features):
        self.partial_fit_batches.append(features.copy())
        return self

    def predict(self, features):
        return [0 if row[0] >= row[1] else 1 for row in features]
```

把以下测试追加到 `BuilderHelpersTest`：

```python
    def test_fit_category_minibatch_kmeans_fits_each_category_and_safe_mean(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            num_total_clusters=3,
            kmeans_batch_size=16,
            seed=42,
            category_key="semantic_category",
        )
        plan = {
            "categories": ["fraud", "illegal"],
            "category_token_counts": {"fraud": 2, "illegal": 2},
            "category_cluster_counts": {"fraud": 1, "illegal": 2},
            "raw_to_center_category": {"fraud": "fraud", "illegal": "illegal"},
        }
        h1 = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        s1 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        h2 = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        s2 = torch.tensor([[4.0, 0.0], [0.0, 4.0]])

        with patch.object(builder.random, "choice", side_effect=["refusal1", "refusal2"]):
            with patch.object(builder, "iter_valid_sample_tokens", side_effect=[(h1, s1.mean(dim=0)), (h2, s2.mean(dim=0))]):
                kmeans_by_category, safe_mean, skipped = builder.fit_category_minibatch_kmeans(
                    _Model(),
                    _SequenceTokenizer(),
                    [
                        {"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"},
                        {"prompt": "prompt", "response": "harmful2", "semantic_category": "fraud"},
                    ],
                    ["refusal1", "refusal2"],
                    args,
                    torch.device("cpu"),
                    plan,
                    kmeans_factory=lambda category, n_clusters: _RecordingCategoryKMeans(category),
                )

        self.assertEqual(skipped, 0)
        self.assertTrue(torch.allclose(safe_mean, torch.tensor([1.5, 1.5])))
        self.assertEqual(sorted(kmeans_by_category), ["fraud", "illegal"])
        self.assertEqual(len(kmeans_by_category["illegal"].partial_fit_batches), 1)
        self.assertEqual(len(kmeans_by_category["fraud"].partial_fit_batches), 1)

    def test_accumulate_category_cluster_sums_uses_global_offsets(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            num_total_clusters=3,
            category_key="semantic_category",
        )
        plan = {
            "categories": ["fraud", "illegal"],
            "category_token_counts": {"fraud": 2, "illegal": 2},
            "category_cluster_counts": {"fraud": 1, "illegal": 2},
            "raw_to_center_category": {"fraud": "fraud", "illegal": "illegal"},
        }
        h1 = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        h2 = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

        with patch.object(builder.random, "choice", side_effect=["refusal1", "refusal2"]):
            with patch.object(builder, "iter_valid_sample_tokens", side_effect=[(h1, torch.zeros(2)), (h2, torch.zeros(2))]):
                cluster_sums, cluster_counts, skipped, center_categories, center_cluster_ids, cluster_category_counts = (
                    builder.accumulate_category_cluster_sums(
                        _Model(),
                        _SequenceTokenizer(),
                        [
                            {"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"},
                            {"prompt": "prompt", "response": "harmful2", "semantic_category": "fraud"},
                        ],
                        ["refusal1", "refusal2"],
                        {
                            "fraud": _RecordingCategoryKMeans("fraud"),
                            "illegal": _NearestAxisKMeans(),
                        },
                        args,
                        torch.device("cpu"),
                        plan,
                    )
                )

        self.assertEqual(skipped, 0)
        self.assertEqual(center_categories, ["fraud", "illegal", "illegal"])
        self.assertEqual(center_cluster_ids, [0, 0, 1])
        self.assertTrue(torch.equal(cluster_sums, torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])))
        self.assertEqual(cluster_counts.tolist(), [2, 1, 1])
        self.assertEqual(cluster_category_counts, {"fraud": [2], "illegal": [1, 1]})
```

- [ ] **步骤 2：运行 category flow 测试，确认失败**

运行：

```bash
python -m unittest "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_fit_category_minibatch_kmeans_fits_each_category_and_safe_mean" "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_accumulate_category_cluster_sums_uses_global_offsets" -v
```

预期：

```text
ERROR: AttributeError，因为 category KMeans 函数还不存在
```

## 任务 6：实现 category KMeans 流程

**文件：**
- 修改：`utils/make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：添加 category offset helper**

把以下 helper 添加到 `make_category_cluster_plan` 后：

```python
def category_cluster_offsets(category_plan: dict) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for category in category_plan["categories"]:
        offsets[category] = cursor
        cursor += int(category_plan["category_cluster_counts"][category])
    return offsets
```

- [ ] **步骤 2：添加 category fit 函数**

把以下函数添加到 `fit_minibatch_kmeans` 后：

```python
def fit_category_minibatch_kmeans(
    model,
    tokenizer,
    harmful: list[dict],
    refusals: list[str],
    args,
    device: torch.device,
    category_plan: dict,
    kmeans_factory=None,
):
    if kmeans_factory is None:
        def kmeans_factory(_category: str, n_clusters: int):
            return MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=args.kmeans_batch_size,
                random_state=args.seed,
                n_init="auto",
            )

    kmeans_by_category = {
        category: kmeans_factory(category, int(category_plan["category_cluster_counts"][category]))
        for category in category_plan["categories"]
    }
    buffers: dict[str, list[torch.Tensor]] = {category: [] for category in category_plan["categories"]}
    fitted_batches: dict[str, int] = {category: 0 for category in category_plan["categories"]}
    safe_sum = None
    safe_count = 0
    skipped = 0

    for idx, sample in enumerate(tqdm(harmful, desc="Stage 3 pass 1: fit category clusters")):
        raw_category = resolve_category(sample, getattr(args, "category_key", None))
        category = category_plan["raw_to_center_category"].get(raw_category, raw_category)
        if category not in kmeans_by_category:
            skipped += 1
            continue
        refusal = random.choice(refusals)
        try:
            item = iter_valid_sample_tokens(model, tokenizer, sample, refusal, args, device)
        except RuntimeError as exc:
            print(f"[{idx}] category pass 1 forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as exc:
            print(f"[{idx}] category pass 1 sample failed: {exc}")
            skipped += 1
            continue
        if item is None:
            skipped += 1
            continue

        h_tokens, safe_mean_i = item
        safe_sum = safe_mean_i if safe_sum is None else safe_sum + safe_mean_i
        safe_count += 1

        features = unit(h_tokens).cpu()
        required = int(category_plan["category_cluster_counts"][category])
        if fitted_batches[category] == 0:
            buffers[category].append(features)
            buffered = torch.cat(buffers[category], dim=0)
            if buffered.shape[0] < required:
                continue
            kmeans_by_category[category].partial_fit(buffered.numpy())
            buffers[category].clear()
        else:
            kmeans_by_category[category].partial_fit(features.numpy())
        fitted_batches[category] += 1

    missing = [category for category, count in fitted_batches.items() if count == 0]
    if missing or safe_count == 0:
        raise RuntimeError(
            f"No samples processed successfully for category clusters: missing={missing}, safe_count={safe_count}"
        )
    return kmeans_by_category, safe_sum / safe_count, skipped
```

- [ ] **步骤 3：添加 category accumulate 函数**

把以下函数添加到 `accumulate_cluster_sums` 后：

```python
def accumulate_category_cluster_sums(
    model,
    tokenizer,
    harmful: list[dict],
    refusals: list[str],
    kmeans_by_category: dict,
    args,
    device: torch.device,
    category_plan: dict,
):
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    if d_model is None:
        raise RuntimeError("Unable to infer hidden dimension from model config.")

    offsets = category_cluster_offsets(category_plan)
    total_clusters = sum(int(category_plan["category_cluster_counts"][category]) for category in category_plan["categories"])
    cluster_sums = torch.zeros(total_clusters, int(d_model), dtype=torch.float32)
    cluster_counts = torch.zeros(total_clusters, dtype=torch.long)
    center_categories: list[str] = []
    center_cluster_ids: list[int] = []
    cluster_category_counts: dict[str, list[int]] = {}
    for category in category_plan["categories"]:
        n_clusters = int(category_plan["category_cluster_counts"][category])
        center_categories.extend([category] * n_clusters)
        center_cluster_ids.extend(range(n_clusters))
        cluster_category_counts[category] = [0] * n_clusters

    skipped = 0
    for idx, sample in enumerate(tqdm(harmful, desc="Stage 3 pass 2: accumulate category clusters")):
        raw_category = resolve_category(sample, getattr(args, "category_key", None))
        category = category_plan["raw_to_center_category"].get(raw_category, raw_category)
        if category not in kmeans_by_category:
            skipped += 1
            continue
        refusal = random.choice(refusals)
        try:
            item = iter_valid_sample_tokens(model, tokenizer, sample, refusal, args, device)
        except RuntimeError as exc:
            print(f"[{idx}] category pass 2 forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as exc:
            print(f"[{idx}] category pass 2 sample failed: {exc}")
            skipped += 1
            continue
        if item is None:
            skipped += 1
            continue

        h_tokens, _safe_mean_i = item
        labels = torch.tensor(kmeans_by_category[category].predict(unit(h_tokens).cpu().numpy()), dtype=torch.long)
        offset = int(offsets[category])
        n_clusters = int(category_plan["category_cluster_counts"][category])
        for local_cluster_id in range(n_clusters):
            mask = labels == local_cluster_id
            if mask.any():
                global_cluster_id = offset + local_cluster_id
                count = int(mask.sum().item())
                cluster_sums[global_cluster_id] += h_tokens[mask].sum(dim=0)
                cluster_counts[global_cluster_id] += count
                cluster_category_counts[category][local_cluster_id] += count

    return cluster_sums, cluster_counts, skipped, center_categories, center_cluster_ids, cluster_category_counts
```

- [ ] **步骤 4：运行 category flow 测试**

运行：

```bash
python -m unittest "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_fit_category_minibatch_kmeans_fits_each_category_and_safe_mean" "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_accumulate_category_cluster_sums_uses_global_offsets" -v
```

预期：

```text
OK
```

## 任务 7：补充 `category_ct_csd` CLI 测试

**文件：**
- 修改：`tests/test_make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：写 CLI metadata 失败测试**

把以下测试追加到 `BuilderHelpersTest`：

```python
    def test_main_writes_category_ct_csd_bank_with_cli_metadata(self):
        class _LoadableModel(_Model):
            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_eval = True
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harmful_path = root / "harmful.json"
            refusals_path = root / "refusals.txt"
            output_dir = root / "out"
            harmful_path.write_text(
                '['
                '{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"},'
                '{"prompt": "prompt", "response": "harmful2", "semantic_category": "fraud"}'
                ']',
                encoding="utf-8",
            )
            refusals_path.write_text("refusal1\n", encoding="utf-8")
            category_plan = {
                "categories": ["fraud", "illegal"],
                "category_token_counts": {"fraud": 2, "illegal": 2},
                "category_cluster_counts": {"fraud": 1, "illegal": 1},
                "raw_to_center_category": {"fraud": "fraud", "illegal": "illegal"},
            }

            with patch.object(builder.AutoTokenizer, "from_pretrained", return_value=_SequenceTokenizer()):
                with patch.object(builder.AutoModel, "from_pretrained", return_value=_LoadableModel()):
                    with patch.object(builder, "count_response_tokens_by_category", return_value={"illegal": 2, "fraud": 2}) as count_mock:
                        with patch.object(builder, "make_category_cluster_plan", return_value=category_plan) as plan_mock:
                            with patch.object(
                                builder,
                                "fit_category_minibatch_kmeans",
                                return_value=({"fraud": object(), "illegal": object()}, torch.tensor([0.5, 0.5]), 1),
                            ) as fit_mock:
                                with patch.object(
                                    builder,
                                    "accumulate_category_cluster_sums",
                                    return_value=(
                                        torch.tensor([[2.0, 0.0], [0.0, 6.0]]),
                                        torch.tensor([2, 3]),
                                        0,
                                        ["fraud", "illegal"],
                                        [0, 0],
                                        {"fraud": [2], "illegal": [3]},
                                    ),
                                ) as accumulate_mock:
                                    builder.main(
                                        [
                                            "--model_path",
                                            "dummy-model",
                                            "--harmful_json",
                                            str(harmful_path),
                                            "--refusals_txt",
                                            str(refusals_path),
                                            "--output_dir",
                                            str(output_dir),
                                            "--target_layer",
                                            "31",
                                            "--max_response_len",
                                            "128",
                                            "--max_total_len",
                                            "2048",
                                            "--method",
                                            "category_ct_csd",
                                            "--num_total_clusters",
                                            "2",
                                            "--kmeans_batch_size",
                                            "16",
                                            "--category_key",
                                            "semantic_category",
                                            "--device",
                                            "cpu",
                                            "--seed",
                                            "42",
                                        ]
                                    )

            state = torch.load(output_dir / "ct_csd_bank.pt", map_location="cpu", weights_only=True)

        self.assertTrue(count_mock.called)
        self.assertTrue(plan_mock.called)
        self.assertTrue(fit_mock.called)
        self.assertTrue(accumulate_mock.called)
        self.assertEqual(state["format"], "ct_csd_v1")
        self.assertEqual(state["config"]["method"], "category_ct_csd")
        self.assertEqual(state["config"]["category_key"], "semantic_category")
        self.assertEqual(state["center_categories"], ["fraud", "illegal"])
        self.assertEqual(state["cluster_ids"].tolist(), [0, 0])
        self.assertEqual(state["config"]["skipped_pass1"], 1)
        self.assertEqual(state["config"]["skipped_pass2"], 0)
```

- [ ] **步骤 2：运行 CLI 测试，确认失败**

运行：

```bash
python -m unittest "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_main_writes_category_ct_csd_bank_with_cli_metadata" -v
```

预期：

```text
ERROR: argument --method: invalid choice: 'category_ct_csd'
```

## 任务 8：把 `category_ct_csd` 接入 CLI

**文件：**
- 修改：`utils/make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：扩展 CLI 参数**

在 `main` 中把：

```python
parser.add_argument("--method", choices=["ct_csd"], default="ct_csd")
```

替换为：

```python
parser.add_argument("--method", choices=["ct_csd", "category_ct_csd"], default="ct_csd")
parser.add_argument("--category_key", default="semantic_category")
```

- [ ] **步骤 2：替换 `main` 中 fit / accumulate / state creation 逻辑**

用以下代码替换当前 KMeans fit、accumulate 和 state 创建块：

```python
    if args.method == "category_ct_csd":
        category_token_counts = count_response_tokens_by_category(tokenizer, harmful, args)
        category_plan = make_category_cluster_plan(category_token_counts, args.num_total_clusters)
        print(f"Category token counts: {category_plan['category_token_counts']}")
        print(f"Category cluster counts: {category_plan['category_cluster_counts']}")
        kmeans_by_category, safe_mean, skipped_pass1 = fit_category_minibatch_kmeans(
            model,
            tokenizer,
            harmful,
            refusals,
            args,
            device,
            category_plan,
        )
        (
            cluster_sums,
            cluster_counts,
            skipped_pass2,
            center_categories,
            center_cluster_ids,
            cluster_category_counts,
        ) = accumulate_category_cluster_sums(
            model,
            tokenizer,
            harmful,
            refusals,
            kmeans_by_category,
            args,
            device,
            category_plan,
        )
        state = build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=args.target_layer,
            max_response_len=args.max_response_len,
            num_total_clusters=args.num_total_clusters,
            method=args.method,
            category_key=args.category_key,
            center_categories=center_categories,
            center_cluster_ids=center_cluster_ids,
            category_plan=category_plan,
        )
        state["config"]["cluster_category_counts"] = cluster_category_counts
    else:
        kmeans, safe_mean, skipped_pass1 = fit_minibatch_kmeans(model, tokenizer, harmful, refusals, args, device)
        cluster_sums, cluster_counts, skipped_pass2 = accumulate_cluster_sums(
            model, tokenizer, harmful, refusals, kmeans, args, device
        )
        state = build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=args.target_layer,
            max_response_len=args.max_response_len,
            num_total_clusters=args.num_total_clusters,
            method=args.method,
            category_key=None,
        )
```

保留现有 metadata 写入：

```python
    state["config"]["harmful_json"] = str(harmful_path)
    state["config"]["refusals_txt"] = str(refusals_path)
    state["config"]["skipped_pass1"] = int(skipped_pass1)
    state["config"]["skipped_pass2"] = int(skipped_pass2)
    state["config"]["seed"] = int(args.seed)
```

- [ ] **步骤 3：运行 CLI metadata 测试**

运行：

```bash
python -m unittest "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_main_writes_category_ct_csd_bank_with_cli_metadata" -v
```

预期：

```text
OK
```

## 任务 9：添加 bank summary 输出

**文件：**
- 修改：`utils/make_ct_csd_llada.py`
- 测试：`tests/test_make_ct_csd_llada.py`

- [ ] **步骤 1：写 summary writer 失败测试**

把以下测试追加到 `BuilderHelpersTest`：

```python
    def test_write_bank_summary_outputs_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            state = {
                "config": {
                    "method": "category_ct_csd",
                    "num_total_clusters": 2,
                    "category_cluster_counts": {"fraud": 1, "illegal": 1},
                    "category_token_counts": {"fraud": 2, "illegal": 3},
                    "cluster_category_counts": {"fraud": [2], "illegal": [3]},
                },
                "cluster_sizes": torch.tensor([2, 3]),
                "center_categories": ["fraud", "illegal"],
                "cluster_ids": torch.tensor([0, 0]),
            }

            builder.write_bank_summary(output_dir, state)

            summary = json.loads((output_dir / "ct_csd_bank_summary.json").read_text(encoding="utf-8"))
            markdown = (output_dir / "cluster_category_distribution.md").read_text(encoding="utf-8")

        self.assertEqual(summary["method"], "category_ct_csd")
        self.assertEqual(summary["num_total_clusters"], 2)
        self.assertEqual(summary["cluster_sizes"], [2, 3])
        self.assertIn("| fraud | 0 | 2 |", markdown)
        self.assertIn("| illegal | 0 | 3 |", markdown)
```

如果 `tests/test_make_ct_csd_llada.py` 顶部还没有 `import json`，补上：

```python
import json
```

- [ ] **步骤 2：运行 summary 测试，确认失败**

运行：

```bash
python -m unittest "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_write_bank_summary_outputs_json_and_markdown" -v
```

预期：

```text
ERROR: AttributeError，因为 write_bank_summary 还不存在
```

- [ ] **步骤 3：添加 summary writer 函数**

把以下函数添加到 `utils/make_ct_csd_llada.py` 的 `main` 前：

```python
def write_bank_summary(output_dir: Path, state: dict) -> None:
    cluster_sizes = [int(x) for x in state["cluster_sizes"].tolist()]
    center_categories = list(state.get("center_categories", ["global"] * len(cluster_sizes)))
    cluster_ids_tensor = state.get("cluster_ids", torch.arange(len(cluster_sizes), dtype=torch.long))
    cluster_ids = [int(x) for x in cluster_ids_tensor.tolist()]
    config = dict(state.get("config", {}))
    rows = []
    for idx, size in enumerate(cluster_sizes):
        rows.append(
            {
                "global_cluster_id": idx,
                "category": center_categories[idx],
                "local_cluster_id": cluster_ids[idx],
                "cluster_size": size,
            }
        )

    summary = {
        "method": config.get("method"),
        "num_total_clusters": int(config.get("num_total_clusters", len(cluster_sizes))),
        "cluster_sizes": cluster_sizes,
        "category_token_counts": config.get("category_token_counts", {}),
        "category_cluster_counts": config.get("category_cluster_counts", {}),
        "cluster_category_counts": config.get("cluster_category_counts", {}),
        "clusters": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ct_csd_bank_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    lines = [
        "# CT-CSD Cluster Category Distribution",
        "",
        f"method: {summary['method']}",
        f"num_total_clusters: {summary['num_total_clusters']}",
        "",
        "| category | local_cluster_id | cluster_size |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['category']} | {row['local_cluster_id']} | {row['cluster_size']} |")
    lines.append("")
    (output_dir / "cluster_category_distribution.md").write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **步骤 4：保存 bank 后调用 summary writer**

在 `main` 中的：

```python
    torch.save(state, out_path)
```

后面添加：

```python
    write_bank_summary(output_dir, state)
```

- [ ] **步骤 5：运行 summary 和 CLI 测试**

运行：

```bash
python -m unittest "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_write_bank_summary_outputs_json_and_markdown" "tests.test_make_ct_csd_llada.BuilderHelpersTest.test_main_writes_category_ct_csd_bank_with_cli_metadata" -v
```

预期：

```text
OK
```

## 任务 10：为 `CTCSDBank` 添加 category 诊断

**文件：**
- 修改：`utils/ct_csd_bank.py`
- 修改：`tests/test_ct_csd_bank.py`
- 测试：`tests/test_ct_csd_bank.py`

- [ ] **步骤 1：写 category diagnostics 失败测试**

把以下测试追加到 `tests/test_ct_csd_bank.py` 的 `CTCSDBankTest` 中：

```python
    def test_diagnostics_aggregates_optional_center_categories(self):
        state = self._state()
        state["categories"] = ["fraud", "illegal"]
        state["center_categories"] = ["fraud", "illegal"]
        bank = CTCSDBank.from_state_dict(state, device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0], [0.0, 3.0]]])

        _ = bank.alignment(hidden, theta=2.5, record=True)
        diagnostics = bank.diagnostics()

        self.assertEqual(diagnostics["center_categories"], ["fraud", "illegal"])
        self.assertEqual(diagnostics["category_route_count"], {"fraud": 1, "illegal": 2})
        self.assertEqual(diagnostics["category_active_count"], {"fraud": 1, "illegal": 2})
```

- [ ] **步骤 2：运行 diagnostics 测试，确认失败**

运行：

```bash
python -m unittest "tests.test_ct_csd_bank.CTCSDBankTest.test_diagnostics_aggregates_optional_center_categories" -v
```

预期：

```text
FAIL，因为 diagnostics 还没有输出 center_categories / category counts
```

- [ ] **步骤 3：在 `CTCSDBank.__init__` 中保存可选 category metadata**

在 `self.cluster_sizes = ...` 后添加：

```python
        raw_center_categories = state.get("center_categories")
        if raw_center_categories is None:
            self.center_categories = None
            self.categories = None
        else:
            self.center_categories = [str(category) for category in raw_center_categories]
            if len(self.center_categories) != self.cluster_sizes.shape[0]:
                raise ValueError("center_categories length must match number of centers")
            self.categories = [str(category) for category in state.get("categories", sorted(set(self.center_categories)))]
```

- [ ] **步骤 4：扩展 `diagnostics()` 的 category 聚合**

在 `diagnostics()` 中先构造 `result`，再按需添加 category 字段：

```python
        result = {
            "format": self.format,
            "target_layer": self.target_layer,
            "num_clusters": self.num_clusters,
            "cluster_sizes": [int(x) for x in self.cluster_sizes.tolist()],
            "route_count": [int(x) for x in self.route_count.tolist()],
            "active_count": [int(x) for x in self.active_count.tolist()],
            "total_routed": total_routed,
            "total_active": total_active,
            "activation_rate": activation_rate,
            "route_time_sec": float(self.route_time_sec),
        }
        if self.center_categories is not None:
            route_by_category = {category: 0 for category in self.categories}
            active_by_category = {category: 0 for category in self.categories}
            for idx, category in enumerate(self.center_categories):
                route_by_category[category] = route_by_category.get(category, 0) + int(self.route_count[idx].item())
                active_by_category[category] = active_by_category.get(category, 0) + int(self.active_count[idx].item())
            result["center_categories"] = list(self.center_categories)
            result["category_route_count"] = route_by_category
            result["category_active_count"] = active_by_category
        return result
```

删除旧的直接 `return { ... }` 块。

- [ ] **步骤 5：运行 bank 测试**

运行：

```bash
python -m unittest "tests/test_ct_csd_bank.py" -v
```

预期：

```text
OK
```

## 任务 11：运行 CT-CSD 相关单元测试切片

**文件：**
- 仅测试

- [ ] **步骤 1：运行 CT-CSD 相关单元测试**

运行：

```bash
python -m unittest "tests/test_make_ct_csd_llada.py" "tests/test_ct_csd_bank.py" "tests/test_eval_llada_ct_csd_bank.py" -v
```

预期：

```text
OK
```

- [ ] **步骤 2：依赖可用时运行更完整测试**

运行：

```bash
pytest "tests" -q
```

预期：

```text
全部测试通过；如果本地模型或 API 缺失导致环境型测试失败，记录具体失败测试名并继续 Stage 3 smoke 命令。
```

## 任务 12：运行 Stage 3 bank smoke

**文件：**
- 只产出运行时文件
- 读取：`.worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json`
- 读取：`utils/refusals.txt`

- [ ] **步骤 1：从 Stage 2 结果确认 `MSTAR`**

查看 Stage 2 是否已经产出正式 `M*`：

```bash
find "docs" "outputs" -maxdepth 3 -type f | rg "stage2|m_selection|random_k|ct_csd"
```

如果 Stage 2 还没有最终 `M*`，smoke 阶段临时使用：

```bash
MSTAR=8
```

并在记录里写清楚这是 smoke 参数，不是 Stage 3 正式默认簇数。

- [ ] **步骤 2：运行小样本 category bank 构造**

运行：

```bash
MSTAR=8
python "utils/make_ct_csd_llada.py" \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --harmful_json ".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json" \
  --refusals_txt "utils/refusals.txt" \
  --output_dir "outputs/category_ct_csd_llada_m${MSTAR}_smoke" \
  --target_layer 31 \
  --max_response_len 128 \
  --max_samples 16 \
  --method "category_ct_csd" \
  --num_total_clusters "${MSTAR}" \
  --category_key "semantic_category" \
  --device "cuda" \
  --seed 42
```

预期：

```text
Saved CT-CSD bank -> outputs/category_ct_csd_llada_m8_smoke/ct_csd_bank.pt
cluster_sizes=[...]
```

- [ ] **步骤 3：检查 smoke artifacts**

运行：

```bash
python - <<'PY'
import torch
state = torch.load("outputs/category_ct_csd_llada_m8_smoke/ct_csd_bank.pt", map_location="cpu", weights_only=True)
print(state["format"])
print(state["config"]["method"])
print(state["categories"])
print(state["center_categories"])
print(state["cluster_sizes"].tolist())
PY
```

预期：

```text
ct_csd_v1
category_ct_csd
```

继续检查 markdown：

```bash
sed -n '1,120p' "outputs/category_ct_csd_llada_m8_smoke/cluster_category_distribution.md"
```

预期：

```text
Markdown 表格列出每个 global cluster 的 category、local_cluster_id、cluster_size。
```

## 任务 13：构造 Stage 3 full bank

**文件：**
- 只产出运行时文件

- [ ] **步骤 1：使用 Stage 2 的 `M*` 构造完整 `category_ct_csd` bank**

把下面命令中的 `8` 替换为 Stage 2 实际选出的 `M*`：

```bash
MSTAR=8
python "utils/make_ct_csd_llada.py" \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --harmful_json ".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json" \
  --refusals_txt "utils/refusals.txt" \
  --output_dir "outputs/category_ct_csd_llada_m${MSTAR}" \
  --target_layer 31 \
  --max_response_len 128 \
  --method "category_ct_csd" \
  --num_total_clusters "${MSTAR}" \
  --category_key "semantic_category" \
  --device "cuda" \
  --seed 42
```

预期：

```text
Saved CT-CSD bank -> outputs/category_ct_csd_llada_m8/ct_csd_bank.pt
cluster_sizes=[非空正整数列表]
```

- [ ] **步骤 2：验证现有 `CTCSDBank` 能加载该 bank**

运行：

```bash
python - <<'PY'
import torch
from utils.ct_csd_bank import CTCSDBank
path = "outputs/category_ct_csd_llada_m8/ct_csd_bank.pt"
state = torch.load(path, map_location="cpu", weights_only=True)
bank = CTCSDBank.from_state_dict(state, device=torch.device("cpu"))
print(bank.format)
print(bank.num_clusters)
print(bank.diagnostics()["center_categories"])
PY
```

预期：

```text
ct_csd_v1
8
```

## 任务 14：运行 Stage 3 对照评测

**文件：**
- 只产出运行时文件
- 读取：同一 `MSTAR` 的 Stage 2 CT-CSD bank
- 读取：Stage 3 category bank

- [ ] **步骤 1：运行同 `MSTAR` 的 CT-CSD baseline**

使用 Stage 2 相同评测设置。如果 Stage 2 已经有完全相同的输出，直接复用并在 `docs/stage3_category_ct_csd_metrics.md` 记录路径。

```bash
MSTAR=8
python "eval_llada_steering.py" \
  --csv_path "JBB" \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --generated_samples_path "outputs/jbb_dija_ct_csd_m${MSTAR}" \
  --attack_method "DIJA" \
  --sampler "steering" \
  --steering_vector_path "outputs/ct_csd_llada_m${MSTAR}/ct_csd_bank.pt" \
  --target_layer 31 \
  --alignment_threshold 0.0 \
  --steering_overshoot 1.0 \
  --initial_steering_ratio 0.1 \
  --max_refinement_iters 5 \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --dija_mask_counts 128 \
  --device "cuda"
```

预期：

```text
Saved ... results.json
Saved CT-CSD diagnostics to outputs/jbb_dija_ct_csd_m8/ct_csd_diagnostics.json
```

- [ ] **步骤 2：运行 Category-aware CT-CSD**

运行：

```bash
MSTAR=8
python "eval_llada_steering.py" \
  --csv_path "JBB" \
  --model_path "/dev/shm/LLaDA-8B-Instruct" \
  --generated_samples_path "outputs/jbb_dija_category_ct_csd_m${MSTAR}" \
  --attack_method "DIJA" \
  --sampler "steering" \
  --steering_vector_path "outputs/category_ct_csd_llada_m${MSTAR}/ct_csd_bank.pt" \
  --target_layer 31 \
  --alignment_threshold 0.0 \
  --steering_overshoot 1.0 \
  --initial_steering_ratio 0.1 \
  --max_refinement_iters 5 \
  --sampling_steps 128 \
  --mask_length 128 \
  --block_size 128 \
  --dija_mask_counts 128 \
  --device "cuda"
```

预期：

```text
Saved ... results.json
Saved CT-CSD diagnostics to outputs/jbb_dija_category_ct_csd_m8/ct_csd_diagnostics.json
```

## 任务 15：撰写 Stage 3 metrics 文档

**文件：**
- 新增：`docs/stage3_category_ct_csd_metrics.md`

- [ ] **步骤 1：评测产物存在后创建 metrics 报告骨架**

创建 `docs/stage3_category_ct_csd_metrics.md`，并用实际输出填满每个表格单元：

```markdown
# Stage 3 Category-aware CT-CSD Metrics

## Configuration

| Field | Value |
|---|---|
| MSTAR | 8 |
| target_layer | 31 |
| category_key | semantic_category |
| alignment_threshold | 0.0 |
| steering_overshoot | 1.0 |
| initial_steering_ratio | 0.1 |
| max_refinement_iters | 5 |
| sampling_steps | 128 |
| mask_length | 128 |
| block_size | 128 |
| dija_mask_counts | 128 |

## Bank Artifacts

| Method | Bank Path | Summary Path |
|---|---|---|
| CT-CSD | outputs/ct_csd_llada_m8/ct_csd_bank.pt | outputs/ct_csd_llada_m8/ct_csd_bank_summary.json |
| Category-aware CT-CSD | outputs/category_ct_csd_llada_m8/ct_csd_bank.pt | outputs/category_ct_csd_llada_m8/ct_csd_bank_summary.json |

## Main Safety And Quality Metrics

| Method | ASR ↓ | unsafe_count ↓ | refusal_rate | average_output_length | empty_response_count |
|---|---:|---:|---:|---:|---:|
| CT-CSD | 0.000 | 0 | 0.000 | 0.0 | 0 |
| Category-aware CT-CSD | 0.000 | 0 | 0.000 | 0.0 | 0 |

## Steering Diagnostics

| Method | total_routed | total_active | activation_rate | route_time_sec |
|---|---:|---:|---:|---:|
| CT-CSD | 0 | 0 | 0.0000 | 0.0000 |
| Category-aware CT-CSD | 0 | 0 | 0.0000 | 0.0000 |

## Category Route Diagnostics

| Category | route_count | active_count |
|---|---:|---:|
| category-name | 0 | 0 |

## Decision

Category-aware CT-CSD 只有在保持同一 `MSTAR`、不修改 eval 推理路径、并且 `unsafe_count` 优于或不差于 CT-CSD 且诊断稳定时，才进入 Stage 4 作为 no-probe 主方法。
```

- [ ] **步骤 2：把骨架值替换为实测值**

读取诊断和分布文件：

```bash
cat "outputs/jbb_dija_category_ct_csd_m8/ct_csd_diagnostics.json"
cat "outputs/category_ct_csd_llada_m8/cluster_category_distribution.md"
```

预期：

```text
metrics 文档中没有保留零值占位，所有指标均来自 Stage 3 实际运行产物。
```

## 任务 16：最终验证清单

**文件：**
- 只验证

- [ ] **步骤 1：验证单元测试**

运行：

```bash
python -m unittest "tests/test_make_ct_csd_llada.py" "tests/test_ct_csd_bank.py" "tests/test_eval_llada_ct_csd_bank.py" -v
```

预期：

```text
OK
```

- [ ] **步骤 2：验证 Stage 3 bank 产物**

运行：

```bash
test -f "outputs/category_ct_csd_llada_m8/ct_csd_bank.pt"
test -f "outputs/category_ct_csd_llada_m8/ct_csd_bank_summary.json"
test -f "outputs/category_ct_csd_llada_m8/cluster_category_distribution.md"
```

预期：

```text
无输出，退出码为 0。
```

- [ ] **步骤 3：验证 Stage 3 evaluation 产物**

运行：

```bash
test -f "outputs/jbb_dija_category_ct_csd_m8/results.json"
test -f "outputs/jbb_dija_category_ct_csd_m8/ct_csd_diagnostics.json"
test -f "docs/stage3_category_ct_csd_metrics.md"
```

预期：

```text
无输出，退出码为 0。
```

## 自检

**方案覆盖**
- category 字段优先级由 `resolve_category` 覆盖。
- category-aware offline clustering 由 `fit_category_minibatch_kmeans` 和 `accumulate_category_cluster_sums` 覆盖。
- 总 vector budget `M` 由 `make_category_cluster_plan` 保持。
- bank 兼容性通过继续使用 `format == "ct_csd_v1"` 保持。
- 推理阶段不需要 prompt category，因为 `eval_llada_steering.py` 保持不变。
- MIL token probe 被明确排除在 Stage 3 外。

**占位符检查**
- 本计划没有 `TBD`，没有延后实现项，没有“自行补充测试”之类不可执行描述。

**类型一致性**
- `center_categories` 是 `list[str]`。
- `center_category_ids`、`cluster_ids`、`global_cluster_ids`、`cluster_sizes` 均是长度为 `M` 的 tensor。
- `category_plan["category_cluster_counts"]` 的总和必须等于 `num_total_clusters`。
