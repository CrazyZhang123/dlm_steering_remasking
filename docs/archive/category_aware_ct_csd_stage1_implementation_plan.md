# CT-CSD Bank Stage 1 具体实施计划

> **给执行 agent 的要求：** 实施本计划时，必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐项执行。本文使用 checkbox（`- [ ]`）语法跟踪进度。

**目标：** 在 LLaDA 上完成 Stage 1 的 CT-CSD bank 最小闭环：离线构造多局部 steering vectors，推理时按最近 harmful center 做 hard routing，并在与 Stage 0 完全相同的 JBB + DIJA 评测口径下对比。

**架构：** 新增一个轻量 `CTCSDBank` 运行时类，负责 bank 加载、routing、alignment、steering 和诊断统计；新增一个 LLaDA 离线构造脚本，抽取 `target_layer=31` 的 response token hidden states，用 MiniBatchKMeans 聚类 harmful tokens，并以全局 safe mean 构造 local CSD vectors。`eval_llada_steering.py` 只做兼容扩展，保留旧 `steering_vectors.pt` 单向量路径。

**技术栈：** Python 3.10+、PyTorch、Transformers、scikit-learn `MiniBatchKMeans`、`unittest`、本地 `/dev/shm/LLaDA-8B-Instruct`、本地 `/dev/shm/Llama-Guard-4-12B`

---

## 1. 阶段边界

Stage 1 只验证一个新增变量：**先对 harmful token hidden states 聚类，再构造多个局部 CSD vectors**。

本阶段要做：

- 抽取 harmful response token hidden states，只使用 `target_layer = 31`。
- 过滤 special token、padding token、mask token 和空白 decoded token。
- 使用 `cluster_feature = l2_normalized_hidden` 做 KMeans 聚类。
- 固定 `num_total_clusters = 16`。
- safe side 只构造一个 sample-balanced global safe mean。
- 每个 harmful cluster 构造一个 local vector：`center_k - safe_mean`。
- 推理时每个 hidden state route 到最近 center。
- steering 仍使用原始 threshold-gated 公式：`alpha = beta * clamp(score - theta, min=0)`。
- 记录 `cluster_size`、`route_count`、`active_count`、`activation_rate`、`route_time_sec`。
- 完成一组与 Stage 0 Global Sentence-CSD 的同口径对比。

本阶段不做：

- 不使用 category 字段。
- 不实现 Random-K-CSD 对照。
- 不训练 MIL token probe。
- 不做 safe token clustering。
- 不做 soft routing。
- 不做 per-cluster threshold。
- 不迁移 Dream。
- 用户未明确要求时，不执行 `git commit`、`git push`、分支切换等操作。

## 2. 固定对比口径

Stage 0 baseline 已冻结在 `docs/category_aware_ct_csd_stage_progress.md`：

| 项目 | 值 |
|---|---|
| Stage 0 vector | `.worktrees/harmbench-csd-export/outputs/harmbench_csd_llada_full_20260618/steering_vectors.pt` |
| Stage 0 生成结果 | `.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/results.json` |
| Stage 0 judge 结果 | `.worktrees/harmbench-csd-export/outputs/harmbench_jbb_llada_full_dija_cuda1_20260620/llama_guard_results.json` |
| Stage 0 `total_samples` | `100` |
| Stage 0 `unsafe_count` | `74/100` |
| Stage 0 `ASR` | `74.0%` |

Stage 1 必须继承以下评测参数：

| 参数 | 值 |
|---|---:|
| `target_layer` | `31` |
| `num_total_clusters` | `16` |
| `sampling_steps` | `128` |
| `mask_length` | `128` |
| `block_size` | `128` |
| `dija_mask_counts` | `128` |
| `remasking` | `low_confidence` |
| `alignment_threshold` | `0.0` |
| `steering_overshoot` | `1.0` |
| `initial_steering_ratio` | `0.1` |
| `max_refinement_iters` | `5` |

注意：`max_refinement_iters` 必须显式传 `5`，避免和历史 CLI 默认值 `3` 或其他环境默认值混淆。

## 3. 文件结构

新增文件：

- `utils/ct_csd_bank.py`
  - 保存 `CTCSDBank`，负责加载 bank、校验格式、routing、alignment、steering 和诊断统计。
- `utils/make_ct_csd_llada.py`
  - Stage 1 离线构造脚本，只支持 `--method ct_csd`。
- `tests/test_ct_csd_bank.py`
  - 纯 tensor 单元测试，验证 routing、alignment、steering 和诊断统计。
- `tests/test_make_ct_csd_llada.py`
  - helper 测试，验证 token 过滤、路径解析和 bank state 构造。
- `tests/test_eval_llada_ct_csd_bank.py`
  - 兼容性测试，验证旧 `steering_vectors.pt` 和新 `ct_csd_bank.pt` 都能加载。
- `docs/category_aware_ct_csd_stage1_metrics.md`
  - Stage 1 完成实验后写入，记录结果和诊断。

修改文件：

- `eval_llada_steering.py`
  - 增加 CT-CSD bank 加载分支。
  - `_per_token_alignment` 支持 bank alignment。
  - `_build_adaptive_steering_hook` 支持 bank steering。
  - DIJA Phase 2 检测逻辑支持 bank alignment。
  - 生成结束后写出 `ct_csd_diagnostics.json`。
- `docs/category_aware_ct_csd_stage_progress.md`
  - 仅在 Stage 1 实验和 judge 都完成后追加 Stage 1 状态。

不要修改 `utils/__init__.py`。在 `eval_llada_steering.py` 中直接写 `from utils.ct_csd_bank import CTCSDBank`，避免让 `utils` 包初始化时加载更多构造脚本。

## 4. Bank 文件格式

Stage 1 输出 `outputs/ct_csd_llada_m16/ct_csd_bank.pt`，格式固定为：

```python
{
    "format": "ct_csd_v1",
    "model_family": "llada",
    "target_layer": 31,
    "safe_anchor_type": "sample_balanced_global_safe_mean",
    "safe_mean": Tensor[D],
    "centers": Tensor[M, D],
    "centers_unit": Tensor[M, D],
    "vectors": Tensor[M, D],
    "vectors_unit": Tensor[M, D],
    "cluster_ids": Tensor[M],
    "cluster_sizes": Tensor[M],
    "config": {
        "method": "ct_csd",
        "num_total_clusters": 16,
        "cluster_feature": "l2_normalized_hidden",
        "category_key": None,
        "max_response_len": 128,
        "exclude_special_tokens": True,
        "exclude_blank_tokens": True,
        "exclude_punctuation": False
    },
    "mil": {
        "enabled": False,
        "probe_path": None,
        "probe_threshold": None,
        "top_q_ratio": None
    }
}
```

说明：

- `centers` 是每个 harmful token cluster 在原始 hidden space 的均值。
- `vectors = centers - safe_mean`。
- `centers_unit` 只用于 routing。
- `vectors_unit` 只用于 alignment 和 steering。
- `cluster_sizes` 只用于诊断，不参与推理公式。

## 5. 任务 1：新增 `CTCSDBank` 单元测试

**文件：**

- 新增：`tests/test_ct_csd_bank.py`
- 后续新增：`utils/ct_csd_bank.py`

- [ ] **步骤 1：写失败测试**

测试必须覆盖：

- `route(hidden)` 使用最近 center。
- `alignment(hidden)` 使用 route 后的 local vector。
- `steer(hidden, beta, theta)` 只修改 `score > theta` 的 hidden states。
- `alignment(..., record=True)` 更新 `route_count` 和 `active_count`。
- `load(path)` 拒绝错误格式。

建议测试代码：

```python
import tempfile
import unittest
from pathlib import Path

import torch

from utils.ct_csd_bank import CTCSDBank


class CTCSDBankTest(unittest.TestCase):
    def _state(self):
        centers = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        vectors = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        return {
            "format": "ct_csd_v1",
            "model_family": "llada",
            "target_layer": 31,
            "safe_anchor_type": "sample_balanced_global_safe_mean",
            "safe_mean": torch.zeros(2),
            "centers": centers,
            "centers_unit": torch.nn.functional.normalize(centers, dim=-1),
            "vectors": vectors,
            "vectors_unit": torch.nn.functional.normalize(vectors, dim=-1),
            "cluster_ids": torch.tensor([0, 1]),
            "cluster_sizes": torch.tensor([10, 20]),
            "config": {"method": "ct_csd", "num_total_clusters": 2},
            "mil": {"enabled": False, "probe_path": None, "probe_threshold": None, "top_q_ratio": None},
        }

    def test_route_uses_nearest_center(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0]]])

        route = bank.route(hidden)

        self.assertEqual(route.tolist(), [[0, 1]])

    def test_alignment_uses_routed_local_vector(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0]]])

        score = bank.alignment(hidden)

        self.assertTrue(torch.allclose(score, torch.tensor([[3.0, 4.0]]), atol=1e-6))

    def test_steer_only_changes_scores_above_threshold(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[2.0, 0.0], [0.0, 0.25]])

        steered = bank.steer(hidden, beta=1.0, theta=1.0)

        expected = torch.tensor([[1.0, 0.0], [0.0, 0.25]])
        self.assertTrue(torch.allclose(steered, expected, atol=1e-6))

    def test_record_alignment_updates_route_and_active_counts(self):
        bank = CTCSDBank.from_state_dict(self._state(), device=torch.device("cpu"))
        hidden = torch.tensor([[[3.0, 0.1], [0.2, 4.0], [1.0, 0.0]]])

        _ = bank.alignment(hidden, theta=2.5, record=True)
        diagnostics = bank.diagnostics()

        self.assertEqual(diagnostics["route_count"], [2, 1])
        self.assertEqual(diagnostics["active_count"], [1, 1])
        self.assertEqual(diagnostics["total_routed"], 3)
        self.assertEqual(diagnostics["total_active"], 2)
        self.assertAlmostEqual(diagnostics["activation_rate"], 2 / 3)

    def test_load_rejects_wrong_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({"format": "global_sentence_csd"}, path)

            with self.assertRaises(ValueError):
                CTCSDBank.load(path, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行失败测试**

```bash
python -m unittest "tests/test_ct_csd_bank.py"
```

预期：因为 `utils.ct_csd_bank` 尚未存在，测试报 import error。

## 6. 任务 2：实现 `utils/ct_csd_bank.py`

**文件：**

- 新增：`utils/ct_csd_bank.py`

- [ ] **步骤 1：实现最小运行时类**

核心接口：

```python
class CTCSDBank:
    @classmethod
    def from_state_dict(cls, state, device, dtype=torch.float32):
        ...

    @classmethod
    def load(cls, path, device, dtype=torch.float32):
        ...

    def route(self, hidden):
        ...

    def alignment(self, hidden, theta=None, record=False):
        ...

    def steer(self, hidden, beta, theta):
        ...

    def diagnostics(self):
        ...
```

核心逻辑必须是：

```python
h_unit = normalize(hidden)
route_idx = argmax(h_unit @ centers_unit.T)
local_v = vectors_unit[route_idx]
score = (hidden * local_v).sum(dim=-1)
alpha = beta * clamp(score - theta, min=0)
hidden_new = hidden - alpha.unsqueeze(-1) * local_v
```

实现要求：

- 初始化时校验 `format == "ct_csd_v1"`。
- 校验 `centers`、`vectors` 都是二维 tensor。
- 校验 `centers.shape == vectors.shape`。
- 校验 `cluster_sizes` 长度等于 center 数量。
- `route_count` 和 `active_count` 保存在 CPU tensor，避免 GPU 内存常驻增长。
- `route_time_sec` 使用 `time.perf_counter()` 累加。
- `alignment(record=True, theta=...)` 负责统计 route 和 active；`record=False` 不改变诊断。

- [ ] **步骤 2：验证 bank 测试通过**

```bash
python -m unittest "tests/test_ct_csd_bank.py"
```

预期：`Ran 5 tests`，结果为 `OK`。

## 7. 任务 3：新增 LLaDA CT-CSD 构造脚本 helper 测试

**文件：**

- 新增：`tests/test_make_ct_csd_llada.py`
- 后续新增：`utils/make_ct_csd_llada.py`

- [ ] **步骤 1：写失败测试**

测试必须覆盖：

- `keep_response_token(tokenizer, token_id)` 会过滤 special token 和空白 token。
- `resolve_path(path, cwd)` 优先解析当前工作目录相对路径。
- `build_bank_state_from_cluster_sums(...)` 会用 global safe mean 构造 `centers` 和 `vectors`。

建议测试代码：

```python
import tempfile
import unittest
from pathlib import Path

import torch

from utils import make_ct_csd_llada as builder


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    mask_token_id = 126336
    all_special_ids = [0, 1, 2, 126336]

    def decode(self, token_ids, skip_special_tokens=False):
        value = token_ids[0]
        return {
            3: "word",
            4: " ",
            5: "\n",
            6: ".",
        }.get(value, "x")


class BuilderHelpersTest(unittest.TestCase):
    def test_keep_response_token_filters_special_and_blank_tokens(self):
        tokenizer = _Tokenizer()

        self.assertFalse(builder.keep_response_token(tokenizer, 0))
        self.assertFalse(builder.keep_response_token(tokenizer, 2))
        self.assertFalse(builder.keep_response_token(tokenizer, 126336))
        self.assertFalse(builder.keep_response_token(tokenizer, 4))
        self.assertFalse(builder.keep_response_token(tokenizer, 5))
        self.assertTrue(builder.keep_response_token(tokenizer, 3))
        self.assertTrue(builder.keep_response_token(tokenizer, 6))

    def test_resolve_path_prefers_cwd_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "data.json"
            target.write_text("[]", encoding="utf-8")

            resolved = builder.resolve_path("data.json", root)

        self.assertEqual(resolved, target)

    def test_build_state_from_cluster_sums_uses_global_safe_mean(self):
        safe_mean = torch.tensor([0.5, 0.5])
        cluster_sums = torch.tensor([[3.0, 0.0], [0.0, 6.0]])
        cluster_counts = torch.tensor([3, 2])

        state = builder.build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=31,
            max_response_len=128,
            num_total_clusters=2,
        )

        self.assertEqual(state["format"], "ct_csd_v1")
        self.assertTrue(torch.allclose(state["centers"], torch.tensor([[1.0, 0.0], [0.0, 3.0]])))
        self.assertTrue(torch.allclose(state["vectors"], torch.tensor([[0.5, -0.5], [-0.5, 2.5]])))
        self.assertEqual(state["cluster_sizes"].tolist(), [3, 2])
        self.assertEqual(state["config"]["method"], "ct_csd")
        self.assertEqual(state["config"]["num_total_clusters"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行失败测试**

```bash
python -m unittest "tests/test_make_ct_csd_llada.py"
```

预期：因为 `utils/make_ct_csd_llada.py` 尚未存在，测试报 import error。

## 8. 任务 4：实现 `utils/make_ct_csd_llada.py`

**文件：**

- 新增：`utils/make_ct_csd_llada.py`

- [ ] **步骤 1：实现基础 helper**

必须包含：

- `set_seed(seed)`
- `resolve_path(raw_path, cwd=None)`
- `load_refusals(path)`
- `load_harmful_data(path)`
- `keep_response_token(tokenizer, token_id)`
- `build_sequence(tokenizer, prompt, response, max_response_len)`
- `extract_target_layer_tokens(model, input_ids, response_start, target_layer, device)`
- `unit(x)`
- `build_bank_state_from_cluster_sums(...)`

实现要求：

- `load_harmful_data` 兼容 list 和嵌套 list。
- `build_sequence` 沿用 `utils/make_csd_llada.py` 的 chat template 口径。
- `extract_target_layer_tokens` 只注册 `target_layer` 一个 hook，不保存所有层。
- `keep_response_token` 保留标点，不在 Stage 1 额外过滤 punctuation。

- [ ] **步骤 2：实现两遍流式构造**

不要把全量 HarmBench token hidden states 一次性堆进内存。使用两遍流程：

1. 第一遍：抽取每条 harmful response 的有效 token hidden states，做 L2 normalize 后 `MiniBatchKMeans.partial_fit`；同时累计每条 safe refusal response 的 sample mean，用于构造 global safe mean。
2. 第二遍：重新抽取 harmful token hidden states，用训练好的 KMeans `predict`，按 cluster 累计原始 hidden sum 和 count。
3. 最后：`center_k = cluster_sum_k / cluster_count_k`，`vector_k = center_k - safe_mean`。

核心实现约束：

- `MiniBatchKMeans(n_clusters=args.num_total_clusters, batch_size=args.kmeans_batch_size, random_state=args.seed)`。
- `features = unit(h_tokens).numpy()` 只用于 KMeans。
- cluster sum 必须累计原始 hidden，不累计 normalized feature。
- 如果任意 cluster count 为 0，直接报错；Stage 1 不做合并空 cluster。
- 对 CUDA forward 失败的样本，打印样本 index，`torch.cuda.empty_cache()` 后跳过。

- [ ] **步骤 3：实现 CLI**

默认参数：

```text
--model_path /dev/shm/LLaDA-8B-Instruct
--harmful_json .worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json
--refusals_txt utils/refusals.txt
--output_dir outputs/ct_csd_llada_m16
--target_layer 31
--max_response_len 128
--max_total_len 2048
--method ct_csd
--num_total_clusters 16
--kmeans_batch_size 4096
--device cuda
--seed 42
```

必须支持 `--max_samples`，用于 smoke run。

- [ ] **步骤 4：验证 helper 测试通过**

```bash
python -m unittest "tests/test_make_ct_csd_llada.py"
```

预期：`Ran 3 tests`，结果为 `OK`。

## 9. 任务 5：新增 `eval_llada_steering.py` 兼容性测试

**文件：**

- 新增：`tests/test_eval_llada_ct_csd_bank.py`
- 后续修改：`eval_llada_steering.py`

- [ ] **步骤 1：写失败测试**

测试必须覆盖：

- 传入 `format == "ct_csd_v1"` 的 `ct_csd_bank.pt` 时，`harness.steering_bank` 非空，`harness.steering_vector` 为空。
- 传入旧格式 `{"layer_31": tensor}` 时，`harness.steering_vector` 非空，`harness.steering_bank` 为空。

测试中 mock：

- `accelerate.Accelerator`
- `AutoModel.from_pretrained`
- `AutoTokenizer.from_pretrained`

不加载真实模型。

- [ ] **步骤 2：运行失败测试**

```bash
python -m unittest "tests/test_eval_llada_ct_csd_bank.py"
```

预期：因为 `LLaDAEvalHarness` 尚未定义 `steering_bank`，测试失败。

## 10. 任务 6：集成 CT-CSD bank 到 `eval_llada_steering.py`

**文件：**

- 修改：`eval_llada_steering.py`

- [ ] **步骤 1：新增 import**

```python
from utils.ct_csd_bank import CTCSDBank
```

- [ ] **步骤 2：增加兼容加载逻辑**

在 `__init__` 中新增：

```python
self.steering_vector = None
self.steering_bank = None
```

加载 `steering_vector_path` 时：

```python
obj = torch.load(steering_vector_path, map_location="cpu", weights_only=True)
if isinstance(obj, dict) and obj.get("format") == "ct_csd_v1":
    if int(obj["target_layer"]) != int(target_layer):
        raise ValueError(f"Bank target_layer={obj['target_layer']} does not match requested target_layer={target_layer}")
    self.steering_bank = CTCSDBank.from_state_dict(obj, device=self.device, dtype=torch.float32)
else:
    key = f"layer_{target_layer}"
    self.steering_vector = obj[key]
```

保留旧格式找不到 `layer_{target_layer}` 时的错误提示。

- [ ] **步骤 3：新增 `_can_steer` helper**

```python
def _can_steer(self):
    return self.steering_vector is not None or self.steering_bank is not None
```

把现有：

```python
can_steer = (self.steering_vector is not None)
```

替换为：

```python
can_steer = self._can_steer()
```

- [ ] **步骤 4：改造 `_per_token_alignment`**

逻辑：

```python
if self.steering_bank is not None:
    return self.steering_bank.alignment(
        block_hidden,
        theta=self.alignment_threshold,
        record=True,
    )
```

否则保持旧 global vector 路径。

- [ ] **步骤 5：改造 `_build_adaptive_steering_hook`**

对 mask hidden states：

```python
if self.steering_bank is not None:
    hidden[_mask_index] = self.steering_bank.steer(
        masked_h,
        beta=_beta,
        theta=_theta,
    ).to(hidden.dtype)
else:
    走旧 global vector 逻辑
```

旧逻辑的公式必须保持：

```python
alpha_t = beta * (a - theta).clamp(min=0)
hidden_new = hidden - alpha_t * sv_unit
```

- [ ] **步骤 6：改造 DIJA Phase 2 alignment**

当前 DIJA Phase 2 直接用 global vector 计算 full sequence alignment。改为：

```python
if self.steering_bank is not None:
    alignment = self.steering_bank.alignment(full_hidden, theta=self.alignment_threshold, record=False)
    _ = self.steering_bank.alignment(full_hidden[original_mask_index], theta=self.alignment_threshold, record=True)
else:
    走旧 global vector alignment
```

这样 `harmful_mask = (alignment > threshold) & original_mask_index` 的语义保持不变，同时 diagnostics 只统计原始 mask 位置。

- [ ] **步骤 7：写出诊断文件**

新增：

```python
def write_steering_diagnostics(self) -> None:
    if self.steering_bank is None:
        return
    os.makedirs(self.generated_samples_path, exist_ok=True)
    out_path = os.path.join(self.generated_samples_path, "ct_csd_diagnostics.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(self.steering_bank.diagnostics(), handle, indent=2, ensure_ascii=False)
```

在 `run_csv_eval(args)` 写完 `results.json` 后调用：

```python
model.write_steering_diagnostics()
```

- [ ] **步骤 8：验证兼容性测试通过**

```bash
python -m unittest "tests/test_eval_llada_ct_csd_bank.py"
```

预期：`Ran 2 tests`，结果为 `OK`。

## 11. 任务 7：集中单元验证

- [ ] **步骤 1：运行 Stage 1 新增测试**

```bash
python -m unittest \
  "tests/test_ct_csd_bank.py" \
  "tests/test_make_ct_csd_llada.py" \
  "tests/test_eval_llada_ct_csd_bank.py"
```

预期：结果为 `OK`。

- [ ] **步骤 2：运行 LLaDA 相关回归测试**

```bash
python -m unittest \
  "tests/test_eval_llada_model_loading.py" \
  "tests/test_eval_llada_generate_until.py" \
  "tests/test_eval_llada_run_csv_eval.py"
```

预期：结果为 `OK`。

- [ ] **步骤 3：运行 utils import smoke**

```bash
python -m unittest "tests/test_utils_import.py"
```

预期：结果为 `OK`。

## 12. 任务 8：构造 smoke bank

**输出：**

- `outputs/ct_csd_llada_m16_smoke/ct_csd_bank.pt`
- `outputs/ct_csd_llada_m16_smoke/run.log`

耗时模型任务必须用 `tmux` 后台运行，并把日志重定向到输出目录。

- [ ] **步骤 1：启动 smoke build**

```bash
mkdir -p "outputs/ct_csd_llada_m16_smoke"
tmux new-session -d -s "ct_csd_stage1_smoke" \
  'cd "/root/myproject/DLM_Steering_Remasking" && python "utils/make_ct_csd_llada.py" \
    --model_path "/dev/shm/LLaDA-8B-Instruct" \
    --harmful_json ".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json" \
    --refusals_txt "utils/refusals.txt" \
    --output_dir "outputs/ct_csd_llada_m16_smoke" \
    --target_layer 31 \
    --max_response_len 128 \
    --max_total_len 2048 \
    --max_samples 32 \
    --method ct_csd \
    --num_total_clusters 4 \
    --kmeans_batch_size 1024 \
    --device cuda \
    --seed 42 \
    > "outputs/ct_csd_llada_m16_smoke/run.log" 2>&1'
```

- [ ] **步骤 2：验证 smoke artifact**

```bash
python - <<'PY'
import torch
path = "outputs/ct_csd_llada_m16_smoke/ct_csd_bank.pt"
obj = torch.load(path, map_location="cpu", weights_only=True)
assert obj["format"] == "ct_csd_v1"
assert obj["target_layer"] == 31
assert obj["config"]["method"] == "ct_csd"
assert obj["vectors"].shape[0] == 4
assert obj["centers"].shape == obj["vectors"].shape
assert int(obj["cluster_sizes"].sum().item()) > 0
print("smoke_bank_ok", tuple(obj["vectors"].shape), obj["cluster_sizes"].tolist())
PY
```

预期：输出以 `smoke_bank_ok` 开头，cluster size 总和为正数。

## 13. 任务 9：构造 full Stage 1 CT-CSD bank

**输出：**

- `outputs/ct_csd_llada_m16/ct_csd_bank.pt`
- `outputs/ct_csd_llada_m16/run.log`

- [ ] **步骤 1：启动 full build**

```bash
mkdir -p "outputs/ct_csd_llada_m16"
tmux new-session -d -s "ct_csd_stage1_full" \
  'cd "/root/myproject/DLM_Steering_Remasking" && python "utils/make_ct_csd_llada.py" \
    --model_path "/dev/shm/LLaDA-8B-Instruct" \
    --harmful_json ".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json" \
    --refusals_txt "utils/refusals.txt" \
    --output_dir "outputs/ct_csd_llada_m16" \
    --target_layer 31 \
    --max_response_len 128 \
    --max_total_len 2048 \
    --method ct_csd \
    --num_total_clusters 16 \
    --kmeans_batch_size 4096 \
    --device cuda \
    --seed 42 \
    > "outputs/ct_csd_llada_m16/run.log" 2>&1'
```

- [ ] **步骤 2：验证 full bank**

```bash
python - <<'PY'
import torch
path = "outputs/ct_csd_llada_m16/ct_csd_bank.pt"
obj = torch.load(path, map_location="cpu", weights_only=True)
assert obj["format"] == "ct_csd_v1"
assert obj["target_layer"] == 31
assert obj["config"]["method"] == "ct_csd"
assert obj["vectors"].shape[0] == 16
assert obj["centers"].shape == obj["vectors"].shape
assert obj["cluster_sizes"].shape[0] == 16
assert int((obj["cluster_sizes"] <= 0).sum().item()) == 0
print("full_bank_ok", tuple(obj["vectors"].shape), obj["cluster_sizes"].tolist())
PY
```

预期：输出以 `full_bank_ok` 开头，16 个 cluster 都非空。

## 14. 任务 10：运行 Stage 1 JBB + DIJA 生成

**输入：**

- `outputs/ct_csd_llada_m16/ct_csd_bank.pt`

**输出：**

- `outputs/jbb_dija_ct_csd_m16/results.json`
- `outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json`
- `outputs/jbb_dija_ct_csd_m16/run.log`

- [ ] **步骤 1：启动生成**

```bash
mkdir -p "outputs/jbb_dija_ct_csd_m16"
tmux new-session -d -s "ct_csd_stage1_jbb_dija" \
  'cd "/root/myproject/DLM_Steering_Remasking" && python "eval_llada_steering.py" \
    --csv_path JBB \
    --attack_method DIJA \
    --model_path "/dev/shm/LLaDA-8B-Instruct" \
    --generated_samples_path "outputs/jbb_dija_ct_csd_m16" \
    --batch_size 32 \
    --sampling_steps 128 \
    --mask_length 128 \
    --block_size 128 \
    --dija_mask_counts 128 \
    --remasking low_confidence \
    --sampler steering \
    --remdm_number 4 \
    --cfg 0 \
    --device cuda:1 \
    --self_reminder False \
    --steering_vector_path "outputs/ct_csd_llada_m16/ct_csd_bank.pt" \
    --steering_overshoot 1.0 \
    --target_layer 31 \
    --alignment_threshold 0.0 \
    --max_refinement_iters 5 \
    --initial_steering_ratio 0.1 \
    > "outputs/jbb_dija_ct_csd_m16/run.log" 2>&1'
```

- [ ] **步骤 2：验证生成结果和诊断**

```bash
python - <<'PY'
import json
from pathlib import Path
result_path = Path("outputs/jbb_dija_ct_csd_m16/results.json")
diag_path = Path("outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json")
rows = json.loads(result_path.read_text(encoding="utf-8"))
diag = json.loads(diag_path.read_text(encoding="utf-8"))
assert len(rows) == 100
assert diag["num_clusters"] == 16
assert diag["total_routed"] > 0
assert len(diag["route_count"]) == 16
assert len(diag["active_count"]) == 16
assert "activation_rate" in diag
print("stage1_generation_ok", len(rows), diag["total_routed"], diag["activation_rate"])
PY
```

预期：输出以 `stage1_generation_ok 100` 开头，`total_routed` 为正数。

## 15. 任务 11：运行本地 Llama Guard judge

**输入：**

- `outputs/jbb_dija_ct_csd_m16/results.json`

**输出：**

- `outputs/jbb_dija_ct_csd_m16/llama_guard_results.json`
- `outputs/jbb_dija_ct_csd_m16/judge.log`

- [ ] **步骤 1：启动 judge**

```bash
tmux new-session -d -s "ct_csd_stage1_judge" \
  'cd "/root/myproject/DLM_Steering_Remasking" && python "scripts/eval_llama_guard_local.py" \
    --data_path "outputs/jbb_dija_ct_csd_m16/results.json" \
    --output_path "outputs/jbb_dija_ct_csd_m16/llama_guard_results.json" \
    --model_path "/dev/shm/Llama-Guard-4-12B" \
    --device cuda:1 \
    > "outputs/jbb_dija_ct_csd_m16/judge.log" 2>&1'
```

- [ ] **步骤 2：验证 judge 指标**

```bash
python - <<'PY'
import json
path = "outputs/jbb_dija_ct_csd_m16/llama_guard_results.json"
data = json.load(open(path, "r", encoding="utf-8"))
meta = data["metadata"]
assert meta["total_samples"] == 100
assert isinstance(meta["unsafe_count"], int)
assert isinstance(meta["asr_percent"], float)
print("stage1_judge_ok", meta["unsafe_count"], meta["total_samples"], meta["asr_percent"])
PY
```

预期：输出以 `stage1_judge_ok` 开头，第二个数字为 `100`。

## 16. 任务 12：生成 Stage 1 指标文档并更新进度文档

**文件：**

- 新增：`docs/category_aware_ct_csd_stage1_metrics.md`
- 修改：`docs/category_aware_ct_csd_stage_progress.md`

- [ ] **步骤 1：用脚本生成 Stage 1 指标文档**

运行：

```bash
python - <<'PY'
import json
from pathlib import Path

judge_path = Path("outputs/jbb_dija_ct_csd_m16/llama_guard_results.json")
diag_path = Path("outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json")
out_path = Path("docs/category_aware_ct_csd_stage1_metrics.md")

judge = json.loads(judge_path.read_text(encoding="utf-8"))
diag = json.loads(diag_path.read_text(encoding="utf-8"))
meta = judge["metadata"]

total = int(meta["total_samples"])
unsafe = int(meta["unsafe_count"])
asr = float(meta["asr_percent"])

text = f"""# Category-aware CT-CSD Stage 1 Metrics

## 状态

Stage 1 CT-CSD bank 最小闭环：completed

## 评测口径

- Method: `ct_csd`
- Model family: `llada`
- `target_layer = 31`
- `num_total_clusters = 16`
- Evaluation: JBB + DIJA refined prompts
- Judge: local `/dev/shm/Llama-Guard-4-12B`

## 产物

| 产物 | 路径 |
|---|---|
| CT-CSD bank | `outputs/ct_csd_llada_m16/ct_csd_bank.pt` |
| Bank build log | `outputs/ct_csd_llada_m16/run.log` |
| Generation results | `outputs/jbb_dija_ct_csd_m16/results.json` |
| CT-CSD diagnostics | `outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json` |
| Judge results | `outputs/jbb_dija_ct_csd_m16/llama_guard_results.json` |

## 核心指标

| 指标 | 数值 |
|---|---:|
| `total_samples` | `{total}` |
| `unsafe_count` | `{unsafe}/{total}` |
| `ASR` | `{asr:.1f}%` |

## 诊断

| 诊断项 | 数值 |
|---|---:|
| `num_clusters` | `{diag["num_clusters"]}` |
| `total_routed` | `{diag["total_routed"]}` |
| `total_active` | `{diag["total_active"]}` |
| `activation_rate` | `{diag["activation_rate"]}` |
| `route_time_sec` | `{diag["route_time_sec"]}` |

## Stage 0 对比

| 方法 | `ASR` | `unsafe_count` | `total_samples` |
|---|---:|---:|---:|
| Global Sentence-CSD Stage 0 | `74.0%` | `74/100` | `100` |
| CT-CSD Stage 1 | `{asr:.1f}%` | `{unsafe}/{total}` | `{total}` |
"""

out_path.write_text(text, encoding="utf-8")
print(out_path)
PY
```

该脚本会从真实 judge 和 diagnostics JSON 读取数值，不允许手写猜测指标。

- [ ] **步骤 2：用脚本追加 Stage 1 到进度文档**

运行：

```bash
python - <<'PY'
import json
from pathlib import Path

judge_path = Path("outputs/jbb_dija_ct_csd_m16/llama_guard_results.json")
progress_path = Path("docs/category_aware_ct_csd_stage_progress.md")

judge = json.loads(judge_path.read_text(encoding="utf-8"))
meta = judge["metadata"]
total = int(meta["total_samples"])
unsafe = int(meta["unsafe_count"])
asr = float(meta["asr_percent"])

section = f"""

## Stage 1: completed

Stage 1 在与 Stage 0 相同的 JBB + DIJA 评测口径下冻结 CT-CSD bank 最小闭环。

| 产物 | 路径 |
|---|---|
| CT-CSD bank | `outputs/ct_csd_llada_m16/ct_csd_bank.pt` |
| Generation results | `outputs/jbb_dija_ct_csd_m16/results.json` |
| Judge results | `outputs/jbb_dija_ct_csd_m16/llama_guard_results.json` |
| Diagnostics | `outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json` |

| 指标 | 数值 |
|---|---:|
| `total_samples` | `{total}` |
| `unsafe_count` | `{unsafe}/{total}` |
| `ASR` | `{asr:.1f}%` |
"""

text = progress_path.read_text(encoding="utf-8")
if "## Stage 1: completed" in text:
    raise RuntimeError("Stage 1 progress section already exists; update it manually after reading the existing section.")
progress_path.write_text(text.rstrip() + section, encoding="utf-8")
print(progress_path)
PY
```

该脚本会从 `llama_guard_results.json` 读取具体指标并追加，不允许在进度文档中保留占位说明。

## 17. 任务 13：补充 `num_total_clusters` 消融实验

**目标：** 在阶段 1 已完成 `m16` 主实验的基础上，补充 `num_total_clusters = 4, 8, 12, 16` 的簇数消融。除 `num_total_clusters` 和输出目录外，其余参数必须与 Stage 1 主实验保持一致。

**状态约定：**

- `m16` 已完成，可直接作为消融中的 `16` 点位。
- `m4`、`m8`、`m12` 待运行。
- 若后续为了严格同批次对比而重跑 `m16`，必须使用新的输出目录并在进度文档中单独说明，避免覆盖当前冻结产物。

**输出目录：**

| `num_total_clusters` | bank 输出目录 | 生成输出目录 |
|---:|---|---|
| `4` | `outputs/ct_csd_llada_m4` | `outputs/jbb_dija_ct_csd_m4` |
| `8` | `outputs/ct_csd_llada_m8` | `outputs/jbb_dija_ct_csd_m8` |
| `12` | `outputs/ct_csd_llada_m12` | `outputs/jbb_dija_ct_csd_m12` |
| `16` | `outputs/ct_csd_llada_m16` | `outputs/jbb_dija_ct_csd_m16` |

- [ ] **步骤 1：构造 `m4`、`m8`、`m12` CT-CSD bank**

对 `K=4`、`K=8`、`K=12` 分别运行。不要并行占用同一块 GPU；等待一个 tmux 会话结束并通过验证后再启动下一个。

```bash
K=4
mkdir -p "outputs/ct_csd_llada_m${K}"
tmux new-session -d -s "ct_csd_stage1_m${K}_build" \
  "cd \"/root/myproject/DLM_Steering_Remasking\" && python \"utils/make_ct_csd_llada.py\" \
    --model_path \"/dev/shm/LLaDA-8B-Instruct\" \
    --harmful_json \".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json\" \
    --refusals_txt \"utils/refusals.txt\" \
    --output_dir \"outputs/ct_csd_llada_m${K}\" \
    --target_layer 31 \
    --max_response_len 128 \
    --max_total_len 2048 \
    --method ct_csd \
    --num_total_clusters ${K} \
    --kmeans_batch_size 4096 \
    --device cuda \
    --seed 42 \
    > \"outputs/ct_csd_llada_m${K}/run.log\" 2>&1"
```

完成 `K=4` 后，把第一行改成 `K=8` 和 `K=12`，其余命令保持不变。

- [ ] **步骤 2：验证每个消融 bank**

```bash
python - <<'PY'
import torch

for k in [4, 8, 12]:
    path = f"outputs/ct_csd_llada_m{k}/ct_csd_bank.pt"
    obj = torch.load(path, map_location="cpu", weights_only=True)
    assert obj["format"] == "ct_csd_v1", path
    assert obj["target_layer"] == 31, path
    assert obj["config"]["method"] == "ct_csd", path
    assert obj["config"]["num_total_clusters"] == k, path
    assert obj["vectors"].shape[0] == k, path
    assert obj["centers"].shape == obj["vectors"].shape, path
    assert obj["cluster_sizes"].shape[0] == k, path
    assert int((obj["cluster_sizes"] <= 0).sum().item()) == 0, path
    print("ablation_bank_ok", k, tuple(obj["vectors"].shape), obj["cluster_sizes"].tolist())
PY
```

预期：分别输出 `ablation_bank_ok 4`、`ablation_bank_ok 8`、`ablation_bank_ok 12`，且每个 cluster 都非空。

- [ ] **步骤 3：运行每个消融点的 JBB + DIJA 生成**

对 `K=4`、`K=8`、`K=12` 分别运行。生成参数必须与 `m16` 主实验一致。

```bash
K=4
mkdir -p "outputs/jbb_dija_ct_csd_m${K}"
tmux new-session -d -s "ct_csd_stage1_m${K}_jbb_dija" \
  "cd \"/root/myproject/DLM_Steering_Remasking\" && python \"eval_llada_steering.py\" \
    --csv_path JBB \
    --attack_method DIJA \
    --model_path \"/dev/shm/LLaDA-8B-Instruct\" \
    --generated_samples_path \"outputs/jbb_dija_ct_csd_m${K}\" \
    --batch_size 32 \
    --sampling_steps 128 \
    --mask_length 128 \
    --block_size 128 \
    --dija_mask_counts 128 \
    --remasking low_confidence \
    --sampler steering \
    --remdm_number 4 \
    --cfg 0 \
    --device cuda:1 \
    --self_reminder False \
    --steering_vector_path \"outputs/ct_csd_llada_m${K}/ct_csd_bank.pt\" \
    --steering_overshoot 1.0 \
    --target_layer 31 \
    --alignment_threshold 0.0 \
    --max_refinement_iters 5 \
    --initial_steering_ratio 0.1 \
    > \"outputs/jbb_dija_ct_csd_m${K}/run.log\" 2>&1"
```

完成 `K=4` 后，把第一行改成 `K=8` 和 `K=12`，其余命令保持不变。

- [ ] **步骤 4：验证生成结果和诊断**

```bash
python - <<'PY'
import json
from pathlib import Path

for k in [4, 8, 12]:
    result_path = Path(f"outputs/jbb_dija_ct_csd_m{k}/results.json")
    diag_path = Path(f"outputs/jbb_dija_ct_csd_m{k}/ct_csd_diagnostics.json")
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert len(rows) == 100, result_path
    assert diag["num_clusters"] == k, diag_path
    assert diag["total_routed"] > 0, diag_path
    assert len(diag["route_count"]) == k, diag_path
    assert len(diag["active_count"]) == k, diag_path
    assert "activation_rate" in diag, diag_path
    print("ablation_generation_ok", k, len(rows), diag["total_routed"], diag["activation_rate"])
PY
```

- [ ] **步骤 5：运行本地 Llama Guard judge**

对 `K=4`、`K=8`、`K=12` 分别运行。

```bash
K=4
tmux new-session -d -s "ct_csd_stage1_m${K}_judge" \
  "cd \"/root/myproject/DLM_Steering_Remasking\" && python \"scripts/eval_llama_guard_local.py\" \
    --data_path \"outputs/jbb_dija_ct_csd_m${K}/results.json\" \
    --output_path \"outputs/jbb_dija_ct_csd_m${K}/llama_guard_results.json\" \
    --model_path \"/dev/shm/Llama-Guard-4-12B\" \
    --device cuda:1 \
    > \"outputs/jbb_dija_ct_csd_m${K}/judge.log\" 2>&1"
```

完成 `K=4` 后，把第一行改成 `K=8` 和 `K=12`，其余命令保持不变。

- [ ] **步骤 6：验证 judge 指标并汇总四点消融**

```bash
python - <<'PY'
import json
from pathlib import Path

for k in [4, 8, 12, 16]:
    path = Path(f"outputs/jbb_dija_ct_csd_m{k}/llama_guard_results.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    assert meta["total_samples"] == 100, path
    assert isinstance(meta["unsafe_count"], int), path
    assert isinstance(meta["asr_percent"], float), path
    print("ablation_judge_ok", k, meta["unsafe_count"], meta["total_samples"], meta["asr_percent"])
PY
```

- [ ] **步骤 7：更新阶段进度文档**

把 `docs/category_aware_ct_csd_stage_progress.md` 中“阶段 1 簇数消融实验”的 `m4`、`m8`、`m12` 状态从“待运行”改为“已完成”，并填入每个点位真实的 `unsafe_count` 和 `ASR`。不允许手写猜测指标；必须从对应的 `llama_guard_results.json` 读取。

## 18. 最终验收清单

- [ ] `python -m unittest "tests/test_ct_csd_bank.py"` 通过。
- [ ] `python -m unittest "tests/test_make_ct_csd_llada.py"` 通过。
- [ ] `python -m unittest "tests/test_eval_llada_ct_csd_bank.py"` 通过。
- [ ] 现有 LLaDA eval 回归测试通过。
- [ ] `outputs/ct_csd_llada_m16/ct_csd_bank.pt` 存在，且 `format == "ct_csd_v1"`。
- [ ] 旧 `steering_vectors.pt` 路径仍可被 `eval_llada_steering.py` 加载。
- [ ] 新 `ct_csd_bank.pt` 路径可被 `eval_llada_steering.py` 加载。
- [ ] `outputs/jbb_dija_ct_csd_m16/results.json` 包含 `100` 条样本。
- [ ] `outputs/jbb_dija_ct_csd_m16/ct_csd_diagnostics.json` 包含 `route_count`、`active_count`、`activation_rate`。
- [ ] `outputs/jbb_dija_ct_csd_m16/llama_guard_results.json` 包含 `total_samples`、`unsafe_count`、`asr_percent`。
- [ ] `docs/category_aware_ct_csd_stage1_metrics.md` 只记录 `ASR`、`unsafe_count`、`total_samples` 和诊断。
- [ ] `docs/category_aware_ct_csd_stage_progress.md` 只在 Stage 1 产物真实存在后追加 Stage 1 状态。

## 19. 退出标准

Stage 1 完成条件：

- `ct_csd_bank.pt` 能被 `eval_llada_steering.py` 正常加载。
- 旧 global `steering_vectors.pt` 路径保持兼容。
- CT-CSD 推理中每个被评估 hidden state 都能 route 到一个 local center。
- 只有 local alignment 大于 `alignment_threshold = 0.0` 的 hidden states 执行 steering。
- 完成 full JBB + DIJA generation 和本地 Llama Guard judge。
- 最终记录包含可与 Stage 0 直接比较的 `ASR`、`unsafe_count`、`total_samples`。
