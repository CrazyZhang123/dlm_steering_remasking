# DLM Steering 运行手册

> 本文档只保留“怎么跑”和“产物在哪”。结果解释统一收口到 `docs/experiment_summary.md`。

## 1. 环境与模型

- 工作目录：仓库根目录
- Python：`/root/miniconda3/bin/python`
- LLaDA：

```bash
python "scripts/restore_llada_model.py"
```

- Llama-Guard：

```bash
python "scripts/restore_llama_guard_model.py"
```

## 2. 统一评测口径

- 生成：`JBB + DIJA`
- 样本数：`100`
- 评判：本地 `Llama-Guard-4-12B`
- 关键参数：
  - `target_layer=31`
  - `sampling_steps=128`
  - `mask_length=128`
  - `block_size=128`
  - `dija_mask_counts=128`
  - `alignment_threshold=0.0`
  - `steering_overshoot=1.0`
  - `initial_steering_ratio=0.1`
  - `max_refinement_iters=5`

## 3. 主要实验入口

### 3.1 全局 CT-CSD / 簇数扫描

- 构造 bank：`utils/make_ct_csd_llada.py`
- 多轮扫描脚本：`scripts/run_ct_csd_m_scan_repeat.sh`

### 3.2 Category-aware CT-CSD

- 构造 bank：`utils/make_ct_csd_llada.py --method category_ct_csd`
- 结果目录：`outputs/jbb_dija_category_ct_csd_m*/`

### 3.3 MIL / direction / random / KNN

- 统一入口：`utils/make_ct_csd_llada.py`
- 关键参数：
  - `--token_selection mil_probe_threshold`
  - `--token_selection direction_top_ratio`
  - `--token_selection random_top_ratio`
  - `--token_selection knn_label_clean`

### 3.4 Feature preprocess

- 统一入口：`utils/make_ct_csd_llada.py`
- 关键参数：
  - `--feature_preprocess l2_only`
  - `--feature_preprocess center_l2`
  - `--feature_preprocess center_pca128_l2`
  - `--feature_preprocess center_pca256_l2`

### 3.5 Steering 超参扫描

```bash
bash "scripts/run_steering_hparam_scan.sh" <gpu_id> 3 <os:isr> [<os:isr> ...]
```

### 3.6 Sure/Sorry 极简方向

```bash
bash "scripts/run_sure_sorry_csd_scan.sh" <gpu_id> 3 word:512 word:9605
bash "scripts/run_sure_sorry_csd_scan.sh" <gpu_id> 3 phrase:512 phrase:9605
```

## 4. 结果读取

- 生成结果：

```text
outputs/<experiment>/results.json
```

- 判分结果：

```text
outputs/<experiment>/llama_guard_results.json
```

- 快速读取 ASR：

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("outputs/<experiment>/llama_guard_results.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload["metadata"]["unsafe_count"], payload["metadata"]["asr_percent"])
PY
```

## 5. 当前活跃总表

- 结果与结论：`docs/experiment_summary.md`
- 旧进度 / 日志 / metrics：`docs/archive/`
