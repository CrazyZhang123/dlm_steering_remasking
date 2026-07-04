# 推理批量化重构（feat/batch-inference）

## 动机

此前所有环节逐样本前向：`generate_until()` 对每条样本以 `[1, L]` 跑约 138 次
8B 模型前向（Phase 1 去噪 128 步 + Phase 2 重掩码最多 5×2 次），GPU 利用率低。
本分支引入 batch 维度并行，预估 `gen_batch_size=4` 约 2.5–3x、8 约 3.5–5x。

## 设计要点

- **右 pad + attention_mask**：LLaDA 的 RoPE 位置由 token 下标隐式决定
  （`modeling_llada.py` 的 forward 无 `position_ids`），右 pad 不改变真实 token
  的位置编码；左 pad 会平移位置、数值不等价，故排除。pad 值用 eos（126081）。
- **全 1 mask 被模型丢弃**（`modeling_llada.py` L1251）：`forward_logits` 仅在
  batch 内存在 pad 时才传 `attention_mask`，因此 `gen_batch_size=1` 或等长 batch
  与原实现走完全相同的 kernel 路径，**逐位等价**（同种子下 RNG 消耗形状一致）。
- **eos 提前退出 → 按行冻结**：原实现发现 eos 即整条 return；批量版用 done mask
  冻结该行（后续 block 不再揭码/steering/Phase 2），行内容与原实现一致。
- **Phase 2 逐样本 break → 活跃行压缩**：`n_harmful == 0` 的行退出 batch，
  剩余行继续 refinement；bank 诊断（record=True）只喂活跃行，`route_count`
  口径与逐样本一致。
- **参数**：新增 `--gen_batch_size`（默认 1）。与既有 `--batch_size` 无关
  （后者仅用于 loglikelihood 的 MC 采样）。

## 改动清单

| 文件 | 内容 |
|---|---|
| `utils/batching.py`（新增） | `build_padded_xt` / `pad_token_rows` / `rowwise_topk_transfer` / `forward_logits` / `extract_layer_hidden` |
| `eval_llada_steering.py` | 批量核心 `_block_decode_batch` / `_refine_block_batch` / `_sample_blocks_batch` / `_dija_sample_batch`（模块级）；`llada_conf_sample` / `llada_remask_sample` / `llada_dija_sample` 改为 batch=1 薄委托并新增 `*_batch` 方法；`generate_until` 分 chunk；`--gen_batch_size` |
| `scripts/eval_llama_guard_local.py` | 新增 `generate_guard_texts_batch`（左 pad 批量 generate）与 `--batch_size`（默认 8）；`run_evaluation(batch_size=1)` 保持原路径 |
| `utils/llama_guard.py` | 批量化重写 + 修复两个既有 bug（`response` 字段写入整个列表；渲染失败 `continue` 导致 prompt/result 错位） |
| `scripts/llada_steer.sh` / `scripts/llama_guard.sh` | 透传新参数；`llada_steer.sh` 改为调用仓库根目录 `eval_llada_steering.py` |
| `tests/test_batching_utils.py` / `test_eval_llada_batch_sample.py` / `test_eval_llada_dija_batch.py` / `test_llama_guard_batch.py` / `test_script_entrypoints.py`（新增） | pad 布局、逐行 topk、batch↔single 逐 token 等价（含 inject_prompt / eos 冻结 / 行压缩 / DIJA per-row k 调度 / 仅原始 mask 重掩码）、Llama Guard 左 pad 与 schema、shell 入口脚本引用的 Python 文件存在性 |
| `scripts/build_pap_json.py` / `utils/pap_templates_vendored.py` / `utils/dija_generate_function.py` / `scripts/eval_openai_compatible_judge.py` | 为打通本工作树全量 `unittest` 回归而补的辅助兼容修复：PAP 模板本地回退、DIJA 缺席时的单测兼容实现、OpenAI judge 的 resume |

## 等价性口径

1. **B=1 逐位等价**：同种子下重构前后 `results.json` 应逐字节一致（无 pad 时
   forward 调用形式与 RNG 消耗形状完全相同）。
2. **混合长度 batch**：pad 行走 math-SDPA kernel（float mask 禁用 flash），
   bf16 归约顺序不同 + `torch.rand_like` 形状从 `[1,L,V]` 变 `[B,Lmax,V]`，
   与逐样本运行**不逐位一致，按统计口径验收**（历史运行本就未固定种子）。
3. **Llama Guard 批量**：贪心解码，批量与逐条输出应逐字一致。

## GPU 验收清单（已完成，2026-07-03/04，详见主仓库 `docs/batch_inference_gpu_acceptance_log.md`）

- [x] B=1 逐位等价：**PASS**。DIJA 前 10 条、seed=42，main 旧实现 vs worktree bs=1，
      `results.json` 逐字节一致（`cmp`）。注：验收输入用 DIJA 而非 zeroshot——
      zeroshot + M12 steering 下模型全部拒答成空 response，等价比对是平凡通过。
- [x] 贪心近似等价：**PASS（token 一致率 100.00%，1402/1402）**。argmax 模式，
      10 条 bs=1 vs bs=4，response 逐字相同。
- [x] 统计等价：**PASS**。JBB DIJA 100 条，bs=1 unsafe 70/100 vs bs=4 unsafe 71/100，
      差 1pp ≤ 2pp（且落在 ASR 单次噪声 ±2.5% 内）。
- [x] 显存与吞吐：完成，但**本机（V100, bf16）加速 ≈1x**（bs=2 1.11x / bs=4 1.06x /
      bs=8 0.97x），显存 16.3→23.1 GiB 近线性。原因：V100 无 bf16 tensor core，
      bs=1 时 GEMM 已算力饱和（MFU 77–86%）；判定为硬件预期而非代码缺陷。
      批量收益需 A100/H100（bf16 tensor core）或 fp16 才能兑现。
      工程建议：本机日常实验用 `--gen_batch_size 1`；Llama Guard 评判保留 batch=8
      （短生成，100 条 ~52s，批量收益真实）。

## 后续（未含在本分支）

- P3：`utils/make_ct_csd_llada.py` 特征提取批量化（一次性构造成本，收益最低）。
- P4：`eval_dream_steering.py` 镜像移植（先确认 Dream modeling 的 pad/position 行为）。
