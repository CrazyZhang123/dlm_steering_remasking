# 批量推理 GPU 实机验收：运行日志与断点重连指南（feat/batch-inference）

> 本文档为**运行中日志 + 交接入口**，供会话中断后重连使用。全部完成后在 §7 汇总结论，
> 并回写 `docs/batch_inference_refactor.md` 的「GPU 验收清单」勾选状态。
> 背景交接文档：`.claude/worktrees/feat-batch-inference/docs/batch_inference_session_log.md`。

## 1. 背景（截至 2026-07-03 15:32 UTC）

- 批量推理重构代码已全部完成并入库：worktree
  `.claude/worktrees/feat-batch-inference/`，分支 **`feat/batch-inference`**（已从
  `worktree-feat-batch-inference` 改名），HEAD `f72b9c3`，已 rebase 到 main（`d4569a2`），
  领先 main 7 个提交、落后 0；worktree 工作区干净。
- 本地全量回归 `unittest discover`：**174 tests OK**（171 批量化 + 3 来自 main 的
  `test_make_ct_csd_llada`）。
- 剩余唯一验收项：**GPU 实机验收**（`docs/batch_inference_refactor.md` §GPU 验收清单），
  即本文档所记录的任务。本轮任务来自用户 2026-07-03 决策："本机测试，继续推进"。
- 未 push；按 `AGENTS.md` 约定，git 外部操作需用户明确要求。

## 2. 验收任务定义（四步）

**输入选择（重要，见 §5 16:04 修正）**：四步统一用 **DIJA** 输入
（`--csv_path JBB --attack_method DIJA`，与 Stage 2 M12 同口径）。**不能用 zeroshot**：
直喂原始有害 goal 时 LLaDA-Instruct 叠加 M12 steering 会全部拒答成**空 response**，
T1/T2 变成"空==空"的平凡通过，验证不了 batch 逻辑（已实测 10/10 全空）。DIJA 口径下
Stage 2 M12 产物 100/100 非空，可有效验证。

| 步骤 | 内容 | 判定标准 |
|---|---|---|
| T1 逐位等价 | DIJA 前 10 条，seed=42，main 旧实现 vs worktree `bs=1` | 两侧 `results.json` **逐字节一致**（`cmp`） |
| T2 argmax 近似等价 | DIJA 前 10 条，`_sample_categorical` patch 成 argmax，worktree `bs=1` vs `bs=4` | response token 一致率 **>99%** |
| T4 吞吐/显存 | DIJA 前 10 条，worktree `bs∈{1,2,4,8}` | 记录 `gen_only_sec` 加速比 + `peak_mem_mib` 曲线 |
| T3 统计等价 | JBB+DIJA 100 条（`dija_mask_counts=128`），worktree `bs=1`(seed42) vs `bs=4`(seed43) + 本地 Llama Guard | unsafe 率差 **≤2pp** |

小样本截断由 `driver.py --limit N` 实现（在 `generate_until` 层截前 N 条 request）。

评测口径与 Stage 2 对齐：M12 bank（`outputs/ct_csd_llada_m12/ct_csd_bank.pt`）、
`target_layer=31`、`sampling_steps=128`、`mask_length=128`、`block_size=128`、
`alignment_threshold=0.0`、`steering_overshoot=1.0`、`initial_steering_ratio=0.1`、
`max_refinement_iters=5`、推理模型 `/dev/shm/LLaDA-8B-Instruct`、评判模型
`/dev/shm/Llama-Guard-4-12B`。

## 3. 执行代码与机制（断连后先读这里）

代码都在 `outputs/batch_accept/`（该目录被 .gitignore 忽略，**不入库**；如目录丢失按
§8 重建）：

- **`outputs/batch_accept/driver.py`**：验收驱动。
  - `--repo` 指定用哪套 `eval_llada_steering.py`（main 根目录 = 旧实现；worktree =
    批量实现），`--workdir` 固定主仓库根（DIJA 数据相对路径需要）。
  - 关键机制：monkeypatch `generate_until`，在**模型加载后、生成开始前**统一
    `set_seed(seed)`——保证 main / worktree 两侧进入生成时 RNG 状态一致（T1 逐位
    等价的前提；两侧加载路径不同，若在进程开头播种会被加载噪声打乱）。
  - `--argmax` 把 `es._sample_categorical` 替换为 `argmax(dim=-1)`（T2 用，消除采样随机）。
  - `--limit N`：在 `generate_until` 层只取前 N 条 request（T1/T2/T4 小样本用）。
  - 每次运行打印 `[driver] gen_only_sec=... peak_mem_mib=...`（生成阶段计时 +
    CUDA 峰值显存，T4 直接从各 `t*.log` 里 grep）。
- **`outputs/batch_accept/accept.sh`**：按 T1 → T2 → T4 → T3 顺序执行全部 9 次推理 +
  2 次评判；每步产物在 `outputs/batch_accept/<tag>/`，日志 `<tag>.log`；主日志
  `accept_main.log` 只记阶段节点与 PASS/DIFF。
  - **注意：脚本无断点续跑**，中断后重启会从头全部重跑；恢复时优先按 §6 手动补跑
    未完成的单步（driver.py 命令行在 accept.sh 里逐条可抄）。
- **`outputs/batch_accept/accept10.csv`**：10 条验收样本（JBB DIJA json 前 10 条的
  vanilla `goal` 字段）。重建方法见 §8。

启动命令（已于 15:32 UTC 执行）：

```bash
tmux new-session -d -s batch_accept -c /root/myproject/DLM_Steering_Remasking \
  "bash outputs/batch_accept/accept.sh > outputs/batch_accept/accept_main.log 2>&1"
```

## 4. 监控方式（低频，默认 30 分钟一次）

```bash
# 阶段节点 + T1 PASS/DIFF 结论
cat outputs/batch_accept/accept_main.log

# 当前单步进度条（换成正在跑的 tag）
tail -c 400 outputs/batch_accept/t1_main.log

# tmux 会话存活
tmux ls   # 应看到 batch_accept

# T3 评判结果（跑完后）
/root/miniconda3/bin/python -c "import json; print(json.load(open('outputs/batch_accept/t3_bs1/llama_guard_results.json'))['metadata'])"
```

## 5. 时间线（边运行边补充）

| 时间 (UTC) | 事件 |
|---|---|
| 2026-07-03 15:26 | 收到任务：推进 `batch_inference_session_log.md` 剩余步骤（GPU 实机验收）。确认 GPU0 空闲（Stage 2 m_scan 已收官）、GPU1 被 `m12_prep_abl`（另一实验）占用；两模型均在 /dev/shm |
| 2026-07-03 15:30 | 写好 `driver.py` / `accept.sh` / `accept10.csv`，py_compile 与 bash -n 通过 |
| 2026-07-03 15:32 | tmux `batch_accept` 启动（GPU0，**zeroshot 版**），T1 main 侧开始 |
| 2026-07-03 15:35 | 冒烟确认：t1_main 推理 4/10（~46.8s/条），bank/模型加载正常，无报错 |
| 2026-07-03 16:04 | **发现 zeroshot 输入失效**：T1/T2 四份产物均 10/10 空 response（M12 steering 下模型对赤裸有害 goal 全部拒答），T1 逐字节一致是"空==空"平凡通过，验证不了 batch 逻辑。对照 Stage 2 M12 DIJA 产物 100/100 非空，判定为验收输入选错，非代码 bug |
| 2026-07-03 16:07 | 停掉 zeroshot 版 tmux，给 driver 加 `--limit`，accept.sh 改为四步统一 DIJA 输入（T1/T2/T4 前 10 条、T3 全 100 条），清理空产物后重启 tmux `batch_accept`。T1 main 重新开始（~50s/条），产物待验非空 |
| 2026-07-03 17:55 | T4 吞吐未提速原因分析完成（纯代码/数据侧，未动 GPU，T3 继续在跑）：主因 V100 无 bf16 tensor core，bs=1 已算力饱和（实效 ~12.1 TFLOPS ≈ CUDA core 峰值 77%+），批量收益上限 ≈1x；pad 浪费/attention_bias/逐步同步为次因。详见 §7.1 |
| 2026-07-03 19:16 | T3 推理 + 双侧 Llama Guard 评判完成，`ACCEPT ALL DONE` 打出，tmux 会话正常退出 |
| 2026-07-04 | 复核 T3 结果并回写结论：bs=1 unsafe 70/100 vs bs=4 unsafe 71/100，差 1pp ≤ 2pp → **T3 PASS**。四步验收全部完成，勾选 refactor.md 清单、更新 session_log |

## 6. 进度与产物核对表（断连后对照此表判断跑到哪一步）

（下表为 16:07 重启的 DIJA 版；全部完成于 2026-07-03 19:16 UTC）

| 步骤 | 产物（`outputs/batch_accept/` 下） | 状态 |
|---|---|---|
| T1 main | `t1_main/results.json` + `t1_main.log` | ✅ 完成 |
| T1 worktree | `t1_wt/results.json` | ✅ 完成 |
| T1 比对 | `accept_main.log` 中 `[T1] PASS/DIFF` 行 | ✅ PASS |
| T2 bs=1 / bs=4 | `t2_bs1/` `t2_bs4/` | ✅ PASS（token 一致率 100%） |
| T4 bs=1/2/4/8 | `t4_bs1/` `t4_bs2/` `t4_bs4/` `t4_bs8/` | ✅ 完成（加速 ≈1x，见 §7.1） |
| T3 bs=1 / bs=4 | `t3_bs1/` `t3_bs4/`（各含 `results.json` + `llama_guard_results.json`） | ✅ PASS（70% vs 71%） |

耗时预估（DIJA 每条 ~50s，比 zeroshot 略慢）：T1 ≈ 8min×2；T2 ≈ 8+3min；
T4 ≈ 8+5+3+2min；T3 是大头（bs=1 ≈ 80min、bs=4 ≈ 25min、评判 ≈ 2min×2）。
**全程 ≈ 3–3.5h，预计 ~19:30 UTC 打出 `ACCEPT ALL DONE`**。

验非空命令（任一产物）：
`/root/miniconda3/bin/python -c "import json; d=json.load(open('outputs/batch_accept/t1_wt/results.json')); print('非空', sum(1 for r in d if r['response'].strip()),'/',len(d))"`

## 7. 结果汇总与结论（完成后填写）

- **T1 逐位等价：PASS**（16:22 UTC）。main 旧实现 vs worktree bs=1，`results.json`
  逐字节一致；两侧 10/10 非空（首条 160+ 字符实质内容），`gen_only_sec` 416.7 vs 417.0，
  `peak_mem_mib` 16275/16275。证明 bs=1 无 pad 路径与原实现完全等价。
- **T2 argmax 近似等价：PASS（token 一致率 100.00%）**（16:35 UTC）。worktree bs=1 vs
  bs=4，argmax 模式，10/10 行 response 完全相同，逐 token 1402/1402 一致（标准 >99%）。
  证明 pad/attention_mask/逐行 top-k/活跃行压缩在多样本并行下完全正确。bs=1 vs bs=4
  峰值显存 16059 → 18322 MiB。
- **T4 吞吐/显存：完成，但加速比未达预期（已分析，结论见 §7.1）**（17:03 UTC）。10 条 DIJA：

  | bs | gen_only_sec | peak_mem_mib | vs bs=1 |
  |---:|---:|---:|---:|
  | 1 | 417.3 | 16275 | 1.00x |
  | 2 | 376.7 | 17242 | 1.11x |
  | 4 | 392.9 | 19183 | 1.06x |
  | 8 | 428.8 | 23064 | 0.97x（更慢） |

  显存随 bs 近线性增长（符合预期）；但吞吐几乎无提升、bs=8 反而更慢，与 refactor.md
  预估的 2.5–5x 严重不符。**注意：这不代表批量逻辑错**（T1 逐位等价 + T2 100% token
  一致已证明数值完全正确），而是**性能收益未兑现**。疑因（待 profile 确认）：DIJA +
  steering 路径存在按行串行操作（Phase 2 逐行 refill、steering hook 逐 token 处理），
  batch 内仍近似串行；叠加 pad 行浪费 + float attention_mask 禁用 flash-attention 的
  开销，抵消并行收益。zeroshot/PAP 等更"纯"的路径可能表现不同，本轮未覆盖（因 zeroshot
  被 steering 拒答成空，无法产出可比样本）。
- **T3 统计等价：PASS**（19:16 UTC 完成，07-04 复核）。JBB DIJA 100 条 + 本地
  Llama Guard：bs=1(seed42) unsafe **70/100（ASR 70.0%）** vs bs=4(seed43) unsafe
  **71/100（ASR 71.0%）**，差 **1pp ≤ 2pp** 标准。同时落在 ASR 单次噪声（±2.5%）
  范围内，与"批量仅改变 RNG 消耗形状、不改变分布"的预期一致。生成侧
  `gen_only_sec` 3633(bs=1) vs 4199(bs=4)——bs=4 更慢与 §7.1 的 V100 结论一致。
  Llama Guard 批量评判（bs=8）单侧 100 条仅 ~52s。

### 最终结论（2026-07-04）

**四步验收全部完成：T1/T2/T3 PASS，T4 完成（加速 ≈1x，判定为 V100 硬件预期而非
代码缺陷，见 §7.1 建议 1"就地接受"）。** 批量化实现的**数值正确性**已被
T1（逐位等价）、T2（100% token 一致）、T3（1pp 统计差）三重背书；**性能收益**
在本机 V100 上不可兑现（bf16 无 tensor core、bs=1 已算力饱和），迁移到
A100/H100 或改用 fp16 时才能体现。工程侧建议：本机日常实验保持
`--gen_batch_size 1`（省显存且逐位可复现），Llama Guard 评判侧保留 batch=8
（评判是短生成，批量收益真实存在）。

### 7.1 T4 吞吐未提速：原因分析（2026-07-03 17:55 UTC，纯代码/数据侧，未动 GPU）

**主因：V100 无 bf16 tensor core，bs=1 时 GEMM 已把 CUDA core 打满，批量没有空闲算力可填。**

- 算力核算：`t4_bs1` gen_only_sec=417.3s ÷（10 样本 × 128 步）= 0.326 s/前向；
  每次前向 FLOPs ≈ 2 × 8e9 参数 × ~246 token ≈ 3.94 TFLOP → **实效 ~12.1 TFLOPS**。
  V100（sm70）tensor core 仅支持 fp16，`torch.bfloat16` matmul 走 CUDA core
  （≈fp32 峰值 14–15.7 TFLOPS）→ **MFU 77–86%，bs=1 已饱和**。旁证：验收全程 GPU util 91–100%。
- 预估失效原因：refactor.md 的 2.5–5x 预估隐含"bs=1 利用率低"（AR decode 式
  memory-bound）假设；**DLM 每步都是全序列前向（prefill 型 GEMM）**，seq≈200–250 时
  bs=1 的 GEMM 已够大。该假设仅在 bf16 tensor core 卡（A100/H100）或 V100+fp16 下成立。
- bs=2 的 1.11x 来自摊薄每前向固定开销（fp64 采样、kernel launch、同步）；bs≥4 被下述
  次因吃回。

**次因（解释 377→393→429 的递增变慢，均已按 10 条真实长度 184–246 token 量化）：**

1. **pad 计算浪费**：按 chunk max-len 精确计算，bs=2 +3.4% / bs=4 +7.0% / bs=8 +9.9%。
2. **混长 chunk 传 attention_mask** → `modeling_llada` 走 attention_bias 相加路径，逐层
   物化 bias；注意本模型 `config.flash_attention=false`，原本就走 sdpa，
   **旧疑因"float mask 禁用 flash-attention"不成立**，实际开销是 bias 物化 + 非 no-mask 路径。
3. **fp64 softmax + Gumbel `rand_like` 作用在含 pad 的 `[B, Lmax, 126k]` 上**：
   bs=1 时约 0.3s/样本（占比小），但随 pad 行同步放大、纯浪费。
4. **逐步 CPU 同步**：`_dija_sample_batch` 的逐行 k 调度 Python 循环
   （`mask_index[j].sum().item()`）每步每行一次 GPU→CPU 同步（bs=8 单 chunk ~128×8 次）；
   bank `_record` 的 `.to("cpu")` 在 refinement 中亦有少量同步。

**旧疑因核对**："Phase 2 逐行 refill / steering hook 逐 token 串行"**不成立**——两者均为
批量张量操作（活跃行压缩 + 展平 mask 索引），真正的按行 Python 循环只有 DIJA k 调度
（代价是同步而非串行前向）。

**建议（按性价比）：**

1. **就地接受**：V100+bf16 上批量收益本就 ≈1x，T4 记为"符合硬件预期"而非代码缺陷；
   批量化的价值在 tensor core 卡（A100/H100）与 fp16 场景，正确性已由 T1/T2 背书。
2. 若必须在 V100 提速：试 `torch.float16`（tensor core ~112 TFLOPS，GEMM 理论提速数倍；
   改动一行 dtype，但需数值回归且破坏与历史 bf16 产物的逐位可比性）。
3. 微优化（可选，合计 <15%，不改变"≈1x"结论）：按长度分桶组 batch（省 7–10% pad）、
   向量化 k 调度循环、fp32 替代 fp64 采样（会改 RNG 口径，慎动）。

## 8. 断点重连步骤（新会话照此恢复）

1. `tmux ls`：若 `batch_accept` 还在 → 只读 §4 的命令看进度，**不要重复启动**。
2. 若会话已退出：`cat outputs/batch_accept/accept_main.log` 看最后节点 +
   对照 §6 产物表确定完成到哪步。
3. 已完成的步骤不要重跑；未完成的步骤从 `accept.sh` 里抄对应 `driver.py` 命令单独补跑
   （或注释掉 accept.sh 中已完成的 run 行后重启 tmux）。
4. 若 `outputs/batch_accept/` 整个丢失：driver.py / accept.sh 的完整内容以本文档 §2/§3
   描述为准重写；`accept10.csv` 重建命令：

```bash
/root/miniconda3/bin/python - <<'EOF'
import json, csv
rows = json.load(open("DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json"))
with open("outputs/batch_accept/accept10.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["prompt"]); w.writeheader()
    for r in rows[:10]:
        w.writerow({"prompt": r.get("goal", r["refined_goal"])})
EOF
```

5. 相关代码分支勿动：验收期间不要在 worktree 上 rebase/切分支（driver 直接 import
   该目录的 `eval_llada_steering.py`）。
