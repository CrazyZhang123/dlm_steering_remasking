# DLM Steering 实验总表

> 状态：截至 `2026-07-05` 的统一结果汇总。本文档收口阶段结果、运行时间、ASR 指标与主要消融结论；旧的进度/日志/metrics 文档统一归档到 `docs/archive/`。

## 1. 统一口径

- 生成口径：`JBB + DIJA`
- 样本数：默认 `100`
- 评判器：本地 `Llama-Guard-4-12B`
- 指标优先级：
  - 单轮结果：用于记录某次实验的直接输出
  - 三轮均值：用于判定真实效果
- 当前全项目统一比较基线：
  - `CT-CSD M12` 三轮 `65 / 70 / 68`
  - 均值 **67.7**
- 判读规则：
  - 单次推理噪声约 `±2.5pp`
  - 三轮均值相对 `67.7` 下降至少 `5pp` 才视为明确收益
- 本地补充来源：
  - `docs/session_notes_20260630_0702_stage4a_tokensel.md` 中关于 `M12` 重跑 `65/70/68`
    与“推理天然非确定”的判断，已并入本文的噪声口径
  - `docs/stage5_ct_csd_pca128_m_scan_12_9_10_11_log.md` 中关于 `center_pca128_l2`
    的 `70/70/72` 结果，已并入 Stage 5 汇总
  - `docs/batch_inference_gpu_acceptance_log.md` 属于工程性能验收，不纳入方法 ASR 主表

## 2. 主结果总览

> 说明：运行时间只写文档里明确出现的实测值或估计值；未在旧文档里单独记时的项目，统一标记为“未单列”。

| 家族 | 代表配置 | 构造时间 | 单轮推理时间 | 单轮评判时间 | 结果 | 当前结论 |
|---|---|---:|---:|---:|---|---|
| Stage 0 | Global Sentence-CSD | 未单列 | 未单列 | 未单列 | `74/100`，`ASR 74.0%` | 冻结历史 baseline，仅作起点 |
| Stage 1 | CT-CSD M16 | 未单列 | 未单列 | 未单列 | `74/100`，`ASR 74.0%` | 最小 CT-CSD 闭环跑通，但不优 |
| Stage 2 | CT-CSD M12（最佳簇数点） | 相邻扫描实测约 `4.8h/bank` | 约 `1h/轮` | 约 `1-2min/轮` | 单轮 `65.0%`；三轮均值 **67.7** | 当前统一主基线 |
| Stage 3 | Category-aware CT-CSD M10 | 未单列 | 未单列 | 未单列 | `67/100`，`ASR 67.0%` | 当前最佳单轮结果；优于 Stage 3 其他 M |
| Stage 4 | MIL probe M16 `tau=0.7` | 未单列 | 未单列 | 未单列 | `71/100`，`ASR 71.0%` | 激活率下降，但 ASR 无收益 |
| Stage 4A | direction M16 `r=0.5/global` | 文档仅记“单 config 数小时级” | 同上 | 同上 | `72/100`，`ASR 72.0%` | 略差于 Stage 3 M16 baseline |
| Stage 4A | direction M16 `r=0.3/category` | 文档仅记“单 config 数小时级” | 同上 | 同上 | `74/100`，`ASR 74.0%` | 更差 |
| Stage 4A/4C | 全局 M12 direction/random | 未单列 | 未单列 | 未单列 | direction `68.0%`；random `70.0%` | direction 略好于 random，但都劣于 M12 baseline |
| Stage 4D | KNN label-clean | bank 已构造 | 未跑推理 | 未跑评判 | retention `99.14%~99.80%` | 几乎不删 token，预期无信息增益 |
| Stage 5 | `center_l2` / `center_pca128_l2` / `center_pca256_l2` | `center_l2` 总计约 `10.5h`；`center_pca256_l2` 总计约 `8h` | 约 `1h/轮` | 约 `1-2min/轮` | 最佳是 `pca256 = 68.0` | 预处理线无正收益 |
| Steering 超参 | `overshoot` / `isr` 扫描 | 不重建 bank | 约 `61m/轮` | 约 `1m/轮` | 最佳 `isr0.2 = 70.0` | 全部差于 `67.7` |
| Sure/Sorry | `ss_phrase_512`（最佳） | `512` 向量为分钟级；`9605` 向量约 `2h` | 约 `61m/轮` | 约 `1m/轮` | `69/72/71`，均值 `70.67` | 极简方向失败 |

## 3. 簇数消融汇总

### 3.1 全局 CT-CSD（Stage 1 / 2）

| 配置 | 单轮 / 三轮 | ASR | 备注 |
|---|---|---:|---|
| M4 | 单轮 | 70.0 | 已完成 |
| M8 | 单轮 | 71.0 | 已完成 |
| M12 | 单轮 | 65.0 | 阶段 1 / 2 最低单轮点 |
| M16 | 单轮 | 74.0 | 闭环主实验点 |
| M9 | 三轮 `70/72/65` | 69.0 | 与 M12 三轮均值持平偏差内 |
| M10 | 三轮 `71/70/73` | 71.33 | 更差 |
| M11 | 三轮 `73/71/71` | 71.67 | 更差 |
| M12 基线重跑 | 三轮 `65/70/68` | **67.7** | 全项目统一均值基线 |

**结论：**

- `M12` 维持为全局 CT-CSD 的最佳簇数点
- `M9/M10/M11` 细扫没有打过 `M12`

### 3.2 Category-aware CT-CSD（Stage 3）

| 配置 | ASR | 备注 |
|---|---:|---|
| Category-aware M4 | 70.0 | 优于全局 M16 |
| Category-aware M8 | 70.0 | 稳定优于全局 M16 |
| Category-aware M10 | **67.0** | Stage 3 最佳 |
| Category-aware M12 | 69.0 | 次优 |
| Category-aware M16 | 71.0 | 初始对照点 |

**结论：**

- `Category-aware M10` 是当前最佳单轮方法
- Category-aware 系列整体优于 `CT-CSD M16`

## 4. Token Selection 消融汇总

| 方法 | 配置 | ASR | 备注 |
|---|---|---:|---|
| MIL probe | Category-aware M16 `tau=0.7` direct | 71.0 | 与 no-probe 持平 |
| MIL probe | Category-aware M16 `tau=0.7` 普通 pipeline | 71.0 | 与 no-probe 持平 |
| direction | M16 `r=0.5 / global` | 72.0 | 略差 |
| direction | M16 `r=0.3 / category` | 74.0 | 更差 |
| direction | 全局 M12 `r=0.5` | 68.0 | 略差于 M12 baseline |
| random | 全局 M12 `r=0.5` | 70.0 | 更差于 direction |
| KNN | 平衡 / 非平衡 KNN | 未评测 ASR | retention 过高，近似无筛选 |

**结论：**

- `direction` 比 `random` 更有信号，但仍然打不过“全部 token”
- `MIL` 没有把降低激活率转成更低 ASR
- `KNN` 在当前设定下几乎不删 token，方法失效

## 5. Feature Preprocess 消融汇总（Stage 5）

| 配置 | 三轮 ASR | 均值 | 相对 `67.7` |
|---|---|---:|---:|
| `l2_only` | `65/70/68` | **67.7** | 基线 |
| `center_pca128_l2` | `70/70/72` | 70.67 | +3.0 |
| `center_l2` | `72/71/74` | 72.33 | +4.6 |
| `center_pca256_l2` | `68/66/70` | 68.0 | +0.3 |

**结论：**

- “去均值”本身有害
- `pca256` 只能追平，不超过 `l2_only`
- Stage 5 预处理线无正收益，已可盖棺

## 6. Steering 超参消融汇总

> 基线：`overshoot=1.0`, `isr=0.1`，三轮均值 `67.7`

| 配置 | 三轮 ASR | 均值 | 相对 `67.7` |
|---|---|---:|---:|
| `os1.5` | `73/74/69` | 72.0 | +4.3 |
| `os2.0` | `69/74/74` | 72.33 | +4.63 |
| `isr0.2` | `70/70/70` | 70.0 | +2.3 |
| `isr0.3` | `70/73/71` | 71.33 | +3.63 |

**结论：**

- “更强 / 更早”的 steering 在当前口径下全为负收益
- 没有任何点位优于 `67.7`

## 7. Sure/Sorry 极简方向消融汇总

> 参考：`no_steering_r1 = 70.0`

| 配置 | 三轮 ASR | 均值 | 相对 `67.7` | 备注 |
|---|---|---:|---:|---|
| `ss_word_512` | `84/78/84` | 82.0 | +14.3 | 明显失败 |
| `ss_word_9605` | `83/81/84` | 82.67 | +14.97 | 明显失败 |
| `ss_phrase_512` | `69/72/71` | **70.67** | +2.97 | 该线最佳，但仍失败 |
| `ss_phrase_9605` | `74/76/71` | 73.67 | +5.97 | 明显失败 |

向量构造时间（文档可证）：

| 配置 | 向量构造时间 | 备注 |
|---|---:|---|
| `ss_word_512` | `6m16s` | 2026-07-04 `14:18:10 -> 14:24:26` |
| `ss_word_9605` | `1h57m59s` | 2026-07-04 `17:30:31 -> 19:28:30` |
| `ss_phrase_9605` | `2h00m58s` | 2026-07-04 `16:28:50 -> 18:29:48` |
| `ss_phrase_512` | 未单独保留首轮主日志 | 仅确认已完成并可复用 |

单轮推理 + 评判时间（同脚本、100 样本）：

| 项目 | 推理时间 | 评判时间 |
|---|---:|---:|
| `no_steering_r1` | `54m45s` | `58s` |
| `ss_phrase_512 r1` | `60m45s` | `56s` |
| `ss_word_512 r1` | `62m04s` | `53s` |
| `ss_word_9605 r1` | `60m42s` | `55s` |
| `ss_phrase_9605 r1` | `60m53s` | `53s` |

**结论：**

- `Sure/Sorry` 极简方向没有接近，更没有超过 `M12 baseline = 67.7`
- 该方向可判定为负结果

## 8. 当前推荐结论

1. 当前统一比较基线仍是 `CT-CSD M12` 三轮均值 **67.7**
2. 当前最佳单轮结果是 `Category-aware CT-CSD M10 = 67.0`
3. 以下方向可暂时视为已证伪或已无继续投入价值：
   - token selection（MIL / direction / random / KNN）
   - feature preprocess（center / PCA）
   - 更强 / 更早的 steering 超参
   - Sure/Sorry 极简方向

## 9. 旧文档归档范围

以下文档内容已并入本总表，后续仅保留归档用途：

- `docs/category_aware_ct_csd_stage_progress.md`
- `docs/category_aware_ct_csd_stage1_metrics.md`
- `docs/stage3_category_ct_csd_metrics.md`
- `docs/stage4_mil_token_probe_progress.md`
- `docs/stage2_ct_csd_m_scan_9_10_11_log.md`
- `docs/stage4_abcd_progress.md`
- `docs/stage5_m12_preprocess_ablation_log.md`
- `docs/steering_hparam_scan_log.md`
- `docs/sure_sorry_csd_scan_log.md`

如需追溯原始时间线、逐轮日志或诊断文件，统一去 `docs/archive/` 与 `outputs/` 查看。

## 10. 本地保留归档说明

主工作区历史上存在一批未提交的实验日志与会话笔记。为保护本地现场，这些文件不删除，
统一放入 `docs/archive/local_preserved/`，同时将其中会影响方法结论的内容吸收到本文：

- `session_notes_20260630_0702_stage4a_tokensel.md`
  - 已吸收：M12 三轮基线噪声判断、token-selection 路线关闭结论
- `stage2_ct_csd_m_scan_9_10_11_log.md`
  - 已吸收：M9/M10/M11 三轮均值与运行时间
- `stage4_abcd_progress.md`
  - 已吸收：direction / random / KNN 的结论
- `stage5_ct_csd_pca128_m_scan_12_9_10_11_log.md`
  - 已吸收：`center_pca128_l2` 的 M12 三轮结果
- `stage5_m12_preprocess_ablation_log.md`
  - 已吸收：`center_l2` / `center_pca256_l2` 的三轮结果与时间
- `steering_hparam_scan_log.md`
  - 已吸收：`overshoot / isr` 扫描结果
- `batch_inference_gpu_acceptance_log.md`
  - 仅保留工程验收用途，不进入方法 ASR 比较
