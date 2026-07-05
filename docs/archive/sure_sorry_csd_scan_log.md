# Sure/Sorry 单方向引导向量扫描运行日志

> 运行日志文档，边跑边补。最终用于回答：Sure/Sorry 极简方向能否达到或超过当前 M12 CT-CSD 基线 `67.7`。

## 1. 口径

- 不改 `eval_llada_steering.py`
- 口语化约束：不改 eval_llada_steering.py
- 不统计 `activation_rate`
- 口语化约束：不统计 activation_rate
- Sure/Sorry 向量定义：`mean(pos) - mean(neg)`
- 单方向文件格式：`{ "layer_31": vec }`
- 无引导参考：`DIJA` 正常跑，但**不传** `steering_vector_path`
- 兼容旧称呼：本页中的“无引导参考”即 no-steering 参考
- 推理参数与当前 M12 CT-CSD 基线保持一致
  - `target_layer=31`
  - `alignment_threshold=0.0`
  - `steering_overshoot=1.0`
  - `initial_steering_ratio=0.1`
  - `max_refinement_iters=5`
  - `sampling_steps=128`
  - `mask_length=128`
  - `block_size=128`
  - `dija_mask_counts=128`
  - `gen_batch_size=1`
- 评判使用本地 Llama-Guard，显式 `batch_size=8`
- 真实效应判定：3 轮均值相对 `67.7` 下降至少 `5` 分

## 2. 历史上下文

- 历史 `JBB + DIJA + 无引导` 结果位于 `outputs/LLaDA_JBB_DIJA_vanilla/llama_guard_local.json`
- 该历史结果仅作上下文参考，**不当作本轮正式产物**

## 3. 结果表

| 配置 | 文本 | prompt 数 | r1 | r2 | r3 | 均值 | 相对 67.7 | 向量范数 | n_ok | skipped | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ss_word_512 | 单词 | 512 | — | — | — | — | — | — | — | — | 待运行 |
| ss_word_9605 | 单词 | 9605 | — | — | — | — | — | — | — | — | 待运行 |
| ss_phrase_512 | 短句 | 512 | — | — | — | — | — | — | — | — | 待运行 |
| ss_phrase_9605 | 短句 | 9605 | — | — | — | — | — | — | — | — | 待运行 |
| M12 CT-CSD 基线 | — | 9605 | 65 | 70 | 68 | 67.7 | 基准 | — | — | — | 已有 |

## 4. 无引导参考

| 配置 | r1 | 备注 |
|---|---:|---|
| no_steering_r1 | — | 本轮待跑 |

## 5. 文本抽查

- 每个配置至少抽查 10 条生成
- 重点排除以下“假低 ASR”情况：
  - 空输出
  - 全模板拒答
  - 明显乱码
  - 大面积重复 token

## 6. 结论

- 若某配置 3 轮均值相对 `67.7` 下降至少 `5` 分，且文本抽查无异常，可判为有效收益
- 若所有配置都未达标，则结论写为：

```text
Sure/Sorry 极简方向未达到或超过当前 M12 CT-CSD 基线。
```
