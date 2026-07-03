"""批量推理基础工具：右 pad 序列构造、有效区 attention mask、逐行 top-k 揭码。

设计约束（详见 docs/plan）：
- LLaDA 的 RoPE 位置由 token 下标隐式决定（forward 无 position_ids 参数），
  因此只能右 pad——右 pad 不改变真实 token 的位置编码，左 pad 会平移。
- modeling_llada 对全 1 的 attention_mask 会直接丢弃（走与不传 mask 相同的
  kernel 路径），所以 forward_logits 仅在存在 pad 时才传 attention_mask，
  保证 batch=1 / 等长 batch 与原单样本实现逐位一致。
"""

import torch


def forward_logits(model, xt, valid=None):
    """跑一次模型前向并返回 logits。

    仅当 valid 含 0（存在 pad 位置）时才传 attention_mask；
    无 pad 时保持 model(xt) 的原调用形式与 kernel 路径。
    """
    if valid is not None and not bool(valid.all()):
        return model(xt, attention_mask=valid.to(torch.float)).logits
    return model(xt).logits


def extract_layer_hidden(model, layer_module, xt, valid=None):
    """通过 forward hook 抓取指定层输出的 hidden state，形状 [B, L, D]。"""
    buffer = [None]

    def _hook(module, inputs, output, _buf=buffer):
        h = output[0] if isinstance(output, tuple) else output
        _buf[0] = h.detach()
        return output

    handle = layer_module.register_forward_hook(_hook)
    try:
        forward_logits(model, xt, valid)
    finally:
        handle.remove()
    return buffer[0]


def build_padded_xt(prompts, mask_length, mask_id, pad_id, inject_prompt=False, device=None):
    """把变长 prompt 批量组装成右 pad 的去噪初始序列。

    每行布局：prompt / (inject_prompt 时再复制一份 prompt) / mask_length 个 mask / pad。

    Args:
        prompts: list[LongTensor]，每个元素为 1-D 的 prompt token 序列。
        mask_length: 生成区长度。
        mask_id: mask token id。
        pad_id: pad 填充值（约定用 eos），不得与 mask_id 相同。
        inject_prompt: 是否在生成区前再放一份 prompt 拷贝。
        device: 目标设备。

    Returns:
        (xt [B, Lmax], valid [B, Lmax] bool, prompt_lens [B], total_lens [B])。
        prompt_lens 为生成区起点（inject_prompt 时为 2P）。
    """
    assert pad_id != mask_id, "pad_id 不能与 mask_id 相同，否则 pad 会被当作待生成位置"
    p_lens = [int(p.shape[0]) for p in prompts]
    gen_starts = [2 * l if inject_prompt else l for l in p_lens]
    totals = [start + mask_length for start in gen_starts]
    max_len = max(totals)

    xt = torch.full((len(prompts), max_len), pad_id, dtype=torch.long, device=device)
    valid = torch.zeros((len(prompts), max_len), dtype=torch.bool, device=device)
    for b, prompt in enumerate(prompts):
        prompt = prompt.to(device)
        xt[b, : p_lens[b]] = prompt
        if inject_prompt:
            xt[b, p_lens[b] : 2 * p_lens[b]] = prompt
        xt[b, gen_starts[b] : totals[b]] = mask_id
        valid[b, : totals[b]] = True

    prompt_lens = torch.tensor(gen_starts, dtype=torch.long, device=device)
    total_lens = torch.tensor(totals, dtype=torch.long, device=device)
    return xt, valid, prompt_lens, total_lens


def pad_token_rows(rows, pad_id=None, device=None):
    """把变长 token 行右 pad 成 [B, Lmax]（DIJA 等已含 mask 的完整序列用）。

    pad_id 为 None 时要求所有行等长（此时无需 pad 值）。

    Returns:
        (xt [B, Lmax], valid [B, Lmax] bool, total_lens [B])。
    """
    lens = [int(r.shape[0]) for r in rows]
    max_len = max(lens)
    if pad_id is None:
        assert min(lens) == max_len, "存在变长行时必须提供 pad_id"
        pad_id = 0

    xt = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    valid = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
    for b, row in enumerate(rows):
        xt[b, : lens[b]] = row.to(device)
        valid[b, : lens[b]] = True
    return xt, valid, torch.tensor(lens, dtype=torch.long, device=device)


def rowwise_topk_transfer(confidence, k):
    """逐行选出 confidence 最高的 k 个位置，返回 bool [B, L]。

    与逐行 torch.topk + 逐行 scatter 等价（B=1 时与原实现选择一致）。
    """
    k = min(int(k), confidence.shape[-1])
    _, select_index = torch.topk(confidence, k=k, dim=-1)
    transfer = torch.zeros_like(confidence, dtype=torch.bool)
    transfer.scatter_(1, select_index, True)
    return transfer
