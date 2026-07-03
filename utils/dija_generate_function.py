"""Local compatibility copy of DIJA's generate_llada with tokens_per_step support.

Source:
https://raw.githubusercontent.com/ZichenWen1/DIJA/main/run_harmbench/utility/generate_function.py
plus the local tokens_per_step patch tracked in scripts/patches/dija_tokens_per_step.patch.
"""

import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    eps = 1e-9
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    return logits + temperature * gumbel_noise


def generate_llada(
    input_ids,
    attention_mask,
    model,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    tokens_per_step=1,
):
    del gen_length, block_length, cfg_scale, remasking
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        assert batch_size == 1
        x = input_ids
        num_transfer_tokens = tokens_per_step
        while (x == mask_id).any():
            mask_index = x == mask_id
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0 = torch.where(mask_index, x0, x)

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            confidence = torch.where(mask_index, x0_p, -float("inf"))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                available = int(mask_index[j].sum().item())
                k = min(num_transfer_tokens, available)
                if k <= 0:
                    continue
                select_index = torch.topk(confidence[j], k=k).indices
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
        return x
