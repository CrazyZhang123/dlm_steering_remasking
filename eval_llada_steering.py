import accelerate
import torch
import pandas as pd
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from utils import REFERENCES, SAFE_REMINDER
from utils.batching import (
    build_padded_xt,
    extract_layer_hidden,
    forward_logits,
    pad_token_rows,
    rowwise_topk_transfer,
)
from utils.ct_csd_bank import CTCSDBank
from transformers import AutoTokenizer, AutoModel
import json
import os
import re
import importlib.util


DIJA_MASK_PATTERN = r'<mask:(\d+)>'
DIJA_MASK_TOKEN_STR = '<|mdm_mask|>'
DIJA_START_TOKEN = '<startoftext>'
DIJA_END_TOKEN = '<endoftext>'


def save_partial_results(output_dir: str, rows: list[dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results.partial.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_harmbench_prefix_templates():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "DIJA",
        "benchmarks",
        "HarmBench",
        "baselines",
        "human_jailbreaks",
        "jailbreaks.py",
    )
    spec = importlib.util.spec_from_file_location("harmbench_human_jailbreaks", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return list(module.JAILBREAKS)


def select_harmbench_prefix_templates(prefixes, random_subset=5, seed=1):
    selected = list(prefixes)
    if isinstance(random_subset, int):
        rng = random.Random(seed)
        rng.shuffle(selected)
        if random_subset >= 0:
            selected = selected[:random_subset]
    return selected


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


# ==================================================================
# 批量采样核心（模块级函数，接收 harness 对象）
#
# 实现为模块级函数而非类方法，原因有二：
# 1. 单样本方法与批量方法共享同一实现（DRY），batch=1 且无 pad 时与
#    历史单样本实现逐位等价（forward 调用形式、RNG 消耗形状均一致）；
# 2. 现有测试以未绑定方式（LLaDAEvalHarness.method(duck_dummy, ...)）
#    调用类方法，模块级函数间互调不依赖 self 上的方法解析。
#
# 批量化关键约定：
# - 变长 prompt 右 pad（pad 值 = eos），attention_mask 屏蔽 pad key；
#   RoPE 位置由下标决定，右 pad 不改变真实 token 的位置编码。
# - 原实现的 eos 提前退出（整条 return）改为按行冻结（done mask），
#   已冻结行不再参与揭码 / steering / Phase 2。
# - Phase 2 的逐样本提前退出（n_harmful == 0 即 break）改为活跃行
#   压缩：已收敛的行退出 batch，剩余行继续 refinement。
# ==================================================================


def _call_refill(harness, xt_rows, block_start, block_end, valid_rows):
    # 仅在存在 pad 时传 valid，保持无 pad 路径与旧签名的调用形式一致
    if valid_rows is not None and not bool(valid_rows.all()):
        return harness._refill_masks_with_steering(xt_rows, block_start, block_end, valid=valid_rows)
    return harness._refill_masks_with_steering(xt_rows, block_start, block_end)


@torch.no_grad()
def _refine_block_batch(harness, xt, valid, blk_start, blk_end, done):
    """Phase 2 批量版：对刚解码完的 block 做安全检测 + 重掩码（原地更新 xt）。

    与单样本语义等价：每行独立执行"检测→无害则收敛退出→有害则 remask+refill"，
    最多 max_refinement_iters 轮；收敛行从活跃集中移除（行压缩），后续轮次的
    前向只包含仍有有害 token 的行。
    """
    active = (~done).nonzero(as_tuple=False).squeeze(-1)
    block_size = harness.block_size
    layer_module = harness.model.model.transformer.blocks[harness.target_layer]
    col_offsets = torch.arange(block_size, device=xt.device)

    for _ in range(harness.max_refinement_iters):
        if active.numel() == 0:
            return

        xt_active = xt[active]
        valid_active = valid[active]
        hidden = extract_layer_hidden(harness.model, layer_module, xt_active, valid_active)
        col_idx = blk_start[active].unsqueeze(1) + col_offsets.unsqueeze(0)          # [Na, block_size]
        block_hidden = hidden.gather(1, col_idx.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        per_token_alignment = harness._per_token_alignment(block_hidden)             # [Na, block_size]

        harmful_mask = per_token_alignment > harness.alignment_threshold
        still_harmful = harmful_mask.reshape(active.numel(), -1).any(dim=1)          # [Na]
        if not bool(still_harmful.any()):
            return

        rows = active[still_harmful]
        remask = torch.zeros((rows.numel(), xt.shape[1]), dtype=torch.bool, device=xt.device)
        remask.scatter_(1, col_idx[still_harmful], harmful_mask[still_harmful])
        full_harmful_mask = torch.zeros_like(xt, dtype=torch.bool)
        full_harmful_mask[rows] = remask
        xt[full_harmful_mask] = harness.mask_id

        xt[rows] = _call_refill(harness, xt[rows], blk_start[rows], blk_end[rows], valid[rows])
        active = rows


@torch.no_grad()
def _block_decode_batch(harness, xt, valid, prompt_lens, enable_steering):
    """Phase 1 逐 block 解码（+ 可选 steering 与 Phase 2）的批量核心。

    xt: [B, Lmax] 右 pad 序列；valid: [B, Lmax]；prompt_lens: [B] 生成区起点。
    B=1 且无 pad 时与原单样本实现逐位等价（同种子下 RNG 消耗一致）。
    """
    batch, max_len = xt.shape
    pos = torch.arange(max_len, device=xt.device)

    assert harness.mask_length % harness.block_size == 0
    num_blocks = harness.mask_length // harness.block_size

    assert harness.sampling_steps % num_blocks == 0
    steps = harness.sampling_steps // num_blocks

    assert harness.mask_length % harness.sampling_steps == 0

    can_steer = enable_steering and harness._can_steer()
    initial_steering_steps = int(steps * harness.initial_steering_ratio) if can_steer else 0
    k = harness.mask_length // harness.sampling_steps
    eos_id = harness.tokenizer.eos_token_id
    done = torch.zeros(batch, dtype=torch.bool, device=xt.device)

    for num_block in range(num_blocks):
        blk_start = prompt_lens + num_block * harness.block_size                     # [B]
        blk_end = blk_start + harness.block_size                                     # [B]
        in_block = (pos.unsqueeze(0) >= blk_start.unsqueeze(1)) & (pos.unsqueeze(0) < blk_end.unsqueeze(1))

        for i in range(steps):
            mask_index = (xt == harness.mask_id)

            hook_handle = None
            if can_steer and i < initial_steering_steps:
                block_mask_index = mask_index & in_block & ~done.unsqueeze(1)
                if block_mask_index.any():
                    steering_hook = harness._build_adaptive_steering_hook(block_mask_index)
                    hook_handle = harness.model.model.transformer.blocks[harness.target_layer].register_forward_hook(steering_hook)

            try:
                logits = forward_logits(harness.model, xt, valid)
            finally:
                if hook_handle is not None:
                    hook_handle.remove()

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0 = _sample_categorical(p)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            # 逐行 block 边界约束：只允许在各自 block 范围内产生新 token
            x0_p[pos.unsqueeze(0) >= blk_end.unsqueeze(1)] = -np.inf
            x0 = torch.where(mask_index, x0, xt)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = rowwise_topk_transfer(confidence, k) & ~done.unsqueeze(1)
            xt[transfer_index] = x0[transfer_index]

        if can_steer:
            _refine_block_batch(harness, xt, valid, blk_start, blk_end, done)

        # 原实现在 block 完成后发现 eos 即整条 return；批量版冻结该行
        done |= ((xt == eos_id) & valid).any(dim=1)
        if bool(done.all()):
            break

    return xt


@torch.no_grad()
def _sample_blocks_batch(harness, prompts, inject_prompt, enable_steering):
    """构造右 pad 批量序列并跑批量 block 解码，返回按行截去 pad 的 1-D 序列列表。"""
    device = harness.model.device
    xt, valid, prompt_lens, total_lens = build_padded_xt(
        prompts,
        mask_length=harness.mask_length,
        mask_id=harness.mask_id,
        pad_id=harness.tokenizer.eos_token_id,
        inject_prompt=inject_prompt,
        device=device,
    )
    xt = _block_decode_batch(harness, xt, valid, prompt_lens, enable_steering)
    return [xt[b, : int(total_lens[b])] for b in range(xt.shape[0])]


@torch.no_grad()
def _dija_sample_batch(harness, refined_goals, vanilla_goals, mask_counts):
    """DIJA 模式批量采样，返回 [(generated_ids_1d, matching_count), ...]。"""
    device = harness.device
    rows, matching_counts = [], []
    for refined_goal, vanilla_goal in zip(refined_goals, vanilla_goals):
        refined_text = harness._build_dija_prompt_text(refined_goal, mask_counts)
        refined_ids = harness.tokenizer(refined_text, return_tensors="pt")["input_ids"].to(device)

        if vanilla_goal is not None:
            vanilla_text = harness.tokenizer.apply_chat_template(
                [{"role": "user", "content": vanilla_goal}],
                tokenize=False,
                add_generation_prompt=True,
            )
            vanilla_ids = harness.tokenizer(vanilla_text, return_tensors="pt")["input_ids"].to(device)
            common_len = min(vanilla_ids.shape[1], refined_ids.shape[1])
            eq = (vanilla_ids[0, :common_len] == refined_ids[0, :common_len])
            first_diff = (~eq).nonzero(as_tuple=False)
            matching_count = int(first_diff[0].item()) if first_diff.numel() > 0 else common_len
        else:
            matching_count = 0

        rows.append(refined_ids[0])
        matching_counts.append(matching_count)

    lens = {int(r.shape[0]) for r in rows}
    pad_id = getattr(harness.tokenizer, "eos_token_id", None) if len(lens) > 1 else None
    xt, valid, total_lens = pad_token_rows(rows, pad_id=pad_id, device=device)
    original_mask_index = (xt == harness.mask_id)

    can_steer = harness._can_steer()
    steps = harness.sampling_steps
    initial_steering_steps = int(steps * harness.initial_steering_ratio) if can_steer else 0

    for i in range(steps):
        mask_index = (xt == harness.mask_id)
        if not mask_index.any():
            break

        hook_handle = None
        if can_steer and i < initial_steering_steps:
            steering_hook = harness._build_adaptive_steering_hook(mask_index)
            hook_handle = harness.model.model.transformer.blocks[harness.target_layer].register_forward_hook(steering_hook)

        try:
            logits = forward_logits(harness.model, xt, valid)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0 = _sample_categorical(p)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

        x0 = torch.where(mask_index, x0, xt)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        steps_remaining = steps - i
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            remaining = int(mask_index[j].sum().item())
            if remaining == 0:
                continue
            k = max(1, (remaining + steps_remaining - 1) // steps_remaining)
            k = min(k, remaining)
            _, select_index = torch.topk(confidence[j], k=k)
            transfer_index[j, select_index] = True
        xt[transfer_index] = x0[transfer_index]

    # ---- Phase 2 (DIJA): 全部解码完成后做 batch remask ----
    # 不同于 block 模式按 block 做 remask，DIJA 对整个序列一次性检测+remask
    # harmful_mask 用 original_mask_index 过滤，只对初始就是 mask 的位置做 remask
    if can_steer:
        active = original_mask_index.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
        layer_module = harness.model.model.transformer.blocks[harness.target_layer]
        for refinement_iter in range(harness.max_refinement_iters):
            if active.numel() == 0:
                break
            xt_active = xt[active]
            valid_active = valid[active]
            original_mask_active = original_mask_index[active]

            full_hidden = extract_layer_hidden(harness.model, layer_module, xt_active, valid_active)
            if harness.steering_bank is not None:
                alignment = harness.steering_bank.alignment(
                    full_hidden,
                    theta=harness.alignment_threshold,
                    record=False,
                )
                _ = harness.steering_bank.alignment(
                    full_hidden[original_mask_active],
                    theta=harness.alignment_threshold,
                    record=True,
                )
            else:
                sv = harness.steering_vector.to(full_hidden.device, dtype=full_hidden.dtype)
                sv_unit = sv / (sv.norm() + 1e-8)
                alignment = (full_hidden * sv_unit).sum(dim=-1)

            harmful_mask = (alignment > harness.alignment_threshold) & original_mask_active
            still_harmful = harmful_mask.any(dim=1)
            if not bool(still_harmful.any()):
                break

            rows_idx = active[still_harmful]
            xt_rows = xt[rows_idx]
            xt_rows[harmful_mask[still_harmful]] = harness.mask_id
            xt[rows_idx] = _call_refill(harness, xt_rows, 0, xt.shape[1], valid[rows_idx])
            active = rows_idx

    return [
        (xt[b, : int(total_lens[b])], matching_counts[b])
        for b in range(xt.shape[0])
    ]


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=128,
        generated_samples_path='',
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        sampling_steps=128,
        mask_length=128,
        block_size=32,
        remasking='low_confidence',
        device="cuda",
        sampler='',
        remdm_number=0,
        steering_vector_path='',
        steering_overshoot=1.0,
        target_layer=24,
        alignment_threshold=0.1,
        max_refinement_iters=5,
        initial_steering_ratio=0.5,
        dija_mask_counts=36,
        inject_prompt=True,
        attack_method='zeroshot',
        gen_batch_size=1,
    ):
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.generated_samples_path = generated_samples_path
        self.sampler = sampler
        self.attack_method = attack_method
        self.remdm_number = remdm_number

        self.cfg = cfg
        self.sampling_steps = sampling_steps
        self.mask_length = mask_length
        self.block_size = block_size
        self.remasking = remasking
        print(self.generated_samples_path)

        self.steering_vector = None
        self.steering_bank = None
        self.steering_overshoot = steering_overshoot
        self.target_layer = target_layer
        self.alignment_threshold = alignment_threshold
        self.max_refinement_iters = max_refinement_iters
        self.initial_steering_ratio = initial_steering_ratio
        self.dija_mask_counts = dija_mask_counts
        self.inject_prompt = inject_prompt
        self.gen_batch_size = int(gen_batch_size)
        if inject_prompt:
            print("inject_prompt enabled: a copy of the prompt is placed at the front of the "
                  "generation region; only the trailing mask_length tokens are decoded as output.")
        if steering_vector_path:
            print(f"Loading steering vector from {steering_vector_path}")
            obj = torch.load(steering_vector_path, map_location="cpu", weights_only=True)
            if isinstance(obj, dict) and obj.get("format") == "ct_csd_v1":
                if int(obj["target_layer"]) != int(target_layer):
                    raise ValueError(
                        f"Bank target_layer={obj['target_layer']} does not match requested target_layer={target_layer}"
                    )
                self.steering_bank = CTCSDBank.from_state_dict(obj, device=self.device, dtype=torch.float32)
                print(f"CT-CSD bank loaded (layer {target_layer}, clusters={self.steering_bank.num_clusters})")
            else:
                key = f'layer_{target_layer}'
                if key not in obj:
                    available = list(obj.keys())
                    raise ValueError(f"Layer {target_layer} not found. Available: {available}")
                self.steering_vector = obj[key]
                print(f"Steering vector loaded (layer {target_layer}, norm={self.steering_vector.norm().item():.4f})")
            print(f"steering_overshoot(beta)={steering_overshoot}, alignment_threshold={alignment_threshold}, "
                  f"initial_steering_ratio={initial_steering_ratio}, max_refinement_iters={max_refinement_iters}")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _can_steer(self):
        return self.steering_vector is not None or self.steering_bank is not None

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    @torch.no_grad()
    def llada_conf_sample(self, prompt):
        # 单样本入口：等价于 batch=1 的批量实现（无 pad 时逐位一致）
        return _sample_blocks_batch(self, [prompt[0]], inject_prompt=False, enable_steering=False)[0].unsqueeze(0)

    @torch.no_grad()
    def llada_conf_sample_batch(self, prompts):
        # prompts: list[1-D LongTensor]，返回按行截去 pad 的 1-D 序列列表
        return _sample_blocks_batch(self, prompts, inject_prompt=False, enable_steering=False)

    @torch.no_grad()
    def _per_token_alignment(self, block_hidden):
        # 计算 block 内每个 token 的 hidden state 与 steering vector 的对齐程度
        # 返回正值越大，表示该 token 语义越接近有害方向
        if self.steering_bank is not None:
            return self.steering_bank.alignment(
                block_hidden,
                theta=self.alignment_threshold,
                record=True,
            )
        sv = self.steering_vector.to(block_hidden.device, dtype=block_hidden.dtype)
        #  block_hidden 形状为 [1, block_size, hidden_dim]，sv_unit 形状为 [hidden_dim]
        sv_unit = sv / (sv.norm() + 1e-8)
        return (block_hidden * sv_unit).sum(dim=-1)

    def _build_adaptive_steering_hook(self, mask_index):
        # 构建 adaptive steering forward hook
        # 对 mask 位置的 hidden state 做 steering 修正：
        #   - 先计算投影 a = hidden · sv_unit
        #   - 如果 a > theta，则减去 alpha_t = beta * (a - theta) 倍的 sv_unit
        #   - 如果 a <= theta，alpha_t = 0，不做扰动
        # 核心思想：有害对齐强烈 → 引导力度强；有害对齐较弱 → 最小干预
        beta = self.steering_overshoot
        theta = self.alignment_threshold
        sv_ref = self.steering_vector

        def steering_hook(module, input, output, _mask_index=mask_index, _beta=beta, _theta=theta, _sv=sv_ref):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            masked_h = hidden[_mask_index]                                  # [n_masked, D]
            if self.steering_bank is not None:
                hidden[_mask_index] = self.steering_bank.steer(
                    masked_h,
                    beta=_beta,
                    theta=_theta,
                ).to(hidden.dtype)
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

            sv = _sv.to(hidden.device, dtype=hidden.dtype)
            sv_unit = sv / (sv.norm() + 1e-8)
            a = (masked_h * sv_unit.unsqueeze(0)).sum(dim=-1)               # [n_masked]
            alpha_t = _beta * (a - _theta).clamp(min=0)                     # [n_masked]
            hidden[_mask_index] = masked_h - alpha_t.unsqueeze(-1) * sv_unit.unsqueeze(0)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return steering_hook

    @torch.no_grad()
    def _refill_masks_with_steering(self, xt, block_start, block_end, valid=None):
        # Phase 2 辅助方法：对当前 block 的 mask 位置做一次 forward pass，
        # 并用 adaptive steering 生成替换 token。
        # block_start/block_end 支持标量（全 batch 相同）或 [B] 张量（逐行边界）。
        pos = torch.arange(xt.shape[1], device=xt.device)
        if torch.is_tensor(block_start):
            in_block = (pos.unsqueeze(0) >= block_start.unsqueeze(1)) & (pos.unsqueeze(0) < block_end.unsqueeze(1))
        else:
            in_block = ((pos >= block_start) & (pos < block_end)).unsqueeze(0)
        mask_index = (xt == self.mask_id) & in_block
        if not mask_index.any():
            return xt

        steering_hook = self._build_adaptive_steering_hook(mask_index)
        hook_handle = self.model.model.transformer.blocks[self.target_layer].register_forward_hook(steering_hook)

        try:
            logits = forward_logits(self.model, xt, valid)
        finally:
            hook_handle.remove()

        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0 = _sample_categorical(p)

        xt = torch.where(mask_index, x0, xt)
        return xt

    @torch.no_grad()
    def llada_remask_sample(self, prompt):
        # ==============================================================
        # 两阶段安全生成：Adaptive Steering + Harmful Token Remasking
        #
        # Phase 1 (逐 block 解码):
        #   - 将 mask_length 划分为 num_blocks 个 block
        #   - 每个 block 内执行 steps 次 confidence-based 逐步解码
        #   - 每次揭掉 confidence 最高的 k = mask_length / sampling_steps 个 token
        #   - block 边界约束: 置信度在 block_end 之后置 -inf，确保不跨 block
        #   - 前 initial_steering_steps 步应用 adaptive steering hook
        #
        # Phase 2 (block 级安全检测 + remask):
        #   - 一个 block 全部 steps 次迭代完成后才进入 Phase 2
        #   - 提取 block 区域的 hidden states，计算 per-token alignment
        #   - 将对齐超过 alignment_threshold 的 token 标记为有害
        #   - 将有害 token 重新设回 mask_id，用 _refill_masks_with_steering 重新生成
        #   - 最多重试 max_refinement_iters 次，直到 block 内无有害 token
        #
        # 不是每生成一个 token 就去检测安全，
        # 而是等整个 block 的所有 token 解码完成后，才批量检测和 remask
        #
        # 实现位于模块级 _block_decode_batch / _refine_block_batch，
        # 单样本入口等价于 batch=1 的批量调用（无 pad 时逐位一致）。
        # ==============================================================
        return _sample_blocks_batch(self, [prompt[0]], inject_prompt=self.inject_prompt, enable_steering=True)[0].unsqueeze(0)

    @torch.no_grad()
    def llada_remask_sample_batch(self, prompts):
        # prompts: list[1-D LongTensor]，返回按行截去 pad 的 1-D 序列列表
        return _sample_blocks_batch(self, prompts, inject_prompt=self.inject_prompt, enable_steering=True)

    def _build_dija_prompt_text(self, refined_goal, mask_counts):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": refined_goal}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = re.sub(DIJA_MASK_PATTERN, lambda m: DIJA_MASK_TOKEN_STR * int(m.group(1)), prompt)
        if DIJA_MASK_TOKEN_STR not in prompt and mask_counts:
            prompt += DIJA_START_TOKEN + DIJA_MASK_TOKEN_STR * mask_counts + DIJA_END_TOKEN
        return prompt

    @torch.no_grad()
    def llada_dija_sample(self, refined_goal, vanilla_goal=None, mask_counts=36):
        # 单样本入口：等价于 batch=1 的批量实现（无 pad 时逐位一致）
        generated_ids, matching_count = _dija_sample_batch(
            self, [refined_goal], [vanilla_goal], mask_counts
        )[0]
        return generated_ids.unsqueeze(0), matching_count

    @torch.no_grad()
    def llada_dija_sample_batch(self, refined_goals, vanilla_goals, mask_counts=36):
        # refined_goals/vanilla_goals: 等长列表（vanilla 元素可为 None），
        # 返回 [(generated_ids_1d, matching_count), ...]
        return _dija_sample_batch(self, refined_goals, vanilla_goals, mask_counts)

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]):
        is_dija = self.attack_method == 'DIJA'
        # gen_batch_size 控制每次前向的样本数；1 时保持原单样本路径与行为
        gen_batch_size = max(1, int(getattr(self, "gen_batch_size", 1)))

        out, out_for_json = [], []
        progress = tqdm(total=len(requests))
        for chunk_start in range(0, len(requests), gen_batch_size):
            chunk = requests[chunk_start:chunk_start + gen_batch_size]

            # ---- 批量（或单样本）生成，得到每条请求的原始解码文本 ----
            if is_dija:
                refined_goals = [req.args[0] for req in chunk]
                vanilla_goals = [req.args[1].get("vanilla_goal", req.args[0]) for req in chunk]
                if len(chunk) == 1:
                    generated_ids, matching_count = self.llada_dija_sample(
                        refined_goals[0],
                        vanilla_goal=vanilla_goals[0],
                        mask_counts=self.dija_mask_counts,
                    )
                    pairs = [(generated_ids[0], matching_count)]
                else:
                    pairs = self.llada_dija_sample_batch(
                        refined_goals, vanilla_goals, mask_counts=self.dija_mask_counts
                    )
                raw_answers = []
                for generated_ids, matching_count in pairs:
                    generated_answer = self.tokenizer.decode(
                        generated_ids[matching_count:], skip_special_tokens=False
                    )
                    raw_answers.append(generated_answer.split("assistant\n")[0])
            else:
                prompts = [
                    torch.tensor(self.tokenizer(req.args[0])["input_ids"], dtype=torch.long)
                    for req in chunk
                ]
                if len(chunk) == 1:
                    prompt = prompts[0].unsqueeze(0).to(self.device)
                    if self.sampler == 'llada':
                        generated_ids = self.llada_conf_sample(prompt)
                    elif self.sampler == 'steering':
                        generated_ids = self.llada_remask_sample(prompt)
                    rows = [generated_ids[0]]
                else:
                    if self.sampler == 'llada':
                        rows = self.llada_conf_sample_batch(prompts)
                    elif self.sampler == 'steering':
                        rows = self.llada_remask_sample_batch(prompts)
                raw_answers = [
                    self.tokenizer.decode(row[-self.mask_length:], skip_special_tokens=False)
                    for row in rows
                ]

            # ---- 逐条后处理：stop token 截断 + 清理特殊 token ----
            for req, generated_answer in zip(chunk, raw_answers):
                stop_tokens = list(req.args[1]["until"]) + ["<|eot_id|>", self.tokenizer.eos_token]
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

                generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                out.append(generated_answer)
                out_for_json.append({
                    "prompt": req.args[0],
                    "response": generated_answer,
                })

            # partial 结果按 chunk 落盘（gen_batch_size=1 时与原每样本一次相同）
            if self.generated_samples_path:
                save_partial_results(self.generated_samples_path, out_for_json)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            progress.update(len(chunk))
        progress.close()

        return out

    def write_steering_diagnostics(self) -> None:
        if self.steering_bank is None:
            return
        if self.accelerator is not None and not getattr(self.accelerator, "is_main_process", getattr(self, "_rank", 0) == 0):
            return
        os.makedirs(self.generated_samples_path, exist_ok=True)
        out_path = os.path.join(self.generated_samples_path, "ct_csd_diagnostics.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(self.steering_bank.diagnostics(), handle, indent=2, ensure_ascii=False)
        print(f"Saved CT-CSD diagnostics to {out_path}")


def run_csv_eval(args):
    is_local_hf_dataset_dir = os.path.isdir(args.csv_path) and os.path.exists(
        os.path.join(args.csv_path, "dataset_dict.json")
    )

    if 'TruthfulQA' in args.csv_path:
        dataset = load_dataset("domenicrosati/TruthfulQA", split="train")
        question_key = "Question"
    elif 'JBB' in args.csv_path and args.attack_method == "PAP":
        with open("./gpt-oss/JBB_pap.json", "r") as f:
            dataset = json.load(f)
        question_key = "pap_prompt"
    elif 'JBB' in args.csv_path and args.attack_method == "DIJA":
        with open("./DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json", "r") as f:
            dataset = json.load(f)
        question_key = "refined_goal"
    elif 'JBB' in args.csv_path:
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")['harmful']
        question_key = "Goal"
    elif 'AdvBench' in args.csv_path and args.attack_method == "PAP":
        with open("./gpt-oss/AdvBench_pap.json", "r") as f:
            dataset = json.load(f)
        question_key = "pap_prompt"
    elif 'AdvBench' in args.csv_path and args.attack_method == "DIJA":
        with open("./DIJA/run_jailbreakbench/refine_prompt/advbench_data_refined_Qwen.json", "r") as f:
            dataset = json.load(f)
        question_key = "refined_goal"
    elif 'AdvBench' in args.csv_path:
        if is_local_hf_dataset_dir:
            dataset = load_from_disk(args.csv_path)["train"]
        else:
            dataset = load_dataset("walledai/AdvBench", split="train")
        question_key = "prompt"
    elif 'MATH' in args.csv_path:
        dataset = list(load_dataset("HuggingFaceH4/MATH-500")['test'])
        question_key = "problem"
    elif 'MMLU' in args.csv_path:
        subjects = [
            "logical_fallacies",
            "moral_scenarios",
            "philosophy",
        ]
        datasets_list = [
            load_dataset("cais/mmlu", subject, split="test")
            for subject in subjects
        ]
        dataset = list(concatenate_datasets(datasets_list))
        question_key = "question"
    else:
        df = pd.read_csv(args.csv_path)
        assert "prompt" in df.columns, "CSV must have a 'prompt' column"
        dataset = df.to_dict(orient="records")
        question_key = "prompt"

    request_rows = None

    sampler = 'llada_dija' if args.attack_method == "DIJA" else args.sampler

    model = LLaDAEvalHarness(
        model_path=args.model_path,
        generated_samples_path=args.generated_samples_path,
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
        mask_length=args.mask_length,
        block_size=args.block_size,
        remasking=args.remasking,
        sampler=sampler,
        remdm_number=args.remdm_number,
        cfg=args.cfg,
        device=args.device,
        steering_vector_path=args.steering_vector_path,
        steering_overshoot=args.steering_overshoot,
        target_layer=args.target_layer,
        alignment_threshold=args.alignment_threshold,
        max_refinement_iters=args.max_refinement_iters,
        initial_steering_ratio=args.initial_steering_ratio,
        dija_mask_counts=args.dija_mask_counts,
        inject_prompt=args.inject_prompt,
        attack_method=args.attack_method,
        gen_batch_size=getattr(args, "gen_batch_size", 1),
    )

    if 'MATH' in args.csv_path:
        requests = [
            Instance(request_type="generate_until", doc={}, arguments=(row[question_key]+"\nFollowing the format: Solution: <solution>\nAnswer: <answer>", {"until": []}), idx=i)
            for i, row in enumerate(dataset)
        ]
    elif args.attack_method == "DIJA":
        requests = [
            Instance(
                request_type="generate_until",
                doc={},
                arguments=(row[question_key], {"until": [], "vanilla_goal": row.get("goal", row[question_key])}),
                idx=i,
            )
            for i, row in enumerate(dataset)
        ]
    elif 'MMLLU' in args.csv_path:
        requests = [
            Instance(request_type="generate_until", doc={}, arguments=(row[question_key]+"\nChoose the index of the correct answer from the following choices: " + str(row['choices']) + "\nYou need to answer only the index of the correct answer.", {"until": []}), idx=i)
            for i, row in enumerate(dataset)
        ]
    elif args.self_reminder == "True":
        requests = [
            Instance(request_type="generate_until", doc={}, arguments=(SAFE_REMINDER + "\n" + row[question_key], {"until": []}), idx=i)
            for i, row in enumerate(dataset)
        ]
    elif args.attack_method == 'prefix':
        if getattr(args, "prefix_source", "references0") == "harmbench_human_jailbreaks":
            prefixes = select_harmbench_prefix_templates(
                load_harmbench_prefix_templates(),
                random_subset=getattr(args, "prefix_random_subset", 5),
                seed=getattr(args, "prefix_seed", 1),
            )
            request_rows = [
                {"prompt": f"{prefix}\n\n{row[question_key]}"}
                for row in dataset
                for prefix in prefixes
            ]
            requests = [
                Instance(request_type="generate_until", doc={}, arguments=(item["prompt"], {"until": []}), idx=i)
                for i, item in enumerate(request_rows)
            ]
        else:
            requests = [
                Instance(request_type="generate_until", doc={}, arguments=(REFERENCES[0] + " " + row[question_key], {"until": []}), idx=i)
                for i, row in enumerate(dataset)
            ]
    else:
        requests = [
            Instance(request_type="generate_until", doc={}, arguments=(row[question_key], {"until": []}), idx=i)
            for i, row in enumerate(dataset)
        ]

    results = model.generate_until(requests)

    if request_rows is not None:
        out = [{"prompt": request_rows[i]["prompt"], "response": r} for i, r in enumerate(results)]
    else:
        out = [{"prompt": dataset[i][question_key], "response": r} for i, r in enumerate(results)]
    os.makedirs(args.generated_samples_path, exist_ok=True)
    out_path = os.path.join(args.generated_samples_path, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(out)} results to {out_path}")
    if hasattr(model, "write_steering_diagnostics"):
        model.write_steering_diagnostics()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv_path", type=str, default=None)
    pre_args, remaining = parser.parse_known_args()

    if pre_args.csv_path:
        full_parser = argparse.ArgumentParser()
        full_parser.add_argument("--csv_path", type=str, required=True)
        full_parser.add_argument("--model_path", type=str, required=True)
        full_parser.add_argument("--generated_samples_path", type=str, default="./outputs")
        full_parser.add_argument("--batch_size", type=int, default=32)
        full_parser.add_argument("--gen_batch_size", type=int, default=1,
                                 help="generate_until 的批量大小（GPU 并行样本数）。与 --batch_size 无关"
                                      "（后者仅用于 loglikelihood 的 MC 采样）。1 = 原单样本行为；"
                                      "变长 prompt 会右 pad 到 batch 内最大长度并用 attention_mask 屏蔽。")
        full_parser.add_argument("--sampling_steps", type=int, default=128)
        full_parser.add_argument("--mask_length", type=int, default=128)
        full_parser.add_argument("--block_size", type=int, default=128)
        full_parser.add_argument("--remasking", type=str, default="low_confidence")
        full_parser.add_argument("--sampler", type=str, default="steering")
        full_parser.add_argument("--remdm_number", type=int, default=4)
        full_parser.add_argument("--attack_method", type=str, default="zeroshot", choices=["DIJA", "prefix", "PAP", "zeroshot"])
        full_parser.add_argument("--prefix_source", type=str, default="references0", choices=["references0", "harmbench_human_jailbreaks"])
        full_parser.add_argument("--prefix_random_subset", type=int, default=5)
        full_parser.add_argument("--prefix_seed", type=int, default=1)
        full_parser.add_argument("--cfg", type=float, default=0.)
        full_parser.add_argument("--device", type=str, default="cuda")
        full_parser.add_argument("--self_reminder", type=str, default="False")
        full_parser.add_argument("--steering_vector_path", type=str, default='')
        full_parser.add_argument("--steering_overshoot", type=float, default=1.0,
                                 help="Beta multiplier on excess projection (a_t - threshold). "
                                      "1.0 = clip projection to threshold; >1.0 = push below threshold.")
        full_parser.add_argument("--target_layer", type=int, default=31)
        full_parser.add_argument("--alignment_threshold", type=float, default=0.0,
                                 help="Per-token harmful direction projection threshold. "
                                      "Tokens with alignment above this are remasked and regenerated with adaptive steering.")
        full_parser.add_argument("--max_refinement_iters", type=int, default=5,
                                 help="Maximum number of remask-regenerate iterations per Phase 2 block.")
        full_parser.add_argument("--initial_steering_ratio", type=float, default=0.1,
                                 help="Fraction of Phase 1 steps that apply adaptive steering. "
                                      "Prevents harmful skeleton formation while preserving generalization "
                                      "(benign-leaning mask positions get alpha_t=0 due to the threshold gate).")
        full_parser.add_argument("--dija_mask_counts", type=int, default=128,
                                 help="DIJA only: number of trailing <|mdm_mask|> tokens appended after the chat "
                                      "prompt when the refined_goal contains no embedded <mask:N> pattern.")
        full_parser.add_argument("--inject_prompt", default=True, action="store_true",
                                 help="llada_remdm only: inject a copy of the prompt at the front of the "
                                      "generation region so that mask-position hidden states (and the steering "
                                      "decision based on them) are conditioned on doubled prompt context. Only "
                                      "the trailing mask_length tokens are returned as the generated answer.")
        args = full_parser.parse_args()
        run_csv_eval(args)
    else:
        import sys
        sys.argv = [sys.argv[0]] + remaining
        cli_evaluate()
