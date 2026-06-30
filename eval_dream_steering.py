import accelerate
import torch
import pandas as pd
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset, load_dataset, concatenate_datasets
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from utils import REFERENCES, SAFE_REMINDER
from transformers import AutoTokenizer, AutoModel
import json
import os
import re


DIJA_MASK_PATTERN = r'<mask:(\d+)>'
DIJA_MASK_TOKEN_STR = '<|mask|>'
DIJA_START_TOKEN = '<|im_start|>'
DIJA_END_TOKEN = '<|im_end|>'


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


@register_model("dream_dist")
class DreamEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=151666,
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
        max_refinement_iters=3,
        initial_steering_ratio=0.5,
        dija_mask_counts=36,
        inject_prompt=True,
        attack_method='zeroshot',
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

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
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
        self.steering_overshoot = steering_overshoot
        self.target_layer = target_layer
        self.alignment_threshold = alignment_threshold
        self.max_refinement_iters = max_refinement_iters
        self.initial_steering_ratio = initial_steering_ratio
        self.dija_mask_counts = dija_mask_counts
        self.inject_prompt = inject_prompt
        if inject_prompt:
            print("inject_prompt enabled: a copy of the prompt is placed at the front of the "
                  "generation region; only the trailing mask_length tokens are decoded as output.")
        if steering_vector_path:
            print(f"Loading steering vector from {steering_vector_path}")
            vectors = torch.load(steering_vector_path, weights_only=True)
            key = f'layer_{target_layer}'
            if key not in vectors:
                available = list(vectors.keys())
                raise ValueError(f"Layer {target_layer} not found. Available: {available}")
            self.steering_vector = vectors[key]
            print(f"Steering vector loaded (layer {target_layer}, norm={self.steering_vector.norm().item():.4f})")
            print(f"steering_overshoot(beta)={steering_overshoot}, alignment_threshold={alignment_threshold}, "
                  f"initial_steering_ratio={initial_steering_ratio}, max_refinement_iters={max_refinement_iters}")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _get_transformer_layer(self, layer_idx: int):
        m = self.model
        if hasattr(m, 'model') and hasattr(m.model, 'layers'):
            return m.model.layers[layer_idx]
        if hasattr(m, 'model') and hasattr(m.model, 'transformer') and hasattr(m.model.transformer, 'blocks'):
            return m.model.transformer.blocks[layer_idx]
        raise AttributeError(
            f"Cannot find transformer layer {layer_idx}. "
            "Expected model.model.layers or model.model.transformer.blocks."
        )

    def _make_attn(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(ids, dtype=torch.bool, device=ids.device)

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
        attn = self._make_attn(batch)
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])
            attn = torch.cat([attn, self._make_attn(un_batch)])

        logits = self.model(batch, attention_mask=attn).logits

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
    def dream_conf_sample(self, prompt):
        xt = torch.full((1, prompt.shape[1] + self.mask_length), self.mask_id, dtype=torch.long).to(self.model.device)
        xt[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (xt != self.mask_id)
        prompt_len = prompt_index.sum(1).item()

        assert self.mask_length % self.block_size == 0
        num_blocks = self.mask_length // self.block_size

        assert self.sampling_steps % num_blocks == 0
        steps = self.sampling_steps // num_blocks

        assert self.mask_length % self.sampling_steps == 0

        for num_block in range(num_blocks):
            for i in range(steps):
                mask_index = (xt == self.mask_id)

                attn = self._make_attn(xt)
                logits = self.model(xt, attention_mask=attn).logits

                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = _sample_categorical(p)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                x0_p[:, prompt_len + (num_block + 1) * self.block_size:] = -np.inf
                x0 = torch.where(mask_index, x0, xt)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=int(self.mask_length / self.sampling_steps))
                    transfer_index[j, select_index] = True
                xt[transfer_index] = x0[transfer_index]
            if torch.sum(xt == self.tokenizer.eos_token_id) > 0:
                return xt

        return xt

    @torch.no_grad()
    def _extract_block_hidden(self, xt, block_start, block_end):
        # 提取 block 区域 [block_start:block_end] 的隐藏状态
        # 通过 forward hook 获取 target_layer 的输出，然后按 block 边界切片
        hidden_buffer = [None]
        def _extract_hook(module, input, output, _buf=hidden_buffer):
            h = output[0] if isinstance(output, tuple) else output
            _buf[0] = h.detach()
            return output
        layer = self._get_transformer_layer(self.target_layer)
        handle = layer.register_forward_hook(_extract_hook)
        attn = self._make_attn(xt)
        _ = self.model(xt, attention_mask=attn).logits
        handle.remove()
        return hidden_buffer[0][:, block_start:block_end, :]

    @torch.no_grad()
    def _per_token_alignment(self, block_hidden):
        # 计算 block 内每个 token 的 hidden state 与 steering vector 的对齐程度
        # 返回正值越大，表示该 token 语义越接近有害方向
        sv = self.steering_vector.to(block_hidden.device, dtype=block_hidden.dtype)
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
            sv = _sv.to(hidden.device, dtype=hidden.dtype)
            sv_unit = sv / (sv.norm() + 1e-8)
            hidden = hidden.clone()
            masked_h = hidden[_mask_index]                                  # [n_masked, D]
            a = (masked_h * sv_unit.unsqueeze(0)).sum(dim=-1)               # [n_masked]
            alpha_t = _beta * (a - _theta).clamp(min=0)                     # [n_masked]
            hidden[_mask_index] = masked_h - alpha_t.unsqueeze(-1) * sv_unit.unsqueeze(0)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return steering_hook

    @torch.no_grad()
    def _refill_masks_with_steering(self, xt, block_end):
        # Phase 2 辅助方法：对当前所有 mask 位置做一次 forward pass，
        # 并用 adaptive steering 生成替换 token
        mask_index = (xt == self.mask_id)
        if not mask_index.any():
            return xt

        steering_hook = self._build_adaptive_steering_hook(mask_index)
        layer = self._get_transformer_layer(self.target_layer)
        hook_handle = layer.register_forward_hook(steering_hook)

        attn = self._make_attn(xt)
        logits = self.model(xt, attention_mask=attn).logits
        hook_handle.remove()

        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0 = _sample_categorical(p)

        xt = torch.where(mask_index, x0, xt)
        return xt

    @torch.no_grad()
    def dream_steering_sample(self, prompt):
        # ==============================================================
        # 两阶段安全生成：Adaptive Steering + Harmful Token Remasking
        #
        # Phase 1 (逐 block 解码):
        #   - 将 mask_length 划分为 num_blocks 个 block
        #   - 每个 block 内执行 steps 次 confidence-based 逐步解码
        #   - 每次揭掉 confidence 最高的 k = mask_length / sampling_steps 个 token
        #   - block 边界约束: x0_p[:, block_end:] = -np.inf，确保不跨 block
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
        # ==============================================================
        P = prompt.shape[1]
        if self.inject_prompt:
            total_len = P + P + self.mask_length
            xt = torch.full((1, total_len), self.mask_id, dtype=torch.long).to(self.model.device)
            xt[:, :P] = prompt.clone()
            xt[:, P:2 * P] = prompt.clone()
        else:
            xt = torch.full((1, P + self.mask_length), self.mask_id, dtype=torch.long).to(self.model.device)
            xt[:, :P] = prompt.clone()

        prompt_index = (xt != self.mask_id)
        prompt_len = prompt_index.sum(1).item()

        assert self.mask_length % self.block_size == 0
        num_blocks = self.mask_length // self.block_size

        assert self.sampling_steps % num_blocks == 0
        steps = self.sampling_steps // num_blocks

        assert self.mask_length % self.sampling_steps == 0

        can_steer = (self.steering_vector is not None)
        initial_steering_steps = int(steps * self.initial_steering_ratio) if can_steer else 0

        # ---- Phase 1: 逐 block 解码（confidence-based 逐步揭掩码） ----
        for num_block in range(num_blocks):
            block_start = prompt_len + num_block * self.block_size
            block_end = prompt_len + (num_block + 1) * self.block_size

            for i in range(steps):
                mask_index = (xt == self.mask_id)

                # 前 initial_steering_steps 步可选的 adaptive steering
                hook_handle = None
                if can_steer and i < initial_steering_steps:
                    block_mask_index = mask_index.clone()
                    block_mask_index[:, :block_start] = False
                    block_mask_index[:, block_end:] = False
                    if block_mask_index.any():
                        steering_hook = self._build_adaptive_steering_hook(block_mask_index)
                        layer = self._get_transformer_layer(self.target_layer)
                        hook_handle = layer.register_forward_hook(steering_hook)

                attn = self._make_attn(xt)
                logits = self.model(xt, attention_mask=attn).logits

                if hook_handle is not None:
                    hook_handle.remove()

                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = _sample_categorical(p)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                # block 边界约束：只允许在 block 范围内产生新 token
                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, xt)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                # 只揭掉 confidence 最高的 k 个 token
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=int(self.mask_length / self.sampling_steps))
                    transfer_index[j, select_index] = True
                xt[transfer_index] = x0[transfer_index]

            # ---- Phase 2: block 全部解码完成后，做安全检测 + remask ----
            if can_steer:
                for refinement_iter in range(self.max_refinement_iters):
                    block_hidden = self._extract_block_hidden(xt, block_start, block_end)
                    per_token_alignment = self._per_token_alignment(block_hidden)  # [1, block_size]

                    harmful_mask = per_token_alignment > self.alignment_threshold  # [1, block_size]
                    n_harmful = harmful_mask.sum().item()

                    if n_harmful == 0:
                        break

                    full_harmful_mask = torch.zeros_like(xt, dtype=torch.bool)
                    full_harmful_mask[:, block_start:block_end] = harmful_mask
                    xt[full_harmful_mask] = self.mask_id

                    xt = self._refill_masks_with_steering(xt, block_end)

            if torch.sum(xt == self.tokenizer.eos_token_id) > 0:
                return xt

        return xt

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
    def dream_dija_sample(self, refined_goal, vanilla_goal=None, mask_counts=36):
        refined_text = self._build_dija_prompt_text(refined_goal, mask_counts)
        refined_ids = self.tokenizer(refined_text, return_tensors="pt")["input_ids"].to(self.device)

        if vanilla_goal is not None:
            vanilla_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": vanilla_goal}],
                tokenize=False,
                add_generation_prompt=True,
            )
            vanilla_ids = self.tokenizer(vanilla_text, return_tensors="pt")["input_ids"].to(self.device)
            common_len = min(vanilla_ids.shape[1], refined_ids.shape[1])
            eq = (vanilla_ids[0, :common_len] == refined_ids[0, :common_len])
            first_diff = (~eq).nonzero(as_tuple=False)
            matching_count = int(first_diff[0].item()) if first_diff.numel() > 0 else common_len
        else:
            matching_count = 0

        xt = refined_ids.clone()
        original_mask_index = (xt == self.mask_id)
        if not original_mask_index.any():
            return xt, matching_count

        can_steer = (self.steering_vector is not None)
        steps = self.sampling_steps
        initial_steering_steps = int(steps * self.initial_steering_ratio) if can_steer else 0

        for i in range(steps):
            mask_index = (xt == self.mask_id)
            if not mask_index.any():
                break

            hook_handle = None
            if can_steer and i < initial_steering_steps:
                steering_hook = self._build_adaptive_steering_hook(mask_index)
                layer = self._get_transformer_layer(self.target_layer)
                hook_handle = layer.register_forward_hook(steering_hook)

            attn = self._make_attn(xt)
            logits = self.model(xt, attention_mask=attn).logits

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
            for refinement_iter in range(self.max_refinement_iters):
                hidden_buffer = [None]
                def _extract_hook(module, input, output, _buf=hidden_buffer):
                    h = output[0] if isinstance(output, tuple) else output
                    _buf[0] = h.detach()
                    return output
                layer = self._get_transformer_layer(self.target_layer)
                handle = layer.register_forward_hook(_extract_hook)
                attn = self._make_attn(xt)
                _ = self.model(xt, attention_mask=attn).logits
                handle.remove()

                full_hidden = hidden_buffer[0]
                sv = self.steering_vector.to(full_hidden.device, dtype=full_hidden.dtype)
                sv_unit = sv / (sv.norm() + 1e-8)
                alignment = (full_hidden * sv_unit).sum(dim=-1)

                harmful_mask = (alignment > self.alignment_threshold) & original_mask_index
                if harmful_mask.sum().item() == 0:
                    break

                xt[harmful_mask] = self.mask_id
                xt = self._refill_masks_with_steering(xt, block_end=xt.shape[1])

        return xt, matching_count

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]):
        is_dija = self.attack_method == 'DIJA'

        def _tokenize(e):
            if is_dija:
                return {
                    "question_text": e["question"],
                    "vanilla_goal": e.get("vanilla_goal", e["question"]),
                    "until": e["until"],
                }
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        if is_dija:
            ds = [
                {
                    "question": req.args[0],
                    "until": req.args[1]['until'],
                    "vanilla_goal": req.args[1].get('vanilla_goal', req.args[0]),
                }
                for req in requests
            ]
        else:
            ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        if not is_dija:
            ds = ds.with_format("torch")

        out, out_for_json = [], []
        for elem in tqdm(ds):
            stop_tokens = list(elem["until"]) + ["<|eot_id|>", self.tokenizer.eos_token]

            if is_dija:
                generated_ids, matching_count = self.dream_dija_sample(
                    elem["question_text"],
                    vanilla_goal=elem["vanilla_goal"],
                    mask_counts=self.dija_mask_counts,
                )
                generated_answer = self.tokenizer.decode(
                    generated_ids[0][matching_count:], skip_special_tokens=False
                )
                generated_answer = generated_answer.split("assistant\n")[0]
            else:
                prompt = elem["question"].unsqueeze(0).to(self.device)
                if self.sampler == 'llada':
                    generated_answer = self.dream_conf_sample(prompt)
                elif self.sampler == 'steering':
                    generated_answer = self.dream_steering_sample(prompt)
                generated_answer = self.tokenizer.decode(
                    generated_answer[0][-self.mask_length:], skip_special_tokens=False
                )

            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)
            out_for_json.append({
                "prefix": elem["question_text"],
                "result": generated_answer,
            })

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


def run_csv_eval(args):
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

    sampler = 'dream_dija' if args.attack_method == "DIJA" else args.sampler

    model = DreamEvalHarness(
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
    )

    if 'MATH' in args.csv_path:
        requests = [
            Instance(request_type="generate_until", doc={}, arguments=(row[question_key]+"\nFollowing the format: Solution: <solution>\nAnswer: <answer>", {"until": []}), idx=i)
            for i, row in enumerate(dataset)
        ]
    elif args.self_reminder == "True":
        requests = [
            Instance(request_type="generate_until", doc={}, arguments=(SAFE_REMINDER + "\n" + row[question_key], {"until": []}), idx=i)
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
    elif args.attack_method == 'prefix':
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

    out = [{"prompt": dataset[i][question_key], "response": r} for i, r in enumerate(results)]
    os.makedirs(args.generated_samples_path, exist_ok=True)
    out_path = os.path.join(args.generated_samples_path, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(out)} results to {out_path}")


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
        full_parser.add_argument("--sampling_steps", type=int, default=128)
        full_parser.add_argument("--mask_length", type=int, default=128)
        full_parser.add_argument("--block_size", type=int, default=128)
        full_parser.add_argument("--dija_mask_counts", type=int, default=128)
        full_parser.add_argument("--self_reminder", type=str, default="False")
        full_parser.add_argument("--remasking", type=str, default="low_confidence")
        full_parser.add_argument("--sampler", type=str, default="steering",
                                 choices=["steering", "llada"])
        full_parser.add_argument("--remdm_number", type=int, default=4)
        full_parser.add_argument("--attack_method", type=str, default="zeroshot", choices=["DIJA", "prefix", "PAP", "zeroshot"])
        full_parser.add_argument("--cfg", type=float, default=0.)
        full_parser.add_argument("--device", type=str, default="cuda")
        full_parser.add_argument("--steering_vector_path", type=str, default='')
        full_parser.add_argument("--steering_overshoot", type=float, default=1.0,
                                 help="Beta multiplier on excess projection (a_t - threshold). "
                                      "1.0 = clip projection to threshold; >1.0 = push below threshold.")
        full_parser.add_argument("--target_layer", type=int, default=24)
        full_parser.add_argument("--alignment_threshold", type=float, default=0.0,
                                 help="Per-token harmful direction projection threshold. "
                                      "Tokens with alignment above this are remasked and regenerated with adaptive steering.")
        full_parser.add_argument("--max_refinement_iters", type=int, default=3,
                                 help="Maximum number of remask-regenerate iterations per Phase 2 block.")
        full_parser.add_argument("--initial_steering_ratio", type=float, default=0.1,
                                 help="Fraction of Phase 1 steps that apply adaptive steering. "
                                      "Prevents harmful skeleton formation while preserving generalization "
                                      "(benign-leaning mask positions get alpha_t=0 due to the threshold gate).")
        full_parser.add_argument("--dija_mask_counts", type=int, default=128,
                                 help="DIJA only: number of trailing <|mask|> tokens appended after the chat "
                                      "prompt when the refined_goal contains no embedded <mask:N> pattern.")
        full_parser.add_argument("--inject_prompt", default=True, action="store_true",
                                 help="dream_remdm only: inject a copy of the prompt at the front of the "
                                      "generation region so that mask-position hidden states (and the steering "
                                      "decision based on them) are conditioned on doubled prompt context. Only "
                                      "the trailing mask_length tokens are returned as the generated answer.")
        args = full_parser.parse_args()
        run_csv_eval(args)
    else:
        import sys
        sys.argv = [sys.argv[0]] + remaining
        cli_evaluate()
