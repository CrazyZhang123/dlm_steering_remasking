from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_path(raw_path: str, cwd: Path | None = None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    root = cwd or Path.cwd()
    return root / path


def load_refusals(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_harmful_data(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    flat: list[dict] = []
    for item in data:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


CATEGORY_FALLBACK_KEYS = ("semantic_category", "functional_category", "category")


def resolve_category(sample: dict, category_key: str | None = None) -> str:
    keys: list[str] = []
    if category_key:
        keys.append(category_key)
    keys.extend(key for key in CATEGORY_FALLBACK_KEYS if key not in keys)

    for key in keys:
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "unknown"


def keep_response_token(tokenizer, token_id: int) -> bool:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "mask_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            special_ids.add(int(value))
    if int(token_id) in special_ids:
        return False
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return bool(text.strip())


def filter_response_hidden_states(
    tokenizer,
    response_ids: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    filtered_hidden, _filtered_ids = filter_response_tokens(tokenizer, response_ids, hidden)
    return filtered_hidden


def filter_response_tokens(
    tokenizer,
    response_ids: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = torch.tensor(
        [keep_response_token(tokenizer, int(token_id)) for token_id in response_ids[: hidden.shape[0]]],
        dtype=torch.bool,
    )
    return hidden[keep], response_ids[: hidden.shape[0]][keep]


def load_mil_probe(path: Path, target_layer: int, device: torch.device):
    from utils.train_mil_token_probe_llada import LinearMILProbe

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "mil_token_probe_v1":
        raise ValueError(f"Unsupported MIL probe format: {state.get('format')!r}")
    if state.get("model_family") != "llada":
        raise ValueError(f"Unsupported MIL probe model_family: {state.get('model_family')!r}")
    if int(state.get("target_layer")) != int(target_layer):
        raise ValueError(
            f"MIL probe target_layer={state.get('target_layer')} does not match requested target_layer={target_layer}"
        )
    input_dim = int(state["input_dim"])
    probe = LinearMILProbe(input_dim=input_dim)
    probe.load_state_dict(state["state_dict"])
    return probe.to(device).eval(), state


def score_tokens_with_probe(probe, hidden: torch.Tensor) -> torch.Tensor:
    param = next(probe.parameters())
    with torch.no_grad():
        logits = probe(hidden.to(device=param.device, dtype=param.dtype))
        return torch.sigmoid(logits).detach().to("cpu")


def apply_probe_threshold(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("probe_threshold must be in [0, 1].")
    keep = scores >= float(threshold)
    return hidden[keep], token_ids[keep], scores[keep]


def new_probe_diagnostics() -> dict:
    return {
        "total_harmful_tokens_before_probe": 0,
        "total_harmful_tokens_after_probe": 0,
        "per_response_retention_ratio": [],
        "per_category_tokens_before_probe": {},
        "per_category_tokens_after_probe": {},
        "probe_empty_samples": 0,
        "high_score_tokens": [],
    }


def _add_count(counts: dict[str, int], key: str, value: int) -> None:
    counts[key] = int(counts.get(key, 0)) + int(value)


def record_probe_selection(
    diagnostics: dict,
    tokenizer,
    sample_index: int,
    category: str,
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    selected_mask: torch.Tensor,
    max_examples_per_response: int = 5,
) -> None:
    before = int(token_ids.numel())
    after = int(selected_mask.sum().item())
    diagnostics["total_harmful_tokens_before_probe"] += before
    diagnostics["total_harmful_tokens_after_probe"] += after
    diagnostics["per_response_retention_ratio"].append(float(after / before) if before else 0.0)
    _add_count(diagnostics["per_category_tokens_before_probe"], category, before)
    _add_count(diagnostics["per_category_tokens_after_probe"], category, after)
    if before and after == 0:
        diagnostics["probe_empty_samples"] += 1

    if before == 0:
        return
    top_k = min(max_examples_per_response, before)
    top_indices = torch.topk(scores.reshape(-1), k=top_k).indices.tolist()
    for idx in top_indices:
        token_id = int(token_ids[idx].item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        diagnostics["high_score_tokens"].append(
            {
                "sample_index": int(sample_index),
                "category": str(category),
                "token_text": token_text,
                "token_id": token_id,
                "probe_score": float(scores[idx].item()),
                "selected_by_threshold": bool(selected_mask[idx].item()),
            }
        )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def write_probe_diagnostics(
    output_dir: Path,
    diagnostics: dict,
    probe_path: Path,
    probe_threshold: float,
    top_q_ratio: float,
) -> None:
    before = int(diagnostics["total_harmful_tokens_before_probe"])
    after = int(diagnostics["total_harmful_tokens_after_probe"])
    category_before = {
        str(category): int(count)
        for category, count in diagnostics["per_category_tokens_before_probe"].items()
    }
    category_after = {
        str(category): int(count)
        for category, count in diagnostics["per_category_tokens_after_probe"].items()
    }
    per_category_retention = {
        category: (float(category_after.get(category, 0) / count) if count else 0.0)
        for category, count in category_before.items()
    }
    ratios = [float(value) for value in diagnostics["per_response_retention_ratio"]]
    summary = {
        "probe_path": str(probe_path),
        "probe_threshold": float(probe_threshold),
        "top_q_ratio": float(top_q_ratio),
        "total_harmful_tokens_before_probe": before,
        "total_harmful_tokens_after_probe": after,
        "global_retention_ratio": float(after / before) if before else 0.0,
        "probe_empty_samples": int(diagnostics["probe_empty_samples"]),
        "per_category_tokens_before_probe": category_before,
        "per_category_tokens_after_probe": category_after,
        "per_category_retention_ratio": per_category_retention,
        "per_response_retention_ratio": {
            "min": _percentile(ratios, 0.0),
            "p25": _percentile(ratios, 0.25),
            "median": _percentile(ratios, 0.5),
            "p75": _percentile(ratios, 0.75),
            "max": _percentile(ratios, 1.0),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mil_token_selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    examples = sorted(
        diagnostics["high_score_tokens"],
        key=lambda item: float(item["probe_score"]),
        reverse=True,
    )[:100]
    lines = [
        "# MIL High Score Tokens",
        "",
        "| category | token_text | probe_score | selected_by_threshold |",
        "|---|---|---:|---|",
    ]
    for item in examples:
        selected = "yes" if item["selected_by_threshold"] else "no"
        token_text = str(item["token_text"]).replace("|", "\\|").replace("\n", "\\n")
        lines.append(f"| {item['category']} | {token_text} | {item['probe_score']:.6f} | {selected} |")
    lines.append("")
    (output_dir / "mil_high_score_tokens.md").write_text("\n".join(lines), encoding="utf-8")


def new_cluster_token_terms() -> dict[int, Counter]:
    return {}


def record_cluster_token_terms(
    cluster_terms: dict[int, Counter],
    tokenizer,
    labels: torch.Tensor,
    token_ids: torch.Tensor,
    global_offset: int = 0,
) -> None:
    for label, token_id in zip(labels.tolist(), token_ids.tolist()):
        cluster_id = int(global_offset) + int(label)
        if cluster_id not in cluster_terms:
            cluster_terms[cluster_id] = Counter()
        token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        cluster_terms[cluster_id][token_text] += 1


def write_cluster_token_terms(output_dir: Path, state: dict, cluster_terms: dict[int, Counter]) -> None:
    cluster_sizes = [int(x) for x in state["cluster_sizes"].tolist()]
    center_categories = list(state.get("center_categories", ["global"] * len(cluster_sizes)))
    cluster_ids_tensor = state.get("cluster_ids", torch.arange(len(cluster_sizes), dtype=torch.long))
    cluster_ids = [int(x) for x in cluster_ids_tensor.tolist()]
    lines = [
        "# Cluster Token Top Terms",
        "",
        "| global_cluster_id | category | local_cluster_id | cluster_size | top_terms |",
        "|---:|---|---:|---:|---|",
    ]
    for global_cluster_id, size in enumerate(cluster_sizes):
        terms = cluster_terms.get(global_cluster_id, Counter())
        rendered_terms = []
        for term, count in terms.most_common(20):
            safe_term = term.replace("|", "\\|").replace("\n", "\\n")
            rendered_terms.append(f"{safe_term}:{count}")
        top_terms = ", ".join(rendered_terms)
        lines.append(
            f"| {global_cluster_id} | {center_categories[global_cluster_id]} | "
            f"{cluster_ids[global_cluster_id]} | {size} | {top_terms} |"
        )
    lines.append("")
    (output_dir / "cluster_token_top_terms.md").write_text("\n".join(lines), encoding="utf-8")


def build_sequence(tokenizer, prompt: str, response: str, max_response_len: int):
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )[0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(response_ids) > max_response_len:
        response_ids = response_ids[:max_response_len]
    full = torch.cat([prompt_ids, response_ids], dim=0)
    return full, int(prompt_ids.shape[0]), response_ids


def count_response_tokens_by_category(tokenizer, harmful: list[dict], args) -> dict[str, int]:
    # 按类别统计每个 harmful response 中的有效 token 数，
    # 用于后续按比例分配聚类数（token 多的类别分更多 cluster）。
    counts: dict[str, int] = {}
    for sample in harmful:
        response = str(sample.get("response", "")).strip()
        if not response:
            continue
        response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if len(response_ids) > args.max_response_len:
            response_ids = response_ids[: args.max_response_len]
        kept = sum(1 for token_id in response_ids if keep_response_token(tokenizer, int(token_id)))
        if kept <= 0:
            continue
        category = resolve_category(sample, getattr(args, "category_key", None))
        counts[category] = counts.get(category, 0) + int(kept)
    return counts


def make_category_cluster_plan(
    category_token_counts: dict[str, int],
    num_total_clusters: int,
) -> dict:
    """
    给定各类别的 token 数和 cluster 总数，按比例分配每个类别应分到几个 cluster。
    如果类别太多就合并尾巴到 "other"，最后用最大余数法微调使总数精确匹配预算。
    """
    if num_total_clusters <= 0:
        raise ValueError("num_total_clusters must be positive")

    # 只保留 token 数 > 0 的类别
    positive = {
        str(category): int(count)
        for category, count in category_token_counts.items()
        if int(count) > 0
    }
    if not positive:
        raise RuntimeError("No category has usable harmful response tokens.")

    # 按 token 数降序排列，token 数相同时按类别名升序
    # 原理是 Python 元组排序的逐项比较规则：(-count, category_name)
    # 1. 先比 -count：-100 < -50，所以 100 的排在 50 前面 → 实现了 token 数降序。
    # 2. 如果两个 -count 相等（即 token 数相同），再比 category_name："fraud" < "violence" 字符串字母序 → 同 token 数时按名字升序。
    ranked = sorted(positive.items(), key=lambda item: (-item[1], item[0]))
    raw_to_center_category: dict[str, str] = {}

    # 如果类别数超过了总 cluster 数，需要将尾部类别折叠到 "other"
    if len(ranked) > num_total_clusters:
        if num_total_clusters == 1:
            kept: list[tuple[str, int]] = []
            tail = ranked
        else:
            kept = ranked[: num_total_clusters - 1]
            tail = ranked[num_total_clusters - 1 :]
        collapsed: dict[str, int] = {category: count for category, count in kept}
        collapsed["other"] = collapsed.get("other", 0) + sum(count for _category, count in tail)
        for category, _count in kept:
            raw_to_center_category[category] = category
        for category, _count in tail:
            raw_to_center_category[category] = "other"
    else:
        collapsed = dict(ranked)
        raw_to_center_category = {category: category for category in collapsed}

    categories = sorted(collapsed)
    if num_total_clusters < len(categories):
        raise RuntimeError(
            f"Collapsed category count {len(categories)} exceeds cluster budget {num_total_clusters}."
        )

    # 按 token 比例分配 cluster，先取 floor，至少 1
    total = float(sum(collapsed.values()))
    weighted = []
    cluster_counts = {}
    for category in categories:
        exact = num_total_clusters * (collapsed[category] / total)
        base = max(1, int(exact))
        cluster_counts[category] = base
        weighted.append((exact - int(exact), collapsed[category], category))

    # 如果总分配数超出预算，从小数部分最小（且 count 最小、名字靠前）的类别扣减
    while sum(cluster_counts.values()) > num_total_clusters:
        candidates = [
            (fraction, count, category)
            for fraction, count, category in weighted
            if cluster_counts[category] > 1
        ]
        if not candidates:
            break
        _fraction, _count, category = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0]
        cluster_counts[category] -= 1

    # 如果还有剩余配额，补给小数部分最大的类别
    leftover = num_total_clusters - sum(cluster_counts.values())
    if leftover:
        for _fraction, _count, category in sorted(weighted, key=lambda item: (-item[0], -item[1], item[2]))[:leftover]:
            cluster_counts[category] += 1

    return {
        "categories": categories,
        "category_token_counts": {category: int(collapsed[category]) for category in categories},
        "category_cluster_counts": {category: int(cluster_counts[category]) for category in categories},
        "raw_to_center_category": raw_to_center_category,
    }


def category_cluster_offsets(category_plan: dict) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for category in category_plan["categories"]:
        offsets[category] = cursor
        cursor += int(category_plan["category_cluster_counts"][category])
    return offsets


@torch.no_grad()
def extract_target_layer_tokens(
    model,
    input_ids: torch.Tensor,
    response_start: int,
    target_layer: int,
    device: torch.device,
) -> torch.Tensor:
    hidden_buffer = [None]

    def hook(_module, _input, output, _buf=hidden_buffer):
        h = output[0] if isinstance(output, tuple) else output
        _buf[0] = h.detach()
        return output

    handle = model.model.transformer.blocks[target_layer].register_forward_hook(hook)
    try:
        _ = model(input_ids.to(device))
    finally:
        handle.remove()
    return hidden_buffer[0][0, response_start:, :].float().cpu()


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1e-8)


def choose_aux_refusal(refusals: list[str], args, sample_index: int, pass_salt: int) -> str:
    if not refusals:
        raise RuntimeError("No refusal paraphrases available.")
    seed = int(getattr(args, "seed", 0)) + int(pass_salt) * 1_000_003 + int(sample_index)
    rng = random.Random(seed)
    return refusals[rng.randrange(len(refusals))]


TOKEN_SELECTION_CHOICES = (
    "all",
    "direction_top_ratio",
    "random_top_ratio",
    "mil_probe_threshold",
    "knn_label_clean",
)
# pass_salt 已占用：coarse direction=41、route preprocess=53；KNN 安全池用未占用的 47。
KNN_PASS_SALT = 47
FEATURE_PREPROCESS_CHOICES = ("l2_only", "center_l2", "center_pca128_l2", "center_pca256_l2")


def selected_token_count(n_tokens: int, ratio: float, max_selected_tokens: int | None) -> int:
    n_tokens = int(n_tokens)
    if n_tokens <= 0:
        return 0
    if ratio <= 0:
        raise ValueError("selection_ratio must be positive.")
    requested = int(math.ceil(float(ratio) * n_tokens))
    if max_selected_tokens is not None and int(max_selected_tokens) > 0:
        requested = min(requested, int(max_selected_tokens))
    return max(1, min(n_tokens, requested))


def top_ratio_select(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    scores: torch.Tensor,
    ratio: float,
    max_selected_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = selected_token_count(int(hidden.shape[0]), ratio, max_selected_tokens)
    mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
    if count == 0:
        return hidden[:0], token_ids[:0], scores[:0], mask
    flat_scores = scores.reshape(-1).to(device=hidden.device)
    top_indices = torch.topk(flat_scores, k=count).indices
    mask[top_indices] = True
    return hidden[top_indices], token_ids[top_indices], scores.reshape(-1)[top_indices.cpu()], mask.cpu()


def _coarse_direction_for_sample(sample: dict, args) -> torch.Tensor | None:
    direction_type = getattr(args, "coarse_direction_type", "category")
    global_direction = getattr(args, "global_coarse_direction", None)
    if direction_type == "global":
        return global_direction

    category = resolve_category(sample, getattr(args, "category_key", None))
    directions = getattr(args, "coarse_directions_by_category", {}) or {}
    counts = getattr(args, "coarse_direction_token_counts", {}) or {}
    min_tokens = int(getattr(args, "min_coarse_tokens", 0))
    direction = directions.get(category)
    if direction is not None and int(counts.get(category, 0)) >= min_tokens:
        return direction
    return global_direction


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


def _knn_nearest_indices(
    features: torch.Tensor,
    n_query: int,
    k: int,
    backend: str = "auto",
    metric: str = "cosine",
) -> torch.Tensor:
    """返回前 n_query 个向量各自最近的 k 个邻居的全局 index（含自身，调用方负责排除）。

    cosine 度量先做 L2 归一化，使「最大余弦相似」等价于「最小欧氏距离」；随后 faiss 与
    sklearn 统一按「最近欧氏」检索，保证两后端口径一致（不使用 sklearn 的 cosine metric，
    避免与 faiss 内积口径产生数值差异）。
    """
    x = features.float()
    if metric == "cosine":
        x = F.normalize(x, p=2, dim=1)
    elif metric != "euclidean":
        raise ValueError(f"Unsupported knn_metric: {metric!r}")
    x = x.contiguous()
    queries = x[:n_query].contiguous()

    resolved = backend
    if backend == "auto":
        resolved = "faiss" if _faiss_available() else "sklearn"
    if resolved == "faiss":
        import faiss

        index = faiss.IndexFlatL2(int(x.shape[1]))
        index.add(x.numpy())
        _dist, idx = index.search(queries.numpy(), int(k))
        return torch.from_numpy(idx).long()
    if resolved == "sklearn":
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=int(k), algorithm="brute", metric="euclidean")
        nn.fit(x.numpy())
        idx = nn.kneighbors(queries.numpy(), return_distance=False)
        return torch.from_numpy(idx).long()
    raise ValueError(f"Unsupported knn_backend: {backend!r}")


def knn_keep_decisions(
    features: torch.Tensor,
    n_harmful: int,
    k: int,
    keep_ratio: float,
    backend: str = "auto",
    metric: str = "cosine",
) -> torch.Tensor:
    """ENN 标签去噪投票：对前 n_harmful 个有害 token，返回 keep BoolTensor[n_harmful]。

    `features` 前 n_harmful 行为有害（标签 1），其余为安全（标签 0）。对每个有害 token 取最近
    k 个邻居（按 index 排除自身），若有害邻居占比 >= keep_ratio 则保留。极小数据集下若无可用
    邻居则保守保留。
    """
    n_total = int(features.shape[0])
    labels = torch.zeros(n_total, dtype=torch.int8)
    labels[:n_harmful] = 1
    k_query = min(int(k) + 1, n_total)
    neighbor_idx = _knn_nearest_indices(features, n_harmful, k_query, backend, metric)
    keep = torch.zeros(n_harmful, dtype=torch.bool)
    for i in range(n_harmful):
        nb = [j for j in neighbor_idx[i].tolist() if j != i][: int(k)]
        if not nb:
            keep[i] = True
            continue
        ratio = float((labels[nb] == 1).float().mean())
        keep[i] = ratio >= float(keep_ratio)
    return keep


def _knn_top_terms(tokenizer, token_ids: list[int], top_n: int) -> list[list]:
    counter: Counter = Counter()
    for tok in token_ids:
        text = tokenizer.decode([int(tok)], skip_special_tokens=False).strip()
        if text:
            counter[text] += 1
    return [[term, int(count)] for term, count in counter.most_common(top_n)]


def build_knn_keep_masks(
    model,
    tokenizer,
    harmful: list[dict],
    refusals: list[str],
    args,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """pass 0：全局 KNN/ENN 标签去噪，为每个有害样本预计算「保留 token」的 BoolTensor。

    有害池与安全池分两个独立循环收集：
    - 有害池只 forward 有害侧（不抽 refusal、只检查有害侧长度），保证 token 序列与 pass1/2
      逐位对齐；
    - 安全池独立收集，refusal 经 choose_aux_refusal(pass_salt=KNN_PASS_SALT) 抽取，不消耗全局
      random 状态。
    """
    # ---- (1) 有害池：仅 forward 有害侧，与 refusal 解耦 ----
    harmful_vectors: list[torch.Tensor] = []
    harmful_meta: list[tuple[int, int, int]] = []  # (sample_index, position, token_id)
    for idx, sample in enumerate(harmful):
        prompt = sample.get("prompt", "").strip()
        harmful_resp = sample.get("response", "").strip()
        if not prompt or not harmful_resp:
            continue
        ids_h, rs_h, response_ids_h = build_sequence(tokenizer, prompt, harmful_resp, args.max_response_len)
        if len(ids_h) > args.max_total_len:
            continue
        try:
            h_tokens = extract_target_layer_tokens(model, ids_h.unsqueeze(0), rs_h, args.target_layer, device)
        except RuntimeError as exc:
            print(f"[{idx}] knn pass0 harmful forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        h_tokens, h_token_ids = filter_response_tokens(tokenizer, response_ids_h, h_tokens)
        for pos in range(int(h_tokens.shape[0])):
            harmful_vectors.append(h_tokens[pos])
            harmful_meta.append((idx, pos, int(h_token_ids[pos])))

    if not harmful_vectors:
        raise RuntimeError("KNN pass0: no harmful tokens collected.")

    # ---- (2) 安全池：独立循环，choose_aux_refusal 保持 RNG 隔离 ----
    safe_vectors: list[torch.Tensor] = []
    for idx, sample in enumerate(harmful):
        prompt = sample.get("prompt", "").strip()
        if not prompt:
            continue
        refusal = choose_aux_refusal(refusals, args, idx, pass_salt=KNN_PASS_SALT)
        ids_s, rs_s, response_ids_s = build_sequence(tokenizer, prompt, refusal, args.max_response_len)
        if len(ids_s) > args.max_total_len:
            continue
        try:
            s_tokens = extract_target_layer_tokens(model, ids_s.unsqueeze(0), rs_s, args.target_layer, device)
        except RuntimeError as exc:
            print(f"[{idx}] knn pass0 safe forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        s_tokens = filter_response_hidden_states(tokenizer, response_ids_s, s_tokens)
        cap = int(getattr(args, "knn_safe_pool_cap", 0))
        if cap > 0 and int(s_tokens.shape[0]) > cap:
            gen = torch.Generator().manual_seed(int(args.seed) + idx)
            sel = torch.randperm(int(s_tokens.shape[0]), generator=gen)[:cap]
            s_tokens = s_tokens[sel]
        for pos in range(int(s_tokens.shape[0])):
            safe_vectors.append(s_tokens[pos])

    # ---- (3) 建索引 + 查近邻 + ENN 投票 ----
    if not safe_vectors:
        warnings.warn(
            "KNN pass0: 安全池为空（安全侧全部超长或被过滤？），标签去噪退化为全部保留",
            RuntimeWarning,
            stacklevel=2,
        )
    n_h = len(harmful_vectors)
    X = torch.stack(harmful_vectors + safe_vectors, dim=0).float()
    keep = knn_keep_decisions(
        X,
        n_h,
        int(args.knn_k),
        float(args.knn_keep_ratio),
        args.knn_backend,
        args.knn_metric,
    )

    # ---- (4) 回写为按 position 有序的 BoolTensor + 诊断统计 ----
    pairs: dict[int, list[tuple[int, bool]]] = {}
    kept_ids: list[int] = []
    removed_ids: list[int] = []
    for i, (idx, pos, tok) in enumerate(harmful_meta):
        pairs.setdefault(idx, []).append((pos, bool(keep[i])))
        (kept_ids if bool(keep[i]) else removed_ids).append(tok)
    masks = {
        idx: torch.tensor([value for _pos, value in sorted(positions)], dtype=torch.bool)
        for idx, positions in pairs.items()
    }

    kept = int(keep.sum().item())
    args._knn_stats = {
        "total_harmful_tokens": n_h,
        "kept_harmful_tokens": kept,
        "removed_harmful_tokens": n_h - kept,
        "retention": (kept / n_h) if n_h else 0.0,
        "safe_pool_tokens": len(safe_vectors),
        "degenerate": not safe_vectors,
        "knn_k": int(args.knn_k),
        "knn_keep_ratio": float(args.knn_keep_ratio),
        "knn_metric": str(args.knn_metric),
        "knn_backend": str(args.knn_backend),
        "knn_safe_pool_cap": int(getattr(args, "knn_safe_pool_cap", 0)),
        "removed_top_terms": _knn_top_terms(tokenizer, removed_ids, 20),
        "kept_top_terms": _knn_top_terms(tokenizer, kept_ids, 20),
    }
    return masks


def write_knn_label_clean_summary(output_dir: Path, stats: dict) -> None:
    with (output_dir / "knn_label_clean_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)


def select_harmful_response_tokens(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    sample: dict,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = getattr(args, "token_selection", "all")
    if mode in {None, "all", "mil_probe_threshold"} or hidden.numel() == 0:
        return hidden, token_ids

    ratio = float(getattr(args, "selection_ratio", 1.0))
    max_selected = getattr(args, "max_selected_tokens", None)
    if mode == "direction_top_ratio":
        direction = _coarse_direction_for_sample(sample, args)
        if direction is None:
            return hidden, token_ids
        direction = direction.to(device=hidden.device, dtype=hidden.dtype)
        if direction.numel() == 0 or float(direction.norm().item()) <= 0.0:
            return hidden, token_ids
        scores = hidden @ unit(direction).reshape(-1)
        selected_hidden, selected_ids, _scores, _mask = top_ratio_select(hidden, token_ids, scores, ratio, max_selected)
        return selected_hidden, selected_ids

    if mode == "random_top_ratio":
        count = selected_token_count(int(hidden.shape[0]), ratio, max_selected)
        if count == 0:
            return hidden[:0], token_ids[:0]
        generator = torch.Generator(device="cpu")
        seed = int(getattr(args, "seed", 0)) + int(getattr(args, "_sample_index", 0))
        generator.manual_seed(seed)
        indices = torch.randperm(int(hidden.shape[0]), generator=generator)[:count]
        indices = indices.sort().values.to(device=hidden.device)
        return hidden[indices], token_ids[indices.cpu()]

    if mode == "knn_label_clean":
        # 全局 KNN/ENN 标签去噪：保留结论在 pass 0 由 build_knn_keep_masks 预计算。
        masks = getattr(args, "_knn_keep_masks", None)
        if masks is None:
            # 预处理阶段（fit_route_preprocess）缓存未就绪 → 等效 "all"，不在此处过滤。
            return hidden, token_ids
        mask = masks.get(int(getattr(args, "_sample_index", -1)))
        if mask is None:
            # 该样本在 pass 0 被跳过（如长度超限），保守保留并告警以便诊断。
            warnings.warn(
                f"[knn_label_clean] sample {getattr(args, '_sample_index', -1)} 无 keep mask，保守保留全部 token",
                RuntimeWarning,
                stacklevel=2,
            )
            return hidden, token_ids
        if mask.shape[0] != hidden.shape[0]:
            raise RuntimeError(
                f"KNN keep mask 与 token 序列错位: mask={mask.shape[0]} vs hidden={hidden.shape[0]} "
                f"(sample {getattr(args, '_sample_index', -1)})"
            )
        keep = mask.to(device=hidden.device)
        return hidden[keep], token_ids[mask.cpu()]

    raise ValueError(f"Unsupported token_selection: {mode!r}")


def pca_dim_for_mode(mode: str, fallback: int = 128) -> int | None:
    if mode == "center_pca128_l2":
        return 128
    if mode == "center_pca256_l2":
        return 256
    if mode.startswith("center_pca") and mode.endswith("_l2"):
        raw = mode.removeprefix("center_pca").removesuffix("_l2")
        return int(raw) if raw else int(fallback)
    return None


def normalize_route_preprocess(preprocess: dict | None = None) -> dict:
    if preprocess is None:
        return {"mode": "l2_only"}
    mode = str(preprocess.get("mode", "l2_only"))
    state = {"mode": mode}
    if "mean" in preprocess and preprocess["mean"] is not None:
        state["mean"] = preprocess["mean"].detach().float().cpu()
    if "pca_components" in preprocess and preprocess["pca_components"] is not None:
        state["pca_components"] = preprocess["pca_components"].detach().float().cpu()
    if "pca_dim" in preprocess and preprocess["pca_dim"] is not None:
        state["pca_dim"] = int(preprocess["pca_dim"])
    if "requested_pca_dim" in preprocess and preprocess["requested_pca_dim"] is not None:
        state["requested_pca_dim"] = int(preprocess["requested_pca_dim"])
    return state


def transform_route_features(hidden: torch.Tensor, preprocess: dict | None = None) -> torch.Tensor:
    state = normalize_route_preprocess(preprocess)
    mode = state["mode"]
    features = hidden.float()
    if mode == "l2_only":
        return unit(features)
    if mode not in FEATURE_PREPROCESS_CHOICES:
        raise ValueError(f"Unsupported feature_preprocess: {mode!r}")
    mean = state.get("mean")
    if mean is None:
        raise ValueError(f"feature_preprocess={mode!r} requires a mean tensor")
    centered = features - mean.to(device=features.device, dtype=features.dtype)
    if mode == "center_l2":
        return unit(centered)
    components = state.get("pca_components")
    if components is None:
        raise ValueError(f"feature_preprocess={mode!r} requires pca_components")
    projected = centered @ components.to(device=features.device, dtype=features.dtype).T
    return unit(projected)


def extract_sample_response_tokens(
    model,
    tokenizer,
    sample: dict,
    refusal: str,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    prompt = sample.get("prompt", "").strip()
    harmful_resp = sample.get("response", "").strip()
    if not prompt or not harmful_resp:
        return None

    ids_h, rs_h, response_ids_h = build_sequence(tokenizer, prompt, harmful_resp, args.max_response_len)
    ids_s, rs_s, response_ids_s = build_sequence(tokenizer, prompt, refusal, args.max_response_len)
    if len(ids_h) > args.max_total_len or len(ids_s) > args.max_total_len:
        return None

    h_tokens = extract_target_layer_tokens(model, ids_h.unsqueeze(0), rs_h, args.target_layer, device)
    s_tokens = extract_target_layer_tokens(model, ids_s.unsqueeze(0), rs_s, args.target_layer, device)
    h_tokens, h_token_ids = filter_response_tokens(tokenizer, response_ids_h, h_tokens)
    s_tokens = filter_response_hidden_states(tokenizer, response_ids_s, s_tokens)
    return h_tokens, h_token_ids, s_tokens


def _apply_mil_selection_no_diagnostics(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    mil_probe = getattr(args, "mil_probe", None)
    if mil_probe is None or hidden.numel() == 0:
        return hidden, token_ids
    scores = score_tokens_with_probe(mil_probe, hidden)
    selected_hidden, selected_ids, _scores = apply_probe_threshold(
        hidden,
        token_ids,
        scores,
        float(args.probe_threshold),
    )
    return selected_hidden, selected_ids


def fit_coarse_directions(
    model,
    tokenizer,
    harmful: list[dict],
    refusals: list[str],
    args,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int], torch.Tensor, int]:
    category_sums: dict[str, torch.Tensor] = {}
    category_counts: dict[str, int] = {}
    global_harm_sum = None
    global_harm_count = 0
    safe_sum = None
    safe_sample_count = 0
    skipped = 0

    for idx, sample in enumerate(tqdm(harmful, desc="Stage 4A coarse directions")):
        refusal = choose_aux_refusal(refusals, args, idx, pass_salt=41)
        args._sample_index = idx
        try:
            item = extract_sample_response_tokens(model, tokenizer, sample, refusal, args, device)
        except RuntimeError as exc:
            print(f"[{idx}] coarse direction forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as exc:
            print(f"[{idx}] coarse direction sample failed: {exc}")
            skipped += 1
            continue
        if item is None:
            skipped += 1
            continue
        h_tokens, _h_token_ids, s_tokens = item
        if h_tokens.numel() == 0 or s_tokens.numel() == 0:
            skipped += 1
            continue

        category = resolve_category(sample, getattr(args, "category_key", None))
        h_sum = h_tokens.sum(dim=0)
        category_sums[category] = h_sum if category not in category_sums else category_sums[category] + h_sum
        category_counts[category] = int(category_counts.get(category, 0)) + int(h_tokens.shape[0])
        global_harm_sum = h_sum if global_harm_sum is None else global_harm_sum + h_sum
        global_harm_count += int(h_tokens.shape[0])

        s_mean = s_tokens.mean(dim=0)
        safe_sum = s_mean if safe_sum is None else safe_sum + s_mean
        safe_sample_count += 1

    if global_harm_count == 0 or safe_sample_count == 0:
        raise RuntimeError("No response tokens available for coarse direction fitting.")

    safe_mean = safe_sum / float(safe_sample_count)
    directions = {
        category: (category_sums[category] / float(category_counts[category])) - safe_mean
        for category in category_sums
    }
    global_direction = (global_harm_sum / float(global_harm_count)) - safe_mean
    return directions, category_counts, global_direction, skipped


def fit_route_preprocess(
    model,
    tokenizer,
    harmful: list[dict],
    refusals: list[str],
    args,
    device: torch.device,
) -> tuple[dict, int]:
    mode = getattr(args, "feature_preprocess", "l2_only")
    if mode == "l2_only":
        return {"mode": "l2_only"}, 0
    if mode not in FEATURE_PREPROCESS_CHOICES:
        raise ValueError(f"Unsupported feature_preprocess: {mode!r}")

    requested_pca_dim = pca_dim_for_mode(mode, int(getattr(args, "pca_dim", 128)))
    token_sum = None
    token_count = 0
    gram = None
    skipped = 0

    def observe(tokens: torch.Tensor) -> None:
        nonlocal token_sum, token_count, gram
        if tokens.numel() == 0:
            return
        data = tokens.float().cpu()
        token_sum = data.sum(dim=0) if token_sum is None else token_sum + data.sum(dim=0)
        token_count += int(data.shape[0])
        if requested_pca_dim is not None:
            partial_gram = data.T @ data
            gram = partial_gram if gram is None else gram + partial_gram

    for idx, sample in enumerate(tqdm(harmful, desc="Stage 5 fit route preprocess")):
        refusal = choose_aux_refusal(refusals, args, idx, pass_salt=53)
        args._sample_index = idx
        try:
            item = extract_sample_response_tokens(model, tokenizer, sample, refusal, args, device)
        except RuntimeError as exc:
            print(f"[{idx}] route preprocess forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as exc:
            print(f"[{idx}] route preprocess sample failed: {exc}")
            skipped += 1
            continue
        if item is None:
            skipped += 1
            continue
        h_tokens, h_token_ids, s_tokens = item
        h_tokens, h_token_ids = _apply_mil_selection_no_diagnostics(h_tokens, h_token_ids, args)
        if getattr(args, "mil_probe", None) is None:
            h_tokens, h_token_ids = select_harmful_response_tokens(h_tokens, h_token_ids, sample, args)
        observe(h_tokens)
        observe(s_tokens)
        if h_tokens.numel() == 0 and s_tokens.numel() == 0:
            skipped += 1

    if token_count <= 0 or token_sum is None:
        raise RuntimeError("No response tokens available for route preprocessing.")

    mean = token_sum / float(token_count)
    preprocess = {"mode": mode, "mean": mean.float().cpu()}
    if requested_pca_dim is not None:
        actual_dim = min(int(requested_pca_dim), int(token_count), int(mean.shape[0]))
        if actual_dim <= 0:
            raise RuntimeError("PCA preprocessing requires at least one response token and one feature dimension.")
        covariance = gram - float(token_count) * torch.outer(mean, mean)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        top_indices = torch.argsort(eigenvalues, descending=True)[:actual_dim]
        components = eigenvectors[:, top_indices].T.contiguous()
        max_abs_indices = components.abs().argmax(dim=1)
        signs = components[torch.arange(components.shape[0]), max_abs_indices].sign()
        signs[signs == 0] = 1
        components = components * signs.unsqueeze(1)
        preprocess["pca_components"] = components.float().cpu()
        preprocess["pca_dim"] = int(actual_dim)
        preprocess["requested_pca_dim"] = int(requested_pca_dim)
    return preprocess, skipped


def route_centers_from_kmeans(kmeans) -> torch.Tensor | None:
    centers = getattr(kmeans, "cluster_centers_", None)
    if centers is None:
        return None
    return unit(torch.as_tensor(centers, dtype=torch.float32))


def route_centers_from_category_kmeans(kmeans_by_category: dict, category_plan: dict) -> torch.Tensor | None:
    centers = []
    for category in category_plan["categories"]:
        category_centers = route_centers_from_kmeans(kmeans_by_category[category])
        if category_centers is None:
            return None
        centers.append(category_centers)
    return torch.cat(centers, dim=0) if centers else None


def build_bank_state_from_cluster_sums(
    safe_mean: torch.Tensor,
    cluster_sums: torch.Tensor,
    cluster_counts: torch.Tensor,
    target_layer: int,
    max_response_len: int,
    num_total_clusters: int,
    method: str = "ct_csd",
    category_key: str | None = None,
    center_categories: list[str] | None = None,
    center_cluster_ids: list[int] | None = None,
    category_plan: dict | None = None,
    route_preprocess: dict | None = None,
    route_centers: torch.Tensor | None = None,
) -> dict:
    if torch.any(cluster_counts <= 0):
        raise RuntimeError(f"Empty CT-CSD cluster detected: counts={cluster_counts.tolist()}")
    centers = cluster_sums / cluster_counts.unsqueeze(-1).float()
    vectors = centers - safe_mean.unsqueeze(0)
    preprocess = normalize_route_preprocess(route_preprocess)
    if route_centers is None:
        route_centers_tensor = transform_route_features(centers, preprocess)
    else:
        route_centers_tensor = unit(route_centers.float())
    if route_centers_tensor.shape[0] != int(num_total_clusters):
        raise ValueError("route_centers row count must match num_total_clusters")
    cluster_feature = (
        "l2_normalized_hidden"
        if preprocess["mode"] == "l2_only"
        else "route_preprocessed_hidden"
    )
    state = {
        "format": "ct_csd_v1",
        "model_family": "llada",
        "target_layer": int(target_layer),
        "safe_anchor_type": "sample_balanced_global_safe_mean",
        "safe_mean": safe_mean.float().cpu(),
        "raw_centers": centers.float().cpu(),
        "centers": centers.float().cpu(),
        "centers_unit": unit(centers).float().cpu(),
        "route_centers": route_centers_tensor.float().cpu(),
        "vectors": vectors.float().cpu(),
        "vectors_unit": unit(vectors).float().cpu(),
        "cluster_ids": torch.arange(num_total_clusters, dtype=torch.long),
        "global_cluster_ids": torch.arange(num_total_clusters, dtype=torch.long),
        "cluster_sizes": cluster_counts.long().cpu(),
        "preprocess": preprocess,
        "config": {
            "method": method,
            "num_total_clusters": int(num_total_clusters),
            "cluster_feature": cluster_feature,
            "feature_preprocess": preprocess["mode"],
            "pca_dim": preprocess.get("pca_dim"),
            "requested_pca_dim": preprocess.get("requested_pca_dim"),
            "category_key": category_key,
            "max_response_len": int(max_response_len),
            "exclude_special_tokens": True,
            "exclude_blank_tokens": True,
            "exclude_punctuation": False,
        },
        "mil": {
            "enabled": False,
            "probe_path": None,
            "probe_threshold": None,
            "top_q_ratio": None,
        },
    }
    if center_categories is not None:
        if len(center_categories) != num_total_clusters:
            raise ValueError("center_categories length must match num_total_clusters")
        categories = list(category_plan["categories"]) if category_plan is not None else sorted(set(center_categories))
        category_to_id = {category: idx for idx, category in enumerate(categories)}
        state["categories"] = categories
        state["center_categories"] = list(center_categories)
        state["center_category_ids"] = torch.tensor(
            [category_to_id[category] for category in center_categories],
            dtype=torch.long,
        )
        if center_cluster_ids is None:
            raise ValueError("center_cluster_ids is required when center_categories is provided")
        if len(center_cluster_ids) != num_total_clusters:
            raise ValueError("center_cluster_ids length must match num_total_clusters")
        state["cluster_ids"] = torch.tensor(center_cluster_ids, dtype=torch.long)
        if category_plan is not None:
            state["config"]["category_token_counts"] = dict(category_plan["category_token_counts"])
            state["config"]["category_cluster_counts"] = dict(category_plan["category_cluster_counts"])
            state["config"]["raw_to_center_category"] = dict(category_plan["raw_to_center_category"])
    return state


def iter_valid_sample_tokens(
    model,
    tokenizer,
    sample: dict,
    refusal: str,
    args,
    device: torch.device,
):
    item = extract_sample_response_tokens(model, tokenizer, sample, refusal, args, device)
    if item is None:
        return None

    h_tokens, h_token_ids, s_tokens = item
    mil_probe = getattr(args, "mil_probe", None)
    if mil_probe is not None and h_tokens.numel() > 0:
        scores = score_tokens_with_probe(mil_probe, h_tokens)
        selected_mask = scores >= float(args.probe_threshold)
        category = resolve_category(sample, getattr(args, "category_key", None))
        diagnostics = getattr(args, "probe_diagnostics", None)
        record_probe_diagnostics = bool(getattr(args, "record_probe_diagnostics", True))
        if diagnostics is not None and record_probe_diagnostics:
            record_probe_selection(
                diagnostics,
                tokenizer,
                sample_index=int(getattr(args, "_sample_index", -1)),
                category=category,
                token_ids=h_token_ids,
                scores=scores,
                selected_mask=selected_mask,
            )
        h_tokens, h_token_ids, _kept_scores = apply_probe_threshold(
            h_tokens,
            h_token_ids,
            scores,
            float(args.probe_threshold),
        )
        if h_tokens.numel() == 0 and record_probe_diagnostics:
            args.probe_empty_samples = int(getattr(args, "probe_empty_samples", 0)) + 1
    elif h_tokens.numel() > 0:
        h_tokens, h_token_ids = select_harmful_response_tokens(h_tokens, h_token_ids, sample, args)
    args._last_harmful_token_ids = h_token_ids
    if s_tokens.numel() == 0:
        return None
    return h_tokens, s_tokens.mean(dim=0)


def fit_minibatch_kmeans(model, tokenizer, harmful: list[dict], refusals: list[str], args, device: torch.device):
    kmeans = MiniBatchKMeans(
        n_clusters=args.num_total_clusters,
        batch_size=args.kmeans_batch_size,
        random_state=args.seed,
        n_init="auto",
    )
    safe_sum = None
    safe_count = 0
    fitted_batches = 0
    skipped = 0
    first_fit_buffer: list[torch.Tensor] = []

    for idx, sample in enumerate(tqdm(harmful, desc="Stage 1 pass 1: fit clusters")):
        refusal = random.choice(refusals)
        args._sample_index = idx
        args.record_probe_diagnostics = False
        try:
            item = iter_valid_sample_tokens(model, tokenizer, sample, refusal, args, device)
        except RuntimeError as exc:
            print(f"[{idx}] pass 1 forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as exc:
            print(f"[{idx}] pass 1 sample failed: {exc}")
            skipped += 1
            continue
        if item is None:
            skipped += 1
            continue

        h_tokens, safe_mean_i = item
        safe_sum = safe_mean_i if safe_sum is None else safe_sum + safe_mean_i
        safe_count += 1
        if h_tokens.numel() == 0:
            skipped += 1
            continue

        features = transform_route_features(h_tokens, getattr(args, "route_preprocess", None)).cpu()
        if fitted_batches == 0:
            first_fit_buffer.append(features)
            buffered = torch.cat(first_fit_buffer, dim=0)
            if buffered.shape[0] < args.num_total_clusters:
                continue
            kmeans.partial_fit(buffered.numpy())
            first_fit_buffer.clear()
        else:
            kmeans.partial_fit(features.numpy())
        fitted_batches += 1

    if fitted_batches == 0 or safe_count == 0:
        raise RuntimeError("No samples processed successfully during CT-CSD pass 1.")
    return kmeans, safe_sum / safe_count, skipped


def fit_category_minibatch_kmeans(
    model,
    tokenizer,
    harmful: list[dict],
    refusals: list[str],
    args,
    device: torch.device,
    category_plan: dict,
    kmeans_factory=None,
):
    if kmeans_factory is None:
        def kmeans_factory(_category: str, n_clusters: int):
            return MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=args.kmeans_batch_size,
                random_state=args.seed,
                n_init="auto",
            )

    kmeans_by_category = {
        category: kmeans_factory(category, int(category_plan["category_cluster_counts"][category]))
        for category in category_plan["categories"]
    }
    buffers: dict[str, list[torch.Tensor]] = {category: [] for category in category_plan["categories"]}
    fitted_batches: dict[str, int] = {category: 0 for category in category_plan["categories"]}
    safe_sum = None
    safe_count = 0
    skipped = 0

    for idx, sample in enumerate(tqdm(harmful, desc="Stage 3 pass 1: fit category clusters")):
        raw_category = resolve_category(sample, getattr(args, "category_key", None))
        category = category_plan["raw_to_center_category"].get(raw_category, raw_category)
        if category not in kmeans_by_category:
            skipped += 1
            continue
        refusal = random.choice(refusals)
        args._sample_index = idx
        args.record_probe_diagnostics = False
        try:
            item = iter_valid_sample_tokens(model, tokenizer, sample, refusal, args, device)
        except RuntimeError as exc:
            print(f"[{idx}] category pass 1 forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as exc:
            print(f"[{idx}] category pass 1 sample failed: {exc}")
            skipped += 1
            continue
        if item is None:
            skipped += 1
            continue

        h_tokens, safe_mean_i = item
        safe_sum = safe_mean_i if safe_sum is None else safe_sum + safe_mean_i
        safe_count += 1
        if h_tokens.numel() == 0:
            skipped += 1
            continue

        features = transform_route_features(h_tokens, getattr(args, "route_preprocess", None)).cpu()
        required = int(category_plan["category_cluster_counts"][category])
        if fitted_batches[category] == 0:
            buffers[category].append(features)
            buffered = torch.cat(buffers[category], dim=0)
            if buffered.shape[0] < required:
                continue
            kmeans_by_category[category].partial_fit(buffered.numpy())
            buffers[category].clear()
        else:
            kmeans_by_category[category].partial_fit(features.numpy())
        fitted_batches[category] += 1

    missing = [category for category, count in fitted_batches.items() if count == 0]
    if missing or safe_count == 0:
        raise RuntimeError(
            f"No samples processed successfully for category clusters: missing={missing}, safe_count={safe_count}"
        )
    return kmeans_by_category, safe_sum / safe_count, skipped


def accumulate_cluster_sums(
    model,
    tokenizer,
    harmful: list[dict],
    refusals: list[str],
    kmeans,
    args,
    device: torch.device,
):
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    if d_model is None:
        raise RuntimeError("Unable to infer hidden dimension from model config.")
    cluster_sums = torch.zeros(args.num_total_clusters, int(d_model), dtype=torch.float32)
    cluster_counts = torch.zeros(args.num_total_clusters, dtype=torch.long)
    skipped = 0

    for idx, sample in enumerate(tqdm(harmful, desc="Stage 1 pass 2: accumulate clusters")):
        refusal = random.choice(refusals)
        args._sample_index = idx
        args.record_probe_diagnostics = True
        try:
            item = iter_valid_sample_tokens(model, tokenizer, sample, refusal, args, device)
        except RuntimeError as exc:
            print(f"[{idx}] pass 2 forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as exc:
            print(f"[{idx}] pass 2 sample failed: {exc}")
            skipped += 1
            continue
        if item is None:
            skipped += 1
            continue

        h_tokens, _safe_mean_i = item
        if h_tokens.numel() == 0:
            skipped += 1
            continue
        route_features = transform_route_features(h_tokens, getattr(args, "route_preprocess", None))
        labels = torch.tensor(kmeans.predict(route_features.cpu().numpy()), dtype=torch.long)
        token_ids = getattr(args, "_last_harmful_token_ids", None)
        if token_ids is not None and getattr(args, "cluster_token_terms", None) is not None:
            record_cluster_token_terms(args.cluster_token_terms, tokenizer, labels, token_ids)
        for cluster_id in range(args.num_total_clusters):
            mask = labels == cluster_id
            if mask.any():
                cluster_sums[cluster_id] += h_tokens[mask].sum(dim=0)
                cluster_counts[cluster_id] += int(mask.sum().item())

    return cluster_sums, cluster_counts, skipped


def accumulate_category_cluster_sums(
    model,
    tokenizer,
    harmful: list[dict],
    refusals: list[str],
    kmeans_by_category: dict,
    args,
    device: torch.device,
    category_plan: dict,
):
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    if d_model is None:
        raise RuntimeError("Unable to infer hidden dimension from model config.")

    offsets = category_cluster_offsets(category_plan)
    total_clusters = sum(int(category_plan["category_cluster_counts"][category]) for category in category_plan["categories"])
    cluster_sums = torch.zeros(total_clusters, int(d_model), dtype=torch.float32)
    cluster_counts = torch.zeros(total_clusters, dtype=torch.long)
    center_categories: list[str] = []
    center_cluster_ids: list[int] = []
    cluster_category_counts: dict[str, list[int]] = {}
    for category in category_plan["categories"]:
        n_clusters = int(category_plan["category_cluster_counts"][category])
        center_categories.extend([category] * n_clusters)
        center_cluster_ids.extend(range(n_clusters))
        cluster_category_counts[category] = [0] * n_clusters

    skipped = 0
    for idx, sample in enumerate(tqdm(harmful, desc="Stage 3 pass 2: accumulate category clusters")):
        raw_category = resolve_category(sample, getattr(args, "category_key", None))
        category = category_plan["raw_to_center_category"].get(raw_category, raw_category)
        if category not in kmeans_by_category:
            skipped += 1
            continue
        refusal = random.choice(refusals)
        args._sample_index = idx
        args.record_probe_diagnostics = True
        try:
            item = iter_valid_sample_tokens(model, tokenizer, sample, refusal, args, device)
        except RuntimeError as exc:
            print(f"[{idx}] category pass 2 forward failed: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as exc:
            print(f"[{idx}] category pass 2 sample failed: {exc}")
            skipped += 1
            continue
        if item is None:
            skipped += 1
            continue

        h_tokens, _safe_mean_i = item
        if h_tokens.numel() == 0:
            skipped += 1
            continue
        route_features = transform_route_features(h_tokens, getattr(args, "route_preprocess", None))
        labels = torch.tensor(kmeans_by_category[category].predict(route_features.cpu().numpy()), dtype=torch.long)
        offset = int(offsets[category])
        token_ids = getattr(args, "_last_harmful_token_ids", None)
        if token_ids is not None and getattr(args, "cluster_token_terms", None) is not None:
            record_cluster_token_terms(args.cluster_token_terms, tokenizer, labels, token_ids, global_offset=offset)
        n_clusters = int(category_plan["category_cluster_counts"][category])
        for local_cluster_id in range(n_clusters):
            mask = labels == local_cluster_id
            if mask.any():
                global_cluster_id = offset + local_cluster_id
                count = int(mask.sum().item())
                cluster_sums[global_cluster_id] += h_tokens[mask].sum(dim=0)
                cluster_counts[global_cluster_id] += count
                cluster_category_counts[category][local_cluster_id] += count

    return cluster_sums, cluster_counts, skipped, center_categories, center_cluster_ids, cluster_category_counts


def write_bank_summary(output_dir: Path, state: dict) -> None:
    cluster_sizes = [int(x) for x in state["cluster_sizes"].tolist()]
    center_categories = list(state.get("center_categories", ["global"] * len(cluster_sizes)))
    cluster_ids_tensor = state.get("cluster_ids", torch.arange(len(cluster_sizes), dtype=torch.long))
    cluster_ids = [int(x) for x in cluster_ids_tensor.tolist()]
    config = dict(state.get("config", {}))
    rows = []
    for idx, size in enumerate(cluster_sizes):
        rows.append(
            {
                "global_cluster_id": idx,
                "category": center_categories[idx],
                "local_cluster_id": cluster_ids[idx],
                "cluster_size": size,
            }
        )

    summary = {
        "method": config.get("method"),
        "num_total_clusters": int(config.get("num_total_clusters", len(cluster_sizes))),
        "harmful_json": config.get("harmful_json"),
        "refusals_txt": config.get("refusals_txt"),
        "model_path": config.get("model_path"),
        "max_samples": config.get("max_samples"),
        "kmeans_batch_size": config.get("kmeans_batch_size"),
        "seed": config.get("seed"),
        "skipped_pass1": config.get("skipped_pass1"),
        "skipped_pass2": config.get("skipped_pass2"),
        "skipped_coarse_direction": config.get("skipped_coarse_direction"),
        "skipped_preprocess": config.get("skipped_preprocess"),
        "token_selection": config.get("token_selection"),
        "selection_ratio": config.get("selection_ratio"),
        "max_selected_tokens": config.get("max_selected_tokens"),
        "coarse_direction_type": config.get("coarse_direction_type"),
        "min_coarse_tokens": config.get("min_coarse_tokens"),
        "feature_preprocess": config.get("feature_preprocess"),
        "pca_dim": config.get("pca_dim"),
        "requested_pca_dim": config.get("requested_pca_dim"),
        "cluster_sizes": cluster_sizes,
        "category_token_counts": config.get("category_token_counts", {}),
        "category_cluster_counts": config.get("category_cluster_counts", {}),
        "cluster_category_counts": config.get("cluster_category_counts", {}),
        "mil": dict(state.get("mil", {})),
        "clusters": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ct_csd_bank_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    lines = [
        "# CT-CSD Cluster Category Distribution",
        "",
        f"method: {summary['method']}",
        f"num_total_clusters: {summary['num_total_clusters']}",
        "",
        "| category | local_cluster_id | cluster_size |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['category']} | {row['local_cluster_id']} | {row['cluster_size']} |")
    lines.append("")
    (output_dir / "cluster_category_distribution.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/dev/shm/LLaDA-8B-Instruct")
    parser.add_argument("--harmful_json", default=".worktrees/harmbench-csd-export/data/harmbench_testcase_harmful.json")
    parser.add_argument("--refusals_txt", default="utils/refusals.txt")
    parser.add_argument("--output_dir", default="outputs/ct_csd_llada_m16")
    parser.add_argument("--target_layer", type=int, default=31)
    parser.add_argument("--max_response_len", type=int, default=128)
    parser.add_argument("--max_total_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=["ct_csd", "category_ct_csd", "probe_ct_csd", "probe_category_ct_csd"],
        default="ct_csd",
    )
    parser.add_argument("--category_key", default="semantic_category")
    parser.add_argument("--mil_probe_path", default=None)
    parser.add_argument("--probe_threshold", type=float, default=0.7)
    parser.add_argument("--token_selection", choices=TOKEN_SELECTION_CHOICES, default="all")
    parser.add_argument("--selection_ratio", type=float, default=0.3)
    parser.add_argument("--max_selected_tokens", type=int, default=32)
    parser.add_argument("--knn_k", type=int, default=6)
    parser.add_argument("--knn_keep_ratio", type=float, default=0.5)
    parser.add_argument("--knn_metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--knn_backend", choices=["auto", "faiss", "sklearn"], default="auto")
    parser.add_argument(
        "--knn_safe_pool_cap",
        type=int,
        default=0,
        help="0 = 不限（用全部安全 token）；正整数 = 安全池下采样上限",
    )
    parser.add_argument("--coarse_direction_type", choices=["category", "global"], default="category")
    parser.add_argument("--min_coarse_tokens", type=int, default=1024)
    parser.add_argument("--feature_preprocess", choices=FEATURE_PREPROCESS_CHOICES, default="l2_only")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--num_total_clusters", type=int, default=16)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    output_dir = resolve_path(args.output_dir, Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    harmful_path = resolve_path(args.harmful_json, Path.cwd())
    refusals_path = resolve_path(args.refusals_txt, Path.cwd())

    harmful = load_harmful_data(harmful_path)
    if args.max_samples is not None:
        harmful = harmful[: args.max_samples]
    refusals = load_refusals(refusals_path)
    print(f"Loaded {len(harmful)} harmful samples, {len(refusals)} refusal paraphrases")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    is_probe_method = args.method in {"probe_ct_csd", "probe_category_ct_csd"}
    probe_state = None
    probe_path = None
    if is_probe_method:
        if args.mil_probe_path is None:
            raise ValueError("--mil_probe_path is required for probe methods.")
        args.token_selection = "mil_probe_threshold"
        probe_path = resolve_path(args.mil_probe_path, Path.cwd())
        args.mil_probe, probe_state = load_mil_probe(probe_path, args.target_layer, device)
        args.probe_diagnostics = new_probe_diagnostics()
        args.probe_empty_samples = 0
    elif args.token_selection == "mil_probe_threshold":
        raise ValueError("token_selection='mil_probe_threshold' requires a probe method.")
    args.cluster_token_terms = new_cluster_token_terms()
    args.route_preprocess = {"mode": "l2_only"}

    skipped_coarse_direction = 0
    skipped_preprocess = 0
    if args.token_selection == "direction_top_ratio":
        (
            args.coarse_directions_by_category,
            args.coarse_direction_token_counts,
            args.global_coarse_direction,
            skipped_coarse_direction,
        ) = fit_coarse_directions(model, tokenizer, harmful, refusals, args, device)
    args.route_preprocess, skipped_preprocess = fit_route_preprocess(
        model,
        tokenizer,
        harmful,
        refusals,
        args,
        device,
    )

    # pass 0：全局 KNN/ENN 标签去噪，须在预处理之后、pass1/pass2 之前预计算保留 mask。
    if args.token_selection == "knn_label_clean":
        args._knn_keep_masks = build_knn_keep_masks(model, tokenizer, harmful, refusals, args, device)
        write_knn_label_clean_summary(output_dir, args._knn_stats)
        print(
            f"KNN label clean retention: {args._knn_stats['kept_harmful_tokens']}/"
            f"{args._knn_stats['total_harmful_tokens']} "
            f"({args._knn_stats['retention']:.4f})"
        )

    if args.method in {"category_ct_csd", "probe_category_ct_csd"}:
        category_token_counts = count_response_tokens_by_category(tokenizer, harmful, args)
        category_plan = make_category_cluster_plan(category_token_counts, args.num_total_clusters)
        print(f"Category token counts: {category_plan['category_token_counts']}")
        print(f"Category cluster counts: {category_plan['category_cluster_counts']}")
        kmeans_by_category, safe_mean, skipped_pass1 = fit_category_minibatch_kmeans(
            model,
            tokenizer,
            harmful,
            refusals,
            args,
            device,
            category_plan,
        )
        (
            cluster_sums,
            cluster_counts,
            skipped_pass2,
            center_categories,
            center_cluster_ids,
            cluster_category_counts,
        ) = accumulate_category_cluster_sums(
            model,
            tokenizer,
            harmful,
            refusals,
            kmeans_by_category,
            args,
            device,
            category_plan,
        )
        state = build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=args.target_layer,
            max_response_len=args.max_response_len,
            num_total_clusters=args.num_total_clusters,
            method=args.method,
            category_key=args.category_key,
            center_categories=center_categories,
            center_cluster_ids=center_cluster_ids,
            category_plan=category_plan,
            route_preprocess=args.route_preprocess,
            route_centers=route_centers_from_category_kmeans(kmeans_by_category, category_plan),
        )
        state["config"]["cluster_category_counts"] = cluster_category_counts
    else:
        kmeans, safe_mean, skipped_pass1 = fit_minibatch_kmeans(model, tokenizer, harmful, refusals, args, device)
        cluster_sums, cluster_counts, skipped_pass2 = accumulate_cluster_sums(
            model, tokenizer, harmful, refusals, kmeans, args, device
        )
        state = build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=args.target_layer,
            max_response_len=args.max_response_len,
            num_total_clusters=args.num_total_clusters,
            method=args.method,
            category_key=None,
            route_preprocess=args.route_preprocess,
            route_centers=route_centers_from_kmeans(kmeans),
        )
    state["config"]["harmful_json"] = str(harmful_path)
    state["config"]["refusals_txt"] = str(refusals_path)
    state["config"]["model_path"] = str(args.model_path)
    state["config"]["max_samples"] = args.max_samples
    state["config"]["kmeans_batch_size"] = int(args.kmeans_batch_size)
    state["config"]["skipped_pass1"] = int(skipped_pass1)
    state["config"]["skipped_pass2"] = int(skipped_pass2)
    state["config"]["skipped_coarse_direction"] = int(skipped_coarse_direction)
    state["config"]["skipped_preprocess"] = int(skipped_preprocess)
    state["config"]["seed"] = int(args.seed)
    state["config"]["token_selection"] = str(args.token_selection)
    state["config"]["selection_ratio"] = float(args.selection_ratio)
    state["config"]["max_selected_tokens"] = int(args.max_selected_tokens)
    state["config"]["coarse_direction_type"] = str(args.coarse_direction_type)
    state["config"]["min_coarse_tokens"] = int(args.min_coarse_tokens)
    state["config"]["feature_preprocess"] = str(args.route_preprocess.get("mode", args.feature_preprocess))
    state["config"]["pca_dim"] = args.route_preprocess.get("pca_dim")
    state["config"]["requested_pca_dim"] = args.route_preprocess.get("requested_pca_dim")
    if args.token_selection == "direction_top_ratio":
        state["config"]["coarse_direction_token_counts"] = {
            str(category): int(count)
            for category, count in getattr(args, "coarse_direction_token_counts", {}).items()
        }
    if args.token_selection == "knn_label_clean":
        knn_stats = getattr(args, "_knn_stats", {})
        state["config"]["knn_k"] = int(args.knn_k)
        state["config"]["knn_keep_ratio"] = float(args.knn_keep_ratio)
        state["config"]["knn_metric"] = str(args.knn_metric)
        state["config"]["knn_backend"] = str(args.knn_backend)
        state["config"]["knn_safe_pool_cap"] = int(args.knn_safe_pool_cap)
        state["config"]["knn_retention_ratio"] = float(knn_stats.get("retention", 0.0))
    if is_probe_method:
        state["config"]["token_selection"] = "mil_probe_threshold"
        state["config"]["probe_empty_samples"] = int(getattr(args, "probe_empty_samples", 0))
        state["mil"]["enabled"] = True
        state["mil"]["probe_path"] = str(probe_path)
        state["mil"]["probe_threshold"] = float(args.probe_threshold)
        state["mil"]["top_q_ratio"] = float(probe_state.get("top_q_ratio"))

    out_path = output_dir / "ct_csd_bank.pt"
    torch.save(state, out_path)
    write_bank_summary(output_dir, state)
    write_cluster_token_terms(output_dir, state, args.cluster_token_terms)
    if is_probe_method:
        write_probe_diagnostics(
            output_dir,
            args.probe_diagnostics,
            probe_path=probe_path,
            probe_threshold=args.probe_threshold,
            top_q_ratio=float(probe_state.get("top_q_ratio")),
        )
    print(f"Saved CT-CSD bank -> {out_path}")
    print(f"cluster_sizes={state['cluster_sizes'].tolist()}")


if __name__ == "__main__":
    main()
