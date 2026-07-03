import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIJA_TEMPLATES_PATH = ROOT / "DIJA/benchmarks/HarmBench/baselines/pap/templates.py"
FALLBACK_TEMPLATES_PATH = ROOT / "utils/pap_templates_vendored.py"
TEMPLATES_PATH = (
    DEFAULT_DIJA_TEMPLATES_PATH
    if DEFAULT_DIJA_TEMPLATES_PATH.is_file()
    else FALLBACK_TEMPLATES_PATH
)
_TEMPLATES_MODULE_CACHE = None
_MODEL_CACHE = {}
AutoTokenizer = None
AutoModelForCausalLM = None
torch = None


def _normalize_rows(rows, field_name, source):
    normalized = []
    seen = set()

    for row in rows:
        goal = str((row.get(field_name) or "")).strip()
        if not goal or goal in seen:
            continue
        seen.add(goal)
        normalized.append({"goal": goal, "source": source})

    return normalized


def normalize_jbb_rows(rows):
    return _normalize_rows(rows, "Goal", "JBB")


def normalize_advbench_rows(rows):
    return _normalize_rows(rows, "Behavior", "AdvBench")


def _load_templates_module():
    global _TEMPLATES_MODULE_CACHE
    if _TEMPLATES_MODULE_CACHE is not None:
        return _TEMPLATES_MODULE_CACHE

    if not TEMPLATES_PATH.is_file():
        raise FileNotFoundError(f"PAP template file not found: {TEMPLATES_PATH}")

    spec = importlib.util.spec_from_file_location("pap_templates", TEMPLATES_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load PAP templates from {TEMPLATES_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TEMPLATES_MODULE_CACHE = module
    return module


def get_expert_endorsement_template():
    module = _load_templates_module()
    taxonomy = getattr(module, "persuasion_taxonomy", None)
    if taxonomy is None:
        raise RuntimeError(f"PAP templates module missing persuasion_taxonomy: {TEMPLATES_PATH}")

    for template in taxonomy:
        if template.get("ss_technique") == "Expert Endorsement":
            required_fields = ("ss_technique", "ss_definition", "ss_example")
            missing_fields = [field for field in required_fields if not template.get(field)]
            if missing_fields:
                missing_fields_text = ", ".join(missing_fields)
                raise RuntimeError(
                    "Expert Endorsement template missing required fields: "
                    f"{missing_fields_text}"
                )
            return {field: template[field] for field in required_fields}
    raise RuntimeError("Expert Endorsement template not found in PAP persuasion_taxonomy")


def load_one_shot_template():
    module = _load_templates_module()
    one_shot_kd = getattr(module, "one_shot_kd", None)
    if not one_shot_kd:
        raise RuntimeError(f"PAP templates module missing one_shot_kd: {TEMPLATES_PATH}")
    return one_shot_kd


def build_mutation_prompt(goal, template):
    return load_one_shot_template().format(
        technique=template["ss_technique"],
        definition=template["ss_definition"],
        example=template["ss_example"],
        behavior=goal,
    )


def _strip_matching_quotes(text):
    pairs = ('"""', "'''", '"', "'")
    for marker in pairs:
        if text.startswith(marker) and text.endswith(marker):
            return text[len(marker) : -len(marker)].strip()
    return text


def _extract_fenced_block(text):
    match = re.search(r"```[^\n`]*\n?(.*?)```", text, flags=re.DOTALL)
    if not match:
        return text
    return match.group(1).strip()


def _extract_quoted_substring(text):
    patterns = [
        r'"""(.*?)"""',
        r"'''(.*?)'''",
        r'"([^"\n]+(?:\n[^"\n]+)*)"',
        r"'([^'\n]+(?:\n[^'\n]+)*)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def extract_pap_prompt(raw_text):
    cleaned = str(raw_text).strip()
    if not cleaned:
        return cleaned

    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = _extract_fenced_block(cleaned)
        cleaned = _strip_matching_quotes(cleaned)
        cleaned = _extract_quoted_substring(cleaned)
        cleaned = cleaned.strip()

    return cleaned


def write_pap_json(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mutate_goals(rows, model_path, max_new_tokens):
    template = get_expert_endorsement_template()
    payload = []

    for row in rows:
        prompt = build_mutation_prompt(row["goal"], template)
        generated = generate_with_local_model(
            model_path=model_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        payload.append(
            {
                "pap_prompt": extract_pap_prompt(generated),
                "goal": row["goal"],
                "source": row["source"],
            }
        )

    return payload


def parse_non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _ensure_model_dependencies():
    global AutoTokenizer, AutoModelForCausalLM, torch
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        try:
            from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
            from transformers import AutoTokenizer as _AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for local PAP generation"
            ) from exc

        AutoTokenizer = _AutoTokenizer
        AutoModelForCausalLM = _AutoModelForCausalLM
    if torch is None:
        try:
            import torch as _torch
        except ImportError as exc:
            raise RuntimeError("torch is required for local PAP generation") from exc

        torch = _torch


def _get_local_model(model_path):
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    _ensure_model_dependencies()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    cached = (tokenizer, model)
    _MODEL_CACHE[model_path] = cached
    return cached


def _sequence_length(token_ids):
    shape = getattr(token_ids, "shape", None)
    if shape is not None:
        return shape[-1]
    if token_ids and isinstance(token_ids[0], list):
        return len(token_ids[0])
    return len(token_ids)


def _resolve_input_device(model):
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if device not in (None, "disk"):
                return device

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device

    try:
        first_parameter = next(model.parameters())
    except (AttributeError, StopIteration, RuntimeError, TypeError):
        return "cpu"
    return getattr(first_parameter, "device", "cpu")


def generate_with_local_model(model_path, prompt, max_new_tokens):
    tokenizer, model = _get_local_model(model_path)
    _ensure_model_dependencies()
    model_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = _resolve_input_device(model)
    model_inputs = model_inputs.to(device)
    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    input_length = _sequence_length(model_inputs["input_ids"])
    generated_tokens = generated[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["JBB", "AdvBench"], required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--limit", type=parse_non_negative_int, default=0)
    parser.add_argument("--max_new_tokens", type=parse_non_negative_int, default=128)
    args = parser.parse_args()

    rows = load_csv_rows(args.input_path)
    normalized = (
        normalize_jbb_rows(rows)
        if args.dataset_name == "JBB"
        else normalize_advbench_rows(rows)
    )
    if args.limit:
        normalized = normalized[: args.limit]

    payload = mutate_goals(
        normalized,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
    )
    write_pap_json(args.output_path, payload)
    print(f"saved {len(payload)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
