import argparse
import glob
import json
import os
import re
from sympy import sympify
from sympy.parsing.latex import parse_latex
from datasets import load_dataset


def load_math500():
    return list(load_dataset("HuggingFaceH4/MATH-500", split="test"))


def extract_pred(text):
    if text is None:
        return None
    text = str(text)
    if not text.strip():
        return None

    m = list(re.finditer(r"[Aa]nswer\s*:\s*", text))
    if m:
        ans = text[m[-1].end():].strip()
        ans = ans.split("\n")[0].strip()
        if ans:
            return ans

    boxed = last_boxed(text)
    if boxed is not None:
        return boxed

    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def strip_boxed(s):
    inner = last_boxed(s)
    return inner if inner is not None else s


_FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*{([^{}]+)}\s*{([^{}]+)}")
_SQRT_RE = re.compile(r"\\sqrt\s*{([^{}]+)}")


def normalize(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = strip_boxed(s)

    s = s.replace("\\!", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")

    s = re.sub(r"\\text\s*{([^{}]*)}", r"\1", s)
    s = re.sub(r"\\mathrm\s*{([^{}]*)}", r"\1", s)
    s = re.sub(r"\\mbox\s*{([^{}]*)}", r"\1", s)

    s = s.replace("√", "\\sqrt")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt\s*(\d+)", r"\\sqrt{\1}", s)
    s = re.sub(r"\s*/\s*", "/", s)

    if len(s) >= 2 and s[0] == "(" and s[-1] == ")" and "," not in s:
        s = s[1:-1]

    s = re.sub(r"\s+", "", s)

    if s.endswith("."):
        s = s[:-1]

    return s


def to_sympy(s):
    if not s:
        return None
    try:
        return parse_latex(s)
    except Exception:
        pass
    try:
        py = s
        py = _FRAC_RE.sub(r"((\1)/(\2))", py)
        py = _SQRT_RE.sub(r"sqrt(\1)", py)
        py = py.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
        py = py.replace("^", "**")
        return sympify(py)
    except Exception:
        return None


def is_equiv(pred, gold):
    if pred is None:
        return False
    np_, ng = normalize(pred), normalize(gold)
    if np_ == ng:
        return True
    if not np_ or not ng:
        return False

    sp, sg = to_sympy(np_), to_sympy(ng)
    if sp is None or sg is None:
        return False
    try:
        from sympy import simplify
        return simplify(sp - sg) == 0
    except Exception:
        return False


def get_generation(item):
    for key in ("result", "response", "output", "generation"):
        if key in item:
            v = item[key]
            if isinstance(v, list):
                v = v[0] if v else ""
            return v
    return ""


def evaluate(result_path, dataset):
    with open(result_path) as f:
        results = json.load(f)

    if len(results) != len(dataset):
        print(
            f"[warn] {result_path}: result count ({len(results)}) "
            f"!= dataset size ({len(dataset)}); evaluating min length."
        )
    n = min(len(results), len(dataset))

    total = 0
    correct = 0
    invalid = 0
    by_subject = {}
    by_level = {}
    wrong_examples = []

    for i in range(n):
        gold = dataset[i]["answer"]
        subject = dataset[i].get("subject", "unknown")
        level = dataset[i].get("level", "unknown")
        pred = extract_pred(get_generation(results[i]))

        total += 1
        s_stat = by_subject.setdefault(subject, {"total": 0, "correct": 0, "invalid": 0})
        l_stat = by_level.setdefault(level, {"total": 0, "correct": 0, "invalid": 0})
        s_stat["total"] += 1
        l_stat["total"] += 1

        if pred is None:
            invalid += 1
            s_stat["invalid"] += 1
            l_stat["invalid"] += 1
            continue

        if is_equiv(pred, gold):
            correct += 1
            s_stat["correct"] += 1
            l_stat["correct"] += 1
        else:
            if len(wrong_examples) < 10:
                wrong_examples.append({"i": i, "pred": pred, "gold": gold})

    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "by_subject": by_subject,
        "by_level": by_level,
        "wrong_examples": wrong_examples,
    }


def print_report(result_path, stats, show_examples=False):
    t = stats["total"]
    c = stats["correct"]
    inv = stats["invalid"]
    acc = 100.0 * c / t if t else 0.0
    print(f"\n=== {result_path} ===")
    print(f"correct={c}  total={t}  invalid={inv}  acc={acc:.2f}%")

    if stats["by_subject"]:
        print("\nBy subject:")
        header = f"  {'subject':<25} {'correct':>8} {'total':>8} {'invalid':>8} {'acc(%)':>8}"
        print(header)
        for sub in sorted(stats["by_subject"]):
            s = stats["by_subject"][sub]
            sa = 100.0 * s["correct"] / s["total"] if s["total"] else 0.0
            print(f"  {sub:<25} {s['correct']:>8} {s['total']:>8} {s['invalid']:>8} {sa:>8.2f}")

    if stats["by_level"]:
        print("\nBy level:")
        for lvl in sorted(stats["by_level"], key=lambda x: (isinstance(x, str), x)):
            s = stats["by_level"][lvl]
            sa = 100.0 * s["correct"] / s["total"] if s["total"] else 0.0
            print(f"  level {lvl}: {s['correct']}/{s['total']}  invalid={s['invalid']}  acc={sa:.2f}%")

    if show_examples and stats["wrong_examples"]:
        print("\nWrong examples (first 10):")
        for ex in stats["wrong_examples"]:
            print(f"  [{ex['i']}] pred={ex['pred']!r}  gold={ex['gold']!r}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MATH-500 generations.")
    parser.add_argument("paths", nargs="*", help="Result JSON paths")
    parser.add_argument("--glob", help="Glob pattern for result JSONs")
    parser.add_argument("--output", help="Save aggregated report to this JSON file")
    parser.add_argument("--examples", action="store_true", help="Print sample wrong predictions")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob)))
    if not paths:
        parser.error("Provide at least one result path or --glob.")

    dataset = load_math500()

    report = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"[skip] not a file: {path}")
            continue
        stats = evaluate(path, dataset)
        print_report(path, stats, show_examples=args.examples)
        report.append({
            "path": path,
            "total": stats["total"],
            "correct": stats["correct"],
            "invalid": stats["invalid"],
            "accuracy": 100.0 * stats["correct"] / stats["total"] if stats["total"] else 0.0,
            "by_subject": stats["by_subject"],
            "by_level": {str(k): v for k, v in stats["by_level"].items()},
        })

    if len(report) > 1:
        print("\n=== Summary ===")
        header = f"{'file':<70} {'correct':>8} {'total':>8} {'invalid':>8} {'acc(%)':>8}"
        print(header)
        print("-" * len(header))
        for entry in report:
            short = entry["path"] if len(entry["path"]) <= 70 else "..." + entry["path"][-67:]
            print(f"{short:<70} {entry['correct']:>8} {entry['total']:>8} {entry['invalid']:>8} {entry['accuracy']:>8.2f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
