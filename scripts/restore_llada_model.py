import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


DEFAULT_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_LOCAL_DIR = Path("/dev/shm/LLaDA-8B-Instruct")
REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
)


@dataclass(frozen=True)
class RestoreResult:
    status: str
    path: Path


def _indexed_weight_files(index_path: Path) -> set[str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map", {})
    return {str(filename) for filename in weight_map.values()}


def is_model_complete(model_dir: Path) -> bool:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return False

    for filename in REQUIRED_FILES:
        if not (model_dir / filename).is_file():
            return False

    try:
        weight_files = _indexed_weight_files(model_dir / "model.safetensors.index.json")
    except (OSError, json.JSONDecodeError, TypeError):
        return False

    if not weight_files:
        return False

    for filename in weight_files:
        path = model_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            return False

    return True


def restore_model(
    model_id: str = DEFAULT_MODEL_ID,
    local_dir: Path = DEFAULT_LOCAL_DIR,
    max_workers: int = 4,
    force: bool = False,
) -> RestoreResult:
    local_dir = Path(local_dir)
    if not force and is_model_complete(local_dir):
        return RestoreResult(status="already_present", path=local_dir)

    downloaded_path = snapshot_download(
        model_id,
        local_dir=str(local_dir),
        max_workers=max_workers,
    )
    return RestoreResult(status="downloaded", path=Path(downloaded_path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Restore the LLaDA-8B-Instruct model to /dev/shm via ModelScope.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--local_dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Download even if the target directory looks complete.")
    args = parser.parse_args(argv)

    result = restore_model(
        model_id=args.model_id,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
        force=args.force,
    )
    if result.status == "already_present":
        print(f"model already complete: {result.path}")
    else:
        print(f"downloaded model to: {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
