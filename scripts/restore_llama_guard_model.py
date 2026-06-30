import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.restore_llada_model import restore_model


DEFAULT_MODEL_ID = "LLM-Research/Llama-Guard-4-12B"
DEFAULT_LOCAL_DIR = Path("/dev/shm/Llama-Guard-4-12B")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Restore the Llama-Guard-4-12B model to /dev/shm via ModelScope.")
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
