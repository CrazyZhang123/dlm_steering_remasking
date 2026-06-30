#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/root/myproject/DLM_Steering_Remasking}"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_path="${2:-${repo_root}/DLM_Steering_Remasking_selected_non_archives_${timestamp}.zip}"
output_base="$(basename "${output_path}")"

if [[ ! -d "${repo_root}" ]]; then
  echo "Repository directory does not exist: ${repo_root}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output_path}")"

echo "Repository: ${repo_root}"
echo "Output: ${output_path}"
echo "Scope: assets docs scripts utils outputs, plus non-archive files in repository root."
echo "Mode: create archive only; originals are not deleted or modified."

python - "${repo_root}" "${output_path}" <<'PY'
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()

archive_suffixes = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".zst",
)
excluded_dirs = {".git", ".pytest_cache", "__pycache__"}
included_dirs = ("assets", "docs", "scripts", "utils", "outputs")


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in archive_suffixes)


written = 0
with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
    for path in sorted(repo_root.iterdir()):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == output_path:
            continue
        if is_archive(path):
            continue
        archive.write(path, path.relative_to(repo_root))
        written += 1

    for dirname in included_dirs:
        dir_path = repo_root / dirname
        if not dir_path.exists():
            print(f"Skipping missing directory: {dirname}")
            continue
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [dirname for dirname in dirs if dirname not in excluded_dirs]
            root_path = Path(root)
            for filename in files:
                path = root_path / filename
                resolved = path.resolve()
                if resolved == output_path:
                    continue
                if is_archive(path):
                    continue
                archive.write(path, path.relative_to(repo_root))
                written += 1

print(f"Files archived: {written}")
PY

echo "Archive created:"
ls -lh "${output_path}"
