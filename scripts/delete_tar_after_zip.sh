#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/root/myproject/DLM_Steering_Remasking}"
zip_session="${2:-stage_repo_zip_non_archives}"
zip_path="${3:-${repo_root}/DLM_Steering_Remasking_non_archives_20260628_081423.zip}"
tar_path="${4:-${repo_root}/DLM_Steering_Remasking_non_archives_20260628_080933.tar.gz}"
log_path="${repo_root}/outputs/delete_tar_after_zip.log"

mkdir -p "$(dirname "${log_path}")"

echo "[$(date -Is)] Waiting for zip session '${zip_session}' to finish" | tee -a "${log_path}"
while tmux has-session -t "${zip_session}" 2>/dev/null; do
  sleep 30
done

echo "[$(date -Is)] Zip session ended; validating zip" | tee -a "${log_path}"
grep -q "Archive created:" "${repo_root}/outputs/compress_non_archives.log"

python - "${zip_path}" <<'PY' | tee -a "${log_path}"
import sys
import zipfile

zip_path = sys.argv[1]
with zipfile.ZipFile(zip_path) as archive:
    entry_count = len(archive.infolist())
    if entry_count <= 0:
        raise SystemExit("zip has no entries")
print(f"zip entries: {entry_count}")
PY

if [[ -f "${tar_path}" ]]; then
  rm -f -- "${tar_path}"
  echo "[$(date -Is)] Deleted ${tar_path}" | tee -a "${log_path}"
else
  echo "[$(date -Is)] tar.gz already absent: ${tar_path}" | tee -a "${log_path}"
fi
