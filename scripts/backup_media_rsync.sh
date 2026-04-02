#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/backup_media_rsync.sh <src_out_dir> <dst_backup_dir>
#
# Examples:
#   # Server -> local backup
#   ./scripts/backup_media_rsync.sh \
#     DB93-tunnel:/DB/rhome/heyangliu/speech/science_eval_prod \
#     /Users/jiayi/Desktop/speech_backups/science_eval_prod
#
#   # Local -> NAS/remote backup
#   ./scripts/backup_media_rsync.sh \
#     /Users/jiayi/Desktop/speech/science_eval_prod \
#     backup@10.0.0.8:/data/speech_backups/science_eval_prod

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <src_out_dir> <dst_backup_dir>" >&2
  exit 1
fi

SRC="$1"
DST="$2"

is_remote_path() {
  [[ "$1" == *:* ]]
}

if ! is_remote_path "${DST}"; then
  mkdir -p "${DST}"
fi

echo "[backup] src=${SRC}"
echo "[backup] dst=${DST}"

RSYNC_RESUME_OPT="--append"
if rsync --help 2>/dev/null | grep -q -- '--append-verify'; then
  RSYNC_RESUME_OPT="--append-verify"
fi
echo "[backup] rsync resume option: ${RSYNC_RESUME_OPT}"

sync_optional_file() {
  local fname="$1"
  set +e
  rsync -avh "${SRC}/${fname}" "${DST}/"
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "[backup] skip file: ${fname}"
  fi
}

sync_optional_dir() {
  local dname="$1"
  set +e
  rsync -avh --partial "${RSYNC_RESUME_OPT}" "${SRC}/${dname}/" "${DST}/${dname}/"
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "[backup] skip dir: ${dname}"
  fi
}

# Metadata first (small, important)
sync_optional_file "manifest.json"
sync_optional_file "manifest.jsonl"
sync_optional_file "stats.json"
sync_optional_file "run.log"

# Bulk media data
sync_optional_dir "clips_wav"
sync_optional_dir "full_audio"
sync_optional_dir "subtitles"

echo "[backup] done"
