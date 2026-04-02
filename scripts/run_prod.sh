#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_prod.sh <base_dir> <out_dir> <target_count> [seed_file]
# Example:
#   ./scripts/run_prod.sh /data/speech /data/speech/science_eval_prod 10000
#   ./scripts/run_prod.sh /data/speech /data/speech/science_eval_prod 10000 /data/speech/science_video_mining/seed_sources_prod.tsv

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <base_dir> <out_dir> <target_count> [seed_file]" >&2
  exit 1
fi

BASE_DIR="$1"
OUT_DIR="$2"
TARGET_COUNT="$3"
PROJECT_DIR="${BASE_DIR}/science_video_mining"
VOCAB_MIN1="${BASE_DIR}/vocab_min_freq_1.txt"
VOCAB_MIN10="${BASE_DIR}/vocab_min_freq_10.txt"

DEFAULT_SEED_PROD="${PROJECT_DIR}/seed_sources_prod.tsv"
DEFAULT_SEED_BASE="${PROJECT_DIR}/seed_sources.tsv"
if [[ $# -ge 4 ]]; then
  SEED_FILE="$4"
elif [[ -f "${DEFAULT_SEED_PROD}" ]]; then
  SEED_FILE="${DEFAULT_SEED_PROD}"
else
  SEED_FILE="${DEFAULT_SEED_BASE}"
fi

MAX_VIDEOS_PER_SOURCE="${MAX_VIDEOS_PER_SOURCE:-2000}"
MAX_SEGMENTS_PER_VIDEO="${MAX_SEGMENTS_PER_VIDEO:-30}"
MAX_RARE_PER_SENTENCE="${MAX_RARE_PER_SENTENCE:-3}"
MAX_PER_DOMAIN="${MAX_PER_DOMAIN:-0}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
CMD_TIMEOUT="${CMD_TIMEOUT:-60}"
SAVE_EVERY="${SAVE_EVERY:-200}"
PATH_MODE="${PATH_MODE:-relative}"
DEDUP_KEY="${DEDUP_KEY:-segment}"
YTDLP_PROXY="${YTDLP_PROXY:-}"
EXPAND_RETRIES="${EXPAND_RETRIES:-3}"
EXPAND_BACKOFF="${EXPAND_BACKOFF:-2.0}"
FAIL_ON_SOURCE_EXPAND_ERROR="${FAIL_ON_SOURCE_EXPAND_ERROR:-0}"

if [[ ! -f "${SEED_FILE}" ]]; then
  echo "Seed file not found: ${SEED_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VOCAB_MIN1}" || ! -f "${VOCAB_MIN10}" ]]; then
  echo "Vocab files missing under ${BASE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PROJECT_DIR}/build_science_manifest.py" ]]; then
  echo "build_science_manifest.py missing: ${PROJECT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

export PATH="$HOME/.local/bin:$PATH"

echo "[run_prod] base_dir=${BASE_DIR}"
echo "[run_prod] out_dir=${OUT_DIR}"
echo "[run_prod] target_count=${TARGET_COUNT}"
echo "[run_prod] seed_file=${SEED_FILE}"
echo "[run_prod] max_videos_per_source=${MAX_VIDEOS_PER_SOURCE}"
echo "[run_prod] max_segments_per_video=${MAX_SEGMENTS_PER_VIDEO}"
echo "[run_prod] max_per_domain=${MAX_PER_DOMAIN}"
echo "[run_prod] sample_rate=${SAMPLE_RATE}"
echo "[run_prod] cmd_timeout=${CMD_TIMEOUT}"
echo "[run_prod] expand_retries=${EXPAND_RETRIES}"
echo "[run_prod] expand_backoff=${EXPAND_BACKOFF}"
echo "[run_prod] fail_on_source_expand_error=${FAIL_ON_SOURCE_EXPAND_ERROR}"
if [[ -n "${YTDLP_PROXY}" ]]; then
  echo "[run_prod] ytdlp_proxy=${YTDLP_PROXY}"
fi

CMD=(
  python3 "${PROJECT_DIR}/build_science_manifest.py"
  --seed-file "${SEED_FILE}"
  --vocab-min1 "${VOCAB_MIN1}"
  --vocab-min10 "${VOCAB_MIN10}"
  --out-dir "${OUT_DIR}"
  --target-count "${TARGET_COUNT}"
  --max-videos-per-source "${MAX_VIDEOS_PER_SOURCE}"
  --max-segments-per-video "${MAX_SEGMENTS_PER_VIDEO}"
  --max-rare-per-sentence "${MAX_RARE_PER_SENTENCE}"
  --max-per-domain "${MAX_PER_DOMAIN}"
  --download-audio
  --sample-rate "${SAMPLE_RATE}"
  --cmd-timeout "${CMD_TIMEOUT}"
  --expand-retries "${EXPAND_RETRIES}"
  --expand-backoff "${EXPAND_BACKOFF}"
  --save-every "${SAVE_EVERY}"
  --resume
  --path-mode "${PATH_MODE}"
  --dedup-key "${DEDUP_KEY}"
)

if [[ -n "${YTDLP_PROXY}" ]]; then
  CMD+=(--proxy "${YTDLP_PROXY}")
fi
if [[ "${FAIL_ON_SOURCE_EXPAND_ERROR}" == "1" ]]; then
  CMD+=(--fail-on-source-expand-error)
fi

"${CMD[@]}"
