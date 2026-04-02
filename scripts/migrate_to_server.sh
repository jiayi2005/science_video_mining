#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/migrate_to_server.sh <server_user> <server_host> <remote_base> [local_base] [ssh_port]
# Example:
#   ./scripts/migrate_to_server.sh ubuntu 10.0.0.8 /data/speech /Users/jiayi/Desktop/speech 22

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <server_user> <server_host> <remote_base> [local_base] [ssh_port]" >&2
  exit 1
fi

SERVER_USER="$1"
SERVER_HOST="$2"
REMOTE_BASE="$3"
LOCAL_BASE="${4:-/Users/jiayi/Desktop/speech}"
SSH_PORT="${5:-22}"

PROJECT_NAME="science_video_mining"
LOCAL_PROJECT="${LOCAL_BASE}/${PROJECT_NAME}"
REMOTE_PROJECT="${REMOTE_BASE}/${PROJECT_NAME}"

if [[ ! -d "${LOCAL_PROJECT}" ]]; then
  echo "Local project not found: ${LOCAL_PROJECT}" >&2
  exit 1
fi
if [[ ! -f "${LOCAL_BASE}/vocab_min_freq_1.txt" || ! -f "${LOCAL_BASE}/vocab_min_freq_10.txt" ]]; then
  echo "Vocab files missing under ${LOCAL_BASE}" >&2
  exit 1
fi

SSH_OPTS=("-p" "${SSH_PORT}")
REMOTE="${SERVER_USER}@${SERVER_HOST}"

echo "[1/4] Creating remote directories..."
ssh "${SSH_OPTS[@]}" "${REMOTE}" "mkdir -p '${REMOTE_PROJECT}' '${REMOTE_BASE}'"

echo "[2/4] Syncing project code (excluding demo_outputs)..."
rsync -avh --progress \
  -e "ssh -p ${SSH_PORT}" \
  --exclude 'demo_outputs' \
  "${LOCAL_PROJECT}/" \
  "${REMOTE}:${REMOTE_PROJECT}/"

echo "[3/4] Syncing vocab files..."
rsync -avh --progress \
  -e "ssh -p ${SSH_PORT}" \
  "${LOCAL_BASE}/vocab_min_freq_1.txt" \
  "${LOCAL_BASE}/vocab_min_freq_10.txt" \
  "${REMOTE}:${REMOTE_BASE}/"

echo "[4/4] Done. Next login command:"
echo "ssh -p ${SSH_PORT} ${REMOTE}"

echo
cat <<NEXT
After login, run:
  sudo apt-get update && sudo apt-get install -y ffmpeg
  python3 -m pip install --user -U yt-dlp
  export PATH="\$HOME/.local/bin:\$PATH"

Then start production run (example):
  python3 ${REMOTE_PROJECT}/build_science_manifest.py \\
    --seed-file ${REMOTE_PROJECT}/seed_sources.tsv \\
    --vocab-min1 ${REMOTE_BASE}/vocab_min_freq_1.txt \\
    --vocab-min10 ${REMOTE_BASE}/vocab_min_freq_10.txt \\
    --out-dir ${REMOTE_BASE}/science_eval_prod \\
    --target-count 5000 \\
    --max-videos-per-source 2000 \\
    --max-segments-per-video 30 \\
    --max-rare-per-sentence 3 \\
    --download-audio \\
    --sample-rate 16000 \\
    --cmd-timeout 60 \\
    --save-every 200 \\
    --resume \\
    --path-mode relative \\
    --dedup-key segment
NEXT
