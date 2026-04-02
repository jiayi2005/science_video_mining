# Science Video Mining Pipeline

This folder builds a science-focused real-speech evaluation set from online videos with manual subtitles.

## Key Constraints

- Real speech only (no synthetic TTS)
- Prefer manual subtitles over auto captions
- Rare/OOV science words (vs GigaSpeech vocab)
- Per sentence rare words <= 3
- Clip duration in 5-30s

## Files

- `build_science_manifest.py`: Main mining pipeline
- `seed_sources.tsv`: Small/default seed list
- `seed_sources_prod.tsv`: Expanded production seed list for large runs (10k+)
- `scripts/migrate_to_server.sh`: Sync project + vocab to server
- `scripts/run_prod.sh`: Server production entrypoint (resume + checkpoint ready)

## Requirements

- Python 3.9+
- `yt-dlp` in `PATH`
- `ffmpeg` in `PATH` (required for `--download-audio`)

## Local Quick Start

From `/Users/jiayi/Desktop/speech`:

```bash
python3 science_video_mining/build_science_manifest.py \
  --seed-file science_video_mining/seed_sources.tsv \
  --vocab-min1 /Users/jiayi/Desktop/speech/vocab_min_freq_1.txt \
  --vocab-min10 /Users/jiayi/Desktop/speech/vocab_min_freq_10.txt \
  --out-dir /Users/jiayi/Desktop/speech/science_eval_local \
  --target-count 200 \
  --max-videos-per-source 200 \
  --max-segments-per-video 10 \
  --max-rare-per-sentence 3 \
  --download-audio \
  --sample-rate 16000 \
  --cmd-timeout 60 \
  --save-every 50 \
  --resume \
  --path-mode relative \
  --dedup-key segment
```

## Full Server Workflow (Recommended for 10k)

### 1) Migrate Code + Vocab from Local to Server

```bash
cd /Users/jiayi/Desktop/speech/science_video_mining
./scripts/migrate_to_server.sh <server_user> <server_host> <remote_base> /Users/jiayi/Desktop/speech 22
```

Example:

```bash
./scripts/migrate_to_server.sh ubuntu 10.0.0.8 /data/speech /Users/jiayi/Desktop/speech 22
```

### 2) Install Server Dependencies

After SSH login:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg tmux
python3 -m pip install --user -U yt-dlp
export PATH="$HOME/.local/bin:$PATH"
```

### 3) Smoke Test First (100 clips)

```bash
cd <remote_base>/science_video_mining
./scripts/run_prod.sh <remote_base> <remote_base>/science_eval_smoke 100
```

`run_prod.sh` will automatically use `seed_sources_prod.tsv` if it exists.

### 4) Launch Full 10k Run in tmux

```bash
tmux new -s science10k
cd <remote_base>/science_video_mining
./scripts/run_prod.sh <remote_base> <remote_base>/science_eval_prod 10000 | tee -a <remote_base>/science_eval_prod/run.log
```

Detach: `Ctrl+b` then `d`

Reattach later:

```bash
tmux attach -t science10k
```

### 5) Monitor Progress

```bash
wc -l <remote_base>/science_eval_prod/manifest.jsonl
cat <remote_base>/science_eval_prod/stats.json
tail -n 40 <remote_base>/science_eval_prod/run.log
```

If interrupted, rerun the same command. `--resume` is already enabled.

## `run_prod.sh` Tunable Env Vars

You can tune scale without editing scripts:

```bash
export MAX_VIDEOS_PER_SOURCE=2500
export MAX_SEGMENTS_PER_VIDEO=30
export MAX_PER_DOMAIN=0
export CMD_TIMEOUT=90
export SAVE_EVERY=200
export SAMPLE_RATE=16000
export EXPAND_RETRIES=3
export EXPAND_BACKOFF=2.0
export FAIL_ON_SOURCE_EXPAND_ERROR=1
# Optional when server needs proxy to access YouTube:
# export YTDLP_PROXY="socks5://127.0.0.1:1080"
./scripts/run_prod.sh <base_dir> <out_dir> 10000
```

Defaults:

- `MAX_VIDEOS_PER_SOURCE=2000`
- `MAX_SEGMENTS_PER_VIDEO=30`
- `MAX_RARE_PER_SENTENCE=3`
- `MAX_PER_DOMAIN=0` (unlimited)
- `SAMPLE_RATE=16000`
- `CMD_TIMEOUT=60`
- `SAVE_EVERY=200`
- `EXPAND_RETRIES=3`
- `EXPAND_BACKOFF=2.0`
- `FAIL_ON_SOURCE_EXPAND_ERROR=0`
- `PATH_MODE=relative`
- `DEDUP_KEY=segment`

## Outputs

In `out-dir`:

- `manifest.json`: array format
- `manifest.jsonl`: line-delimited format
- `stats.json`: counters/domain distribution
- `subtitles/`: subtitle files
- `clips_wav/`: 5-30s clips (`.wav`)
- `full_audio/`: full source wav cache

Manifest record example:

```json
{
  "uid": 1,
  "URL": "https://www.youtube.com/watch?v=...",
  "start": 12.34,
  "end": 26.78,
  "text": "transcription",
  "audio_path": "clips_wav/xxxxx.wav"
}
```

## Notes

- Some sources may fail due to DRM, geo restrictions, or missing manual subtitles. The script skips automatically.
- For external sharing, prefer `URL/start/end/text` and keep media processing local.
- For finer timestamps, run MFA or Qwen3-ForcedAligner on exported `audio + text`.
