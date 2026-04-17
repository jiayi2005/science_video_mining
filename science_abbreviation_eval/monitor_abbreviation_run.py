#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


FAIL_PATTERNS = {
    'video_info_failed': re.compile(r'video info failed', re.IGNORECASE),
    'expand_failed': re.compile(r'expand failed', re.IGNORECASE),
    'connection_reset': re.compile(r'connection reset by peer', re.IGNORECASE),
    'read_timeout': re.compile(r'read timed out|timeout after', re.IGNORECASE),
}
SOURCE_RE = re.compile(r'\[INFO\]\s+source=(.+)')


def run_cmd(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc


def tmux_alive(session_name: str) -> bool:
    return run_cmd(['tmux', 'has-session', '-t', session_name]).returncode == 0


def process_alive(process_pattern: str, run_dir: str) -> bool:
    proc = run_cmd(['ps', '-ef'])
    hay = proc.stdout.splitlines()
    for line in hay:
        if process_pattern in line and run_dir in line and 'grep -E' not in line:
            return True
    return False


def read_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)


def count_wavs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.wav')


def read_log_delta(log_path: Path, offset: int) -> tuple[str, int]:
    if not log_path.exists():
        return '', 0
    size = log_path.stat().st_size
    if offset < 0 or offset > size:
        offset = 0
    with log_path.open('rb') as f:
        f.seek(offset)
        data = f.read()
    return data.decode('utf-8', errors='ignore'), size


def classify_speed(delta_live: Optional[int]) -> str:
    if delta_live is None:
        return 'baseline'
    if delta_live >= 20:
        return '正常'
    if delta_live >= 5:
        return '偏慢'
    return '异常'


def classify_failures(failure_lines: int) -> str:
    if failure_lines < 10:
        return '网络波动可接受'
    if failure_lines <= 30:
        return '失败偏高'
    return '失败异常'


def extract_current_source(chunk: str, previous: str) -> str:
    matches = SOURCE_RE.findall(chunk)
    if matches:
        return matches[-1].strip()
    return previous


def choose_next_pool(current_pool: str, order: List[str]) -> Optional[str]:
    if current_pool == 'broad_pool':
        return 'stable_ted' if 'stable_ted' in order else None
    if current_pool == 'stable_ted':
        return 'stable_direct' if 'stable_direct' in order else None
    if current_pool == 'stable_direct':
        return 'broad_pool' if 'broad_pool' in order else None
    return order[0] if order else None


def summarize_line(summary: Dict) -> str:
    return (
        f"[{summary['timestamp']}] tmux={'alive' if summary['tmux_alive'] else 'dead'} "
        f"proc={'alive' if summary['process_alive'] else 'dead'} "
        f"pool={summary['source_pool']} source={summary['current_source'] or '-'} "
        f"saved={summary['saved_count']} live={summary['live_clip_count']} "
        f"delta_saved={summary['delta_saved']} delta_live={summary['delta_live']} "
        f"fail_lines={summary['failure_lines']} status={summary['status_level']} "
        f"action={summary['suggested_action']}"
    )


def run_once(args: argparse.Namespace) -> Dict:
    run_dir = Path(args.run_dir)
    monitor_dir = run_dir / 'monitoring'
    monitor_dir.mkdir(parents=True, exist_ok=True)
    state_path = monitor_dir / 'monitor_state.json'
    latest_path = monitor_dir / 'monitor_latest.json'
    events_path = monitor_dir / 'monitor_events.jsonl'
    summary_log = monitor_dir / 'monitor_summary.log'
    alert_path = monitor_dir / 'ALERT.txt'

    source_cfg = json.loads(Path(args.source_config).read_text(encoding='utf-8'))
    order = list(source_cfg.get('order', []))
    next_pool = choose_next_pool(args.source_pool, order)
    next_seed = source_cfg.get(next_pool or '', '')

    prev = {}
    if state_path.exists():
        try:
            prev = json.loads(state_path.read_text(encoding='utf-8'))
        except Exception:
            prev = {}

    prev_saved = prev.get('saved_count')
    prev_live = prev.get('live_clip_count')
    prev_offset = int(prev.get('log_offset', 0) or 0)
    prev_source = str(prev.get('current_source') or '')
    prev_streak = int(prev.get('abnormal_streak', 0) or 0)

    manifest_path = run_dir / 'manifest.jsonl'
    clips_dir = run_dir / 'clips_wav'
    log_path = run_dir / 'run.log'

    saved_count = read_count(manifest_path)
    live_count = count_wavs(clips_dir)
    chunk, new_offset = read_log_delta(log_path, prev_offset)

    lines = [line for line in chunk.splitlines() if line.strip()]
    failure_counts = {name: 0 for name in FAIL_PATTERNS}
    failure_lines = 0
    for line in lines:
        matched = False
        for name, pattern in FAIL_PATTERNS.items():
            if pattern.search(line):
                failure_counts[name] += 1
                matched = True
        if matched:
            failure_lines += 1

    current_source = extract_current_source(chunk, prev_source)
    delta_saved = None if prev_saved is None else saved_count - int(prev_saved)
    delta_live = None if prev_live is None else live_count - int(prev_live)
    unsaved_gap = max(0, live_count - saved_count)

    speed_status = classify_speed(delta_live)
    failure_status = classify_failures(failure_lines)
    tmux_ok = tmux_alive(args.session_name)
    proc_ok = process_alive(args.process_pattern, str(run_dir))

    abnormal_streak = prev_streak
    if speed_status == '异常':
        abnormal_streak += 1
    else:
        abnormal_streak = 0

    if delta_live is None:
        status_level = 'baseline'
        suggested_action = '观察'
    elif not tmux_ok or not proc_ok:
        status_level = '异常'
        suggested_action = '建议切源'
    elif failure_status == '失败异常':
        status_level = '异常'
        suggested_action = '建议切源'
    elif abnormal_streak >= 2:
        status_level = '异常'
        suggested_action = '建议切源'
    elif speed_status == '异常':
        status_level = '异常'
        suggested_action = '观察'
    elif speed_status == '偏慢' or failure_status == '失败偏高':
        status_level = '偏慢'
        suggested_action = '观察'
    else:
        status_level = '正常'
        suggested_action = '继续'

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z').strip()
    summary = {
        'timestamp': timestamp,
        'tmux_alive': tmux_ok,
        'process_alive': proc_ok,
        'saved_count': saved_count,
        'live_clip_count': live_count,
        'delta_saved': delta_saved,
        'delta_live': delta_live,
        'failure_lines': failure_lines,
        'failure_counts': failure_counts,
        'failure_status': failure_status,
        'source_pool': args.source_pool,
        'current_source': current_source,
        'status_level': status_level,
        'suggested_action': suggested_action,
        'abnormal_streak': abnormal_streak,
        'unsaved_gap': unsaved_gap,
        'checkpoint_risk': unsaved_gap > 0,
        'next_pool': next_pool,
        'next_seed': next_seed,
        'session_name': args.session_name,
    }

    latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with events_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')
    with summary_log.open('a', encoding='utf-8') as f:
        f.write(summarize_line(summary) + '\n')

    if suggested_action == '建议切源':
        msg = summarize_line(summary)
        if summary['checkpoint_risk']:
            msg += f" checkpoint_risk=1 unsaved_gap={unsaved_gap}"
        if next_pool and next_seed:
            msg += f" next_pool={next_pool} next_seed={next_seed}"
        alert_path.write_text(msg + '\n', encoding='utf-8')
    elif alert_path.exists():
        alert_path.unlink()

    state = {
        'saved_count': saved_count,
        'live_clip_count': live_count,
        'log_offset': new_offset,
        'current_source': current_source,
        'abnormal_streak': abnormal_streak,
        'timestamp': timestamp,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description='Monitor abbreviation mining speed and suggest source switching.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--session-name', required=True)
    parser.add_argument('--source-pool', required=True, choices=['stable_ted', 'stable_direct', 'broad_pool'])
    parser.add_argument('--source-config', required=True)
    parser.add_argument('--process-pattern', default='build_abbreviation_candidates.py')
    parser.add_argument('--interval-sec', type=int, default=600)
    parser.add_argument('--loop', action='store_true', default=False)
    args = parser.parse_args()

    if not args.loop:
        summary = run_once(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    while True:
        summary = run_once(args)
        print(summarize_line(summary), flush=True)
        time.sleep(max(1, int(args.interval_sec)))


if __name__ == '__main__':
    raise SystemExit(main())
