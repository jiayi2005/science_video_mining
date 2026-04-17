#!/usr/bin/env python3
"""Candidate-only abbreviation mining from science videos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import build_abbreviation_manifest as base


def infer_pron_type(surface: str) -> str:
    if any(ch.isdigit() for ch in surface) and any(ch.isalpha() for ch in surface):
        return 'mixed_alnum'
    if any(ch.islower() for ch in surface) and any(ch.isupper() for ch in surface):
        return 'camel_case'
    if surface.isupper():
        return 'spell_out'
    return 'other'


def candidate_hits(text: str) -> List[Dict]:
    hits: List[Dict] = []
    seen = set()
    for candidate in base.extract_abbreviation_candidates(text):
        key = (candidate.start, candidate.end, candidate.normalized)
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                'surface': candidate.surface,
                'abbr': candidate.normalized,
                'canonical_reading': '',
                'pron_type': infer_pron_type(candidate.surface),
                'confidence_tier': 'candidate_only',
                'rule_id': '',
                'span_start': candidate.start,
                'span_end': candidate.end,
                'is_rule_confirmed': False,
            }
        )
    hits.sort(key=lambda item: (item['span_start'], item['span_end']))
    return hits


def choose_primary_hit(hits: Sequence[Dict]) -> Dict:
    return sorted(
        hits,
        key=lambda item: (
            0 if item['pron_type'] != 'spell_out' else 1,
            item['abbr'],
            item['span_start'],
        ),
    )[0]


def score_segment_candidate_only(
    seg: base.Segment,
    abbr_counter: Counter,
    *,
    max_per_abbr: int,
    max_abbrs_per_sentence: int,
) -> Optional[Dict]:
    hits = candidate_hits(seg.text)
    if not hits:
        return None
    if len(hits) > max_abbrs_per_sentence:
        return None
    for abbr in {hit['abbr'] for hit in hits}:
        if abbr_counter[abbr] >= max_per_abbr:
            return None
    primary = choose_primary_hit(hits)
    return {
        'text_expanded': seg.text,
        'abbreviations': hits,
        'primary_abbr': primary['abbr'],
        'primary_pron_type': primary['pron_type'],
        'primary_confidence_tier': 'candidate_only',
        'match_mode': 'candidate_only',
    }


def save_sample_review(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'uid', 'abbr', 'surface', 'pron_type', 'text', 'audio_path', 'source_domain'
    ]
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_stats(
    out_dir: Path,
    *,
    target_count: int,
    final_count: int,
    inspected_videos: int,
    kept_videos: int,
    source_expand_failures: int,
    video_info_failures: int,
    abbr_counter: Counter,
    pron_counter: Counter,
) -> Dict:
    stats = {
        'target_count': target_count,
        'final_count': final_count,
        'inspected_videos': inspected_videos,
        'kept_videos': kept_videos,
        'source_expand_failures': source_expand_failures,
        'video_info_failures': video_info_failures,
        'match_mode': 'candidate_only',
        'abbr_counter': dict(sorted(abbr_counter.items())),
        'pron_type_counter': dict(sorted(pron_counter.items())),
    }
    with (out_dir / 'stats.json').open('w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def restore_resume_state(
    records: List[Dict],
    existing_video_segments: Dict[str, int],
    abbr_counter: Counter,
    pron_counter: Counter,
) -> int:
    uid = 0
    for record in records:
        uid = max(uid, int(record.get('uid', 0) or 0))
        video_id = str(record.get('video_id') or '')
        if video_id:
            existing_video_segments[video_id] += 1
        seen_abbrs = set()
        seen_pron = set()
        for hit in record.get('abbreviations') or []:
            if not isinstance(hit, dict):
                continue
            abbr = str(hit.get('abbr') or '')
            pron_type = str(hit.get('pron_type') or '')
            if abbr and abbr not in seen_abbrs:
                abbr_counter[abbr] += 1
                seen_abbrs.add(abbr)
            if pron_type and pron_type not in seen_pron:
                pron_counter[pron_type] += 1
                seen_pron.add(pron_type)
    return uid


def build_manifest(args: argparse.Namespace) -> int:
    base.CMD_TIMEOUT_SEC = float(args.cmd_timeout)
    base.YTDLP_PROXY = (args.proxy or '').strip()
    base.YTDLP_RETRIES = max(0, int(args.expand_retries))
    base.YTDLP_RETRY_BACKOFF_SEC = max(0.0, float(args.expand_backoff))
    if base.YTDLP_PROXY:
        print(f'[INFO] yt-dlp proxy={base.YTDLP_PROXY}', file=sys.stderr, flush=True)

    base.validate_dependencies(download_audio=args.download_audio)
    sources = base.read_seed_sources(Path(args.seed_file))
    if not sources:
        raise RuntimeError(f'No valid sources found in {args.seed_file}')

    out_dir = Path(args.out_dir)
    subtitle_dir = out_dir / 'subtitles'
    full_audio_dir = out_dir / 'full_audio'
    clips_dir = out_dir / 'clips_wav'
    out_json = out_dir / 'manifest.json'
    out_jsonl = out_dir / 'manifest.jsonl'
    audit_jsonl = out_dir / 'rule_audit.jsonl'
    review_tsv = out_dir / 'sample_review.tsv'
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    audit_rows: List[Dict] = []
    review_rows: List[Dict] = []
    uid = 0
    inspected_videos = 0
    kept_videos = 0
    source_expand_failures = 0
    video_info_failures = 0
    seen_dedup_keys = set()
    existing_video_segments: Dict[str, int] = defaultdict(int)
    abbr_counter: Counter = Counter()
    pron_counter: Counter = Counter()

    if args.resume and out_json.exists():
        try:
            loaded = json.loads(out_json.read_text(encoding='utf-8'))
            if isinstance(loaded, list):
                records = loaded
        except Exception:
            records = []
        uid = restore_resume_state(records, existing_video_segments, abbr_counter, pron_counter)
        for record in records:
            if args.dedup_key == 'text':
                text = record.get('text')
                if isinstance(text, str):
                    seen_dedup_keys.add(text.strip().lower())
            elif args.dedup_key == 'segment':
                video_id = str(record.get('video_id') or '')
                start = record.get('start')
                end = record.get('end')
                if video_id and isinstance(start, (float, int)) and isinstance(end, (float, int)):
                    seen_dedup_keys.add(f'{video_id}:{float(start):.3f}:{float(end):.3f}')
        print(f'[INFO] resume loaded records={len(records)}', file=sys.stderr, flush=True)

    for source in sources:
        if uid >= args.target_count:
            break
        print(f'[INFO] source={source.url}', file=sys.stderr, flush=True)
        video_urls = base.expand_video_urls(
            source.url,
            args.max_videos_per_source,
            retries=args.expand_retries,
            backoff_sec=args.expand_backoff,
            fail_on_error=args.fail_on_source_expand_error,
        )
        if not video_urls:
            source_expand_failures += 1
            continue

        for video_url in video_urls:
            if uid >= args.target_count:
                break
            inspected_videos += 1
            info, info_err = base.ytdlp_dump_json(video_url, flat_playlist=False, playlist_end=None)
            if not info:
                video_info_failures += 1
                if info_err:
                    print(f'[WARN] video info failed url={video_url} err={info_err}', file=sys.stderr, flush=True)
                continue
            subtitle_lang = base.find_manual_sub_lang(info, args.lang)
            if not subtitle_lang:
                continue
            video_id = str(info.get('id') or '')
            if not video_id:
                continue
            if existing_video_segments[video_id] >= args.max_segments_per_video:
                continue

            subtitle_path = base.download_manual_subtitle(video_url, video_id, subtitle_dir, subtitle_lang)
            if subtitle_path is None or not subtitle_path.exists():
                continue
            cues = base.parse_subtitle_file(subtitle_path)
            if not cues:
                continue
            segments = base.cues_to_segments(
                cues,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                max_gap=args.max_gap,
            )
            if not segments:
                continue

            full_audio_path: Optional[Path] = None
            if args.download_audio:
                full_audio_path = base.ensure_full_audio(
                    video_url=video_url,
                    video_id=video_id,
                    full_audio_dir=full_audio_dir,
                    sample_rate=args.sample_rate,
                )
                if full_audio_path is None:
                    continue

            added_this_video = 0
            segments.sort(key=lambda seg: (len(base.extract_abbreviation_candidates(seg.text)), seg.start))
            for seg in segments:
                if uid >= args.target_count:
                    break
                if added_this_video >= args.max_segments_per_video:
                    break
                if existing_video_segments[video_id] >= args.max_segments_per_video:
                    break
                dedup_key = base.dedup_key_for_segment(seg, video_id, args.dedup_key)
                if dedup_key is not None and dedup_key in seen_dedup_keys:
                    continue

                score = score_segment_candidate_only(
                    seg,
                    abbr_counter,
                    max_per_abbr=args.max_per_abbr,
                    max_abbrs_per_sentence=args.max_abbrs_per_sentence,
                )
                if score is None:
                    continue

                clip_name = f'{video_id}_{int(seg.start * 1000):09d}_{int(seg.end * 1000):09d}.wav'
                clip_path = clips_dir / clip_name
                audio_manifest_path = ''
                if args.download_audio:
                    assert full_audio_path is not None
                    ok = base.cut_clip(
                        full_audio=full_audio_path,
                        clip_out=clip_path,
                        start=seg.start,
                        end=seg.end,
                        sample_rate=args.sample_rate,
                    )
                    if not ok:
                        continue
                    audio_manifest_path = base.manifest_path(clip_path, out_dir, args.path_mode)

                uid += 1
                record = {
                    'uid': uid,
                    'URL': str(info.get('webpage_url') or video_url),
                    'video_id': video_id,
                    'start': round(seg.start, 3),
                    'end': round(seg.end, 3),
                    'audio_path': audio_manifest_path,
                    'subtitle_path': base.manifest_path(subtitle_path, out_dir, args.path_mode),
                    'text': seg.text,
                    'text_expanded': score['text_expanded'],
                    'abbreviations': score['abbreviations'],
                    'primary_abbr': score['primary_abbr'],
                    'primary_pron_type': score['primary_pron_type'],
                    'primary_confidence_tier': score['primary_confidence_tier'],
                    'source_domain': source.domain,
                    'source_note': source.note,
                    'match_mode': 'candidate_only',
                }
                records.append(record)

                seen_abbrs = set()
                seen_pron = set()
                for hit in score['abbreviations']:
                    if hit['abbr'] not in seen_abbrs:
                        abbr_counter[hit['abbr']] += 1
                        seen_abbrs.add(hit['abbr'])
                    if hit['pron_type'] not in seen_pron:
                        pron_counter[hit['pron_type']] += 1
                        seen_pron.add(hit['pron_type'])
                    audit_rows.append(
                        {
                            'uid': uid,
                            'video_id': video_id,
                            'surface': hit['surface'],
                            'abbr': hit['abbr'],
                            'canonical_reading': '',
                            'pron_type': hit['pron_type'],
                            'confidence_tier': 'candidate_only',
                            'rule_id': '',
                            'context_text': seg.text,
                            'match_mode': 'candidate_only',
                        }
                    )
                    review_rows.append(
                        {
                            'uid': uid,
                            'abbr': hit['abbr'],
                            'surface': hit['surface'],
                            'pron_type': hit['pron_type'],
                            'text': seg.text,
                            'audio_path': audio_manifest_path,
                            'source_domain': source.domain,
                        }
                    )

                if dedup_key is not None:
                    seen_dedup_keys.add(dedup_key)
                added_this_video += 1
                existing_video_segments[video_id] += 1

                if args.save_every > 0 and uid % args.save_every == 0:
                    base.save_manifest(records, out_json, out_jsonl)
                    base.save_rule_audit(audit_rows, audit_jsonl)
                    save_sample_review(review_rows, review_tsv)
                    write_stats(
                        out_dir=out_dir,
                        target_count=args.target_count,
                        final_count=len(records),
                        inspected_videos=inspected_videos,
                        kept_videos=kept_videos,
                        source_expand_failures=source_expand_failures,
                        video_info_failures=video_info_failures,
                        abbr_counter=abbr_counter,
                        pron_counter=pron_counter,
                    )
                    print(f'[INFO] checkpoint saved records={len(records)}', file=sys.stderr, flush=True)

            if added_this_video > 0:
                kept_videos += 1
                print(f'[INFO] kept video={video_url} segments={added_this_video} total={uid}', file=sys.stderr, flush=True)
            base.cleanup_full_audio(full_audio_path, keep_full_audio=args.keep_full_audio)

    base.save_manifest(records, out_json, out_jsonl)
    base.save_rule_audit(audit_rows, audit_jsonl)
    save_sample_review(review_rows, review_tsv)
    stats = write_stats(
        out_dir=out_dir,
        target_count=args.target_count,
        final_count=len(records),
        inspected_videos=inspected_videos,
        kept_videos=kept_videos,
        source_expand_failures=source_expand_failures,
        video_info_failures=video_info_failures,
        abbr_counter=abbr_counter,
        pron_counter=pron_counter,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Mine candidate-only abbreviation-focused science video segments.')
    parser.add_argument('--seed-file', required=True, help='TSV source list: domain<TAB>url<TAB>note')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--target-count', type=int, default=2000)
    parser.add_argument('--lang', default='en')
    parser.add_argument('--max-videos-per-source', type=int, default=400)
    parser.add_argument('--max-segments-per-video', type=int, default=10)
    parser.add_argument('--max-per-abbr', type=int, default=3)
    parser.add_argument('--max-abbrs-per-sentence', type=int, default=2)
    parser.add_argument('--min-duration', type=float, default=5.0)
    parser.add_argument('--max-duration', type=float, default=30.0)
    parser.add_argument('--max-gap', type=float, default=1.2)
    parser.add_argument('--download-audio', action='store_true', default=False)
    parser.add_argument('--sample-rate', type=int, default=16000, choices=[16000, 22050, 24000])
    parser.add_argument('--keep-full-audio', action='store_true', default=False)
    parser.add_argument('--cmd-timeout', type=float, default=45.0)
    parser.add_argument('--proxy', default='')
    parser.add_argument('--expand-retries', type=int, default=2)
    parser.add_argument('--expand-backoff', type=float, default=2.0)
    parser.add_argument('--fail-on-source-expand-error', action='store_true', default=False)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--path-mode', choices=['absolute', 'relative'], default='absolute')
    parser.add_argument('--dedup-key', choices=['text', 'segment', 'none'], default='text')
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return build_manifest(args)
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
