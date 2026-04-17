#!/usr/bin/env python3
"""
Build an abbreviation-focused ASR evaluation manifest from science videos.

This script is independent from the rare-word mining pipeline. It mines
subtitle-aligned speech segments that contain abbreviations confirmed by a
rule table, then emits audio clips plus manifest/audit files for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


TIMESPAN_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{2,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{2,3})"
)
SENT_END_RE = re.compile(r"[.!?。！？]$")
TAG_RE = re.compile(r"<[^>]+>")
MUSIC_CUE_RE = re.compile(r"^\[(music|applause|laughter|inaudible).*\]$", re.IGNORECASE)
ABBR_PATTERN = re.compile(r"\b((?=[A-Z0-9]*[A-Z])[A-Z0-9]{2,}|[a-z]+[A-Z][a-zA-Z0-9]*)\b")

VALID_PRON_TYPES = {"spell_out", "lexicalized", "mixed_alnum", "camel_case", "other"}
VALID_CONFIDENCE_TIERS = {"high_conf", "low_conf"}

CMD_TIMEOUT_SEC = 45.0
YTDLP_PROXY = ""
YTDLP_RETRIES = 2
YTDLP_RETRY_BACKOFF_SEC = 2.0


@dataclasses.dataclass
class Source:
    domain: str
    url: str
    note: str = ""


@dataclasses.dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclasses.dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclasses.dataclass(frozen=True)
class AbbreviationRule:
    abbr: str
    canonical_reading: str
    pron_type: str
    confidence_tier: str
    rule_id: str
    notes: str = ""


@dataclasses.dataclass
class Candidate:
    surface: str
    start: int
    end: int
    normalized: str


def summarize_stderr(stderr: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", (stderr or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def run_cmd(
    cmd: Sequence[str],
    *,
    check: bool = True,
    timeout_sec: Optional[float] = None,
) -> subprocess.CompletedProcess:
    if timeout_sec is None:
        timeout_sec = CMD_TIMEOUT_SEC
    try:
        proc = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout
        stderr = exc.stderr
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="ignore")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="ignore")
        proc = subprocess.CompletedProcess(
            args=list(cmd),
            returncode=124,
            stdout=stdout or "",
            stderr=(stderr or "") + f"\n[TIMEOUT after {timeout_sec}s]",
        )
    if check and proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n"
            f"exit={proc.returncode}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    return proc


def ytdlp_base_cmd() -> List[str]:
    cmd = ["yt-dlp"]
    if YTDLP_PROXY:
        cmd.extend(["--proxy", YTDLP_PROXY])
    return cmd


def read_seed_sources(seed_path: Path) -> List[Source]:
    sources: List[Source] = []
    with seed_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                sources.append(Source(domain="unknown", url=parts[0], note=""))
            elif len(parts) == 2:
                sources.append(Source(domain=parts[0].strip(), url=parts[1].strip(), note=""))
            else:
                sources.append(
                    Source(
                        domain=parts[0].strip(),
                        url=parts[1].strip(),
                        note=parts[2].strip(),
                    )
                )
    return sources


def parse_timecode(ts: str) -> float:
    ts = ts.replace(",", ".").strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        raise ValueError(f"Unsupported timecode: {ts}")
    return int(h) * 3600 + int(m) * 60 + float(s)


def normalize_caption_text(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_subtitle_file(path: Path) -> List[Cue]:
    cues: List[Cue] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = TIMESPAN_RE.search(line)
        if not m:
            i += 1
            continue
        start = parse_timecode(m.group("start"))
        end = parse_timecode(m.group("end"))
        i += 1
        text_lines: List[str] = []
        while i < len(lines):
            cur = lines[i].strip()
            if not cur:
                break
            if TIMESPAN_RE.search(cur):
                i -= 1
                break
            if cur.startswith("NOTE") or cur.startswith("STYLE") or cur.startswith("WEBVTT"):
                i += 1
                continue
            text_lines.append(cur)
            i += 1
        text = normalize_caption_text(" ".join(text_lines))
        if text and not MUSIC_CUE_RE.match(text):
            cues.append(Cue(start=start, end=end, text=text))
        i += 1
    return cues


def cues_to_segments(
    cues: Sequence[Cue],
    *,
    min_duration: float,
    max_duration: float,
    max_gap: float = 1.2,
) -> List[Segment]:
    segments: List[Segment] = []
    cur_text: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

    def flush() -> None:
        nonlocal cur_text, cur_start, cur_end
        if cur_start is None or cur_end is None:
            cur_text = []
            return
        text = normalize_caption_text(" ".join(cur_text))
        if not text:
            cur_text = []
            cur_start = None
            cur_end = None
            return
        seg = Segment(start=cur_start, end=cur_end, text=text)
        if min_duration <= seg.duration <= max_duration:
            segments.append(seg)
        cur_text = []
        cur_start = None
        cur_end = None

    for cue in cues:
        if cur_start is None:
            cur_start = cue.start
            cur_end = cue.end
            cur_text = [cue.text]
            continue

        assert cur_end is not None
        if cue.start - cur_end > max_gap:
            flush()
            cur_start = cue.start
            cur_end = cue.end
            cur_text = [cue.text]
            continue

        cur_end = max(cur_end, cue.end)
        cur_text.append(cue.text)
        if (cur_end - cur_start) >= max_duration or SENT_END_RE.search(cue.text):
            flush()

    flush()
    return segments


def ytdlp_dump_json(
    url: str,
    *,
    flat_playlist: bool = False,
    playlist_end: Optional[int] = None,
    retries: Optional[int] = None,
    backoff_sec: Optional[float] = None,
) -> Tuple[Dict, str]:
    if retries is None:
        retries = YTDLP_RETRIES
    if backoff_sec is None:
        backoff_sec = YTDLP_RETRY_BACKOFF_SEC

    cmd = ytdlp_base_cmd() + ["--dump-single-json", "--skip-download"]
    if flat_playlist:
        cmd.append("--flat-playlist")
    if playlist_end is not None and playlist_end > 0:
        cmd.extend(["--playlist-end", str(playlist_end)])
    cmd.append(url)

    attempts = max(0, retries) + 1
    last_err = ""
    for attempt in range(attempts):
        proc = run_cmd(cmd, check=False)
        if proc.returncode == 0:
            try:
                parsed = json.loads(proc.stdout)
            except json.JSONDecodeError:
                last_err = "yt-dlp returned invalid JSON"
            else:
                if isinstance(parsed, dict):
                    return parsed, ""
                last_err = f"yt-dlp JSON root type is {type(parsed).__name__}, expected dict"
        else:
            last_err = summarize_stderr(proc.stderr) or f"yt-dlp exit code {proc.returncode}"
        if attempt + 1 < attempts:
            time.sleep(max(0.0, backoff_sec) * (attempt + 1))
    return {}, last_err


def is_direct_video_url(url: str) -> bool:
    return ("watch?v=" in url) or ("youtu.be/" in url) or ("/talks/" in url)


def expand_video_urls(
    source_url: str,
    max_videos_per_source: int,
    *,
    retries: int,
    backoff_sec: float,
    fail_on_error: bool,
) -> List[str]:
    if is_direct_video_url(source_url):
        return [source_url]

    data, err = ytdlp_dump_json(
        source_url,
        flat_playlist=True,
        playlist_end=max_videos_per_source,
        retries=retries,
        backoff_sec=backoff_sec,
    )
    if not data:
        msg = f"[WARN] expand failed source={source_url} err={err or 'unknown'}"
        print(msg, file=sys.stderr, flush=True)
        if fail_on_error:
            raise RuntimeError(msg)
        return []

    entries = data.get("entries")
    if not isinstance(entries, list):
        msg = f"[WARN] expand returned no entries source={source_url}"
        print(msg, file=sys.stderr, flush=True)
        if fail_on_error:
            raise RuntimeError(msg)
        return []

    urls: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        webpage_url = entry.get("webpage_url")
        if isinstance(webpage_url, str) and webpage_url:
            urls.append(webpage_url)
        else:
            raw_url = entry.get("url")
            if isinstance(raw_url, str) and raw_url:
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw_url):
                    urls.append(f"https://www.youtube.com/watch?v={raw_url}")
                elif raw_url.startswith("http"):
                    urls.append(raw_url)
        if len(urls) >= max_videos_per_source:
            break

    deduped: List[str] = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def find_manual_sub_lang(info: Dict, prefer_lang: str) -> Optional[str]:
    subtitles = info.get("subtitles") or {}
    if not isinstance(subtitles, dict) or not subtitles:
        return None
    langs = list(subtitles.keys())
    if prefer_lang in langs:
        return prefer_lang
    lower_map = {lang.lower(): lang for lang in langs}
    if prefer_lang.lower() in lower_map:
        return lower_map[prefer_lang.lower()]
    for lang in langs:
        ll = lang.lower()
        if ll.startswith(prefer_lang.lower() + "-") or ll.startswith(prefer_lang.lower() + "_"):
            return lang
    return None


def find_subtitle_file(subtitle_dir: Path, video_id: str, lang: str) -> Optional[Path]:
    patterns = [
        f"{video_id}.{lang}*.vtt",
        f"{video_id}.{lang}*.srt",
        f"{video_id}*.{lang}*.vtt",
        f"{video_id}*.{lang}*.srt",
    ]
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(str(subtitle_dir / pattern)))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(matches[0])


def download_manual_subtitle(
    video_url: str,
    video_id: str,
    subtitle_dir: Path,
    subtitle_lang: str,
) -> Optional[Path]:
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(subtitle_dir / "%(id)s.%(ext)s")
    cmd = ytdlp_base_cmd() + [
        "--skip-download",
        "--write-subs",
        "--no-write-auto-subs",
        "--sub-langs",
        subtitle_lang,
        "--sub-format",
        "vtt/srt/best",
        "-o",
        out_tmpl,
        video_url,
    ]
    run_cmd(cmd, check=False, timeout_sec=120.0)
    return find_subtitle_file(subtitle_dir, video_id, subtitle_lang)


def ensure_full_audio(
    video_url: str,
    video_id: str,
    full_audio_dir: Path,
    sample_rate: int,
) -> Optional[Path]:
    full_audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = full_audio_dir / f"{video_id}.wav"
    if audio_path.exists():
        return audio_path

    out_tmpl = str(full_audio_dir / "%(id)s.%(ext)s")
    cmd = ytdlp_base_cmd() + [
        "-x",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        f"ffmpeg:-ar {sample_rate} -ac 1",
        "-o",
        out_tmpl,
        video_url,
    ]
    proc = run_cmd(cmd, check=False, timeout_sec=900.0)
    if proc.returncode != 0:
        return None
    if audio_path.exists():
        return audio_path
    candidates = sorted(full_audio_dir.glob(f"{video_id}.*"))
    for candidate in candidates:
        if candidate.suffix.lower() == ".wav":
            return candidate
    return None


def cut_clip(
    full_audio: Path,
    clip_out: Path,
    start: float,
    end: float,
    sample_rate: int,
) -> bool:
    clip_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(full_audio),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(clip_out),
    ]
    proc = run_cmd(cmd, check=False, timeout_sec=120.0)
    return proc.returncode == 0 and clip_out.exists()


def cleanup_full_audio(path: Optional[Path], keep_full_audio: bool) -> None:
    if keep_full_audio or path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[WARN] failed to delete full audio path={path} err={exc}", file=sys.stderr, flush=True)


def load_abbreviation_rules(rule_path: Path) -> Tuple[Dict[str, AbbreviationRule], Dict[str, AbbreviationRule]]:
    if not rule_path.exists():
        raise RuntimeError(f"Rule file not found: {rule_path}")

    delimiter = "\t" if rule_path.suffix.lower() != ".csv" else ","
    with rule_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = set(reader.fieldnames or [])
        required = {
            "abbr",
            "canonical_reading",
            "pron_type",
            "confidence_tier",
            "rule_id",
            "variants",
            "notes",
        }
        missing = required.difference(fieldnames)
        if missing:
            raise RuntimeError(f"Rule file missing columns: {', '.join(sorted(missing))}")

        by_abbr: Dict[str, AbbreviationRule] = {}
        by_variant: Dict[str, AbbreviationRule] = {}
        for idx, row in enumerate(reader, start=2):
            abbr = (row.get("abbr") or "").strip()
            canonical_reading = (row.get("canonical_reading") or "").strip()
            pron_type = (row.get("pron_type") or "").strip()
            confidence_tier = (row.get("confidence_tier") or "").strip()
            rule_id = (row.get("rule_id") or "").strip()
            variants_text = (row.get("variants") or "").strip()
            notes = (row.get("notes") or "").strip()

            if not abbr:
                raise RuntimeError(f"Rule file line {idx}: empty abbr")
            if not canonical_reading:
                raise RuntimeError(f"Rule file line {idx}: empty canonical_reading for {abbr}")
            if pron_type not in VALID_PRON_TYPES:
                raise RuntimeError(f"Rule file line {idx}: invalid pron_type={pron_type!r}")
            if confidence_tier not in VALID_CONFIDENCE_TIERS:
                raise RuntimeError(f"Rule file line {idx}: invalid confidence_tier={confidence_tier!r}")

            rule = AbbreviationRule(
                abbr=abbr,
                canonical_reading=canonical_reading,
                pron_type=pron_type,
                confidence_tier=confidence_tier,
                rule_id=rule_id or abbr,
                notes=notes,
            )
            if abbr in by_abbr:
                raise RuntimeError(f"Duplicate abbr in rule file: {abbr}")
            by_abbr[abbr] = rule

            variant_keys = [abbr]
            if variants_text:
                variant_keys.extend(v.strip() for v in variants_text.split(",") if v.strip())
            for variant in variant_keys:
                existing = by_variant.get(variant)
                if existing is not None:
                    if existing.rule_id == rule.rule_id:
                        continue
                    raise RuntimeError(f"Duplicate abbr/variant in rule file: {variant}")
                by_variant[variant] = rule
    return by_abbr, by_variant


def normalize_abbreviation(surface: str) -> str:
    token = surface.strip()
    if "-" in token:
        token = token.split("-", 1)[0]
    if len(token) > 2 and token.endswith("s"):
        base = token[:-1]
        if ABBR_PATTERN.fullmatch(base):
            token = base
    return token


def extract_abbreviation_candidates(text: str) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen = set()
    for match in ABBR_PATTERN.finditer(text):
        surface = match.group(0)
        normalized = normalize_abbreviation(surface)
        if not normalized:
            continue
        key = (match.start(), match.end(), normalized)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                surface=surface,
                start=match.start(),
                end=match.end(),
                normalized=normalized,
            )
        )
    return candidates


def match_abbreviation_rules(
    candidates: Sequence[Candidate],
    variant_rules: Dict[str, AbbreviationRule],
) -> Optional[List[Dict]]:
    matched: List[Dict] = []
    surface_to_rule: Dict[Tuple[int, int], str] = {}
    for candidate in candidates:
        rule = variant_rules.get(candidate.surface)
        if rule is None:
            rule = variant_rules.get(candidate.normalized)
        if rule is None:
            continue
        span_key = (candidate.start, candidate.end)
        prev_rule_id = surface_to_rule.get(span_key)
        if prev_rule_id is not None and prev_rule_id != rule.rule_id:
            return None
        surface_to_rule[span_key] = rule.rule_id
        matched.append(
            {
                "surface": candidate.surface,
                "abbr": rule.abbr,
                "canonical_reading": rule.canonical_reading,
                "pron_type": rule.pron_type,
                "confidence_tier": rule.confidence_tier,
                "rule_id": rule.rule_id,
                "span_start": candidate.start,
                "span_end": candidate.end,
            }
        )
    matched.sort(key=lambda item: (item["span_start"], item["span_end"]))
    return matched


def expand_text_with_readings(text: str, hits: Sequence[Dict]) -> Optional[str]:
    expanded = text
    for hit in sorted(hits, key=lambda item: item["span_start"], reverse=True):
        start = int(hit["span_start"])
        end = int(hit["span_end"])
        surface = hit["surface"]
        if expanded[start:end] != surface:
            return None
        expanded = expanded[:start] + hit["canonical_reading"] + expanded[end:]
    return normalize_caption_text(expanded)


def choose_primary_hit(hits: Sequence[Dict]) -> Dict:
    return sorted(
        hits,
        key=lambda item: (
            0 if item["confidence_tier"] == "high_conf" else 1,
            0 if item["pron_type"] != "spell_out" else 1,
            item["abbr"],
            item["span_start"],
        ),
    )[0]


def build_quota_plan(rule_map: Dict[str, AbbreviationRule], target_count: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    active_pron_types = sorted({rule.pron_type for rule in rule_map.values()})
    active_tiers = sorted({rule.confidence_tier for rule in rule_map.values()})
    pron_quota = {
        pron_type: max(1, target_count // max(1, len(active_pron_types)))
        for pron_type in active_pron_types
    }
    tier_quota = {
        tier: max(1, target_count // max(1, len(active_tiers)))
        for tier in active_tiers
    }
    return pron_quota, tier_quota


def quota_allows(
    hits: Sequence[Dict],
    abbr_counter: Counter,
    pron_counter: Counter,
    tier_counter: Counter,
    *,
    max_per_abbr: int,
    pron_quota: Dict[str, int],
    tier_quota: Dict[str, int],
) -> bool:
    hit_abbrs = {hit["abbr"] for hit in hits}
    for abbr in hit_abbrs:
        if abbr_counter[abbr] >= max_per_abbr:
            return False

    primary = choose_primary_hit(hits)
    pron_type = primary["pron_type"]
    confidence_tier = primary["confidence_tier"]

    pron_target = pron_quota.get(pron_type, 0)
    if pron_target > 0 and pron_counter[pron_type] >= pron_target:
        if any(pron_counter[key] < quota for key, quota in pron_quota.items() if quota > 0):
            return False

    tier_target = tier_quota.get(confidence_tier, 0)
    if tier_target > 0 and tier_counter[confidence_tier] >= tier_target:
        if any(tier_counter[key] < quota for key, quota in tier_quota.items() if quota > 0):
            return False

    return True


def score_segment_for_abbreviation(
    seg: Segment,
    variant_rules: Dict[str, AbbreviationRule],
    abbr_counter: Counter,
    pron_counter: Counter,
    tier_counter: Counter,
    *,
    max_per_abbr: int,
    max_abbrs_per_sentence: int,
    pron_quota: Dict[str, int],
    tier_quota: Dict[str, int],
) -> Optional[Dict]:
    candidates = extract_abbreviation_candidates(seg.text)
    if not candidates:
        return None

    hits = match_abbreviation_rules(candidates, variant_rules)
    if not hits:
        return None
    if len(hits) > max_abbrs_per_sentence:
        return None

    text_expanded = expand_text_with_readings(seg.text, hits)
    if not text_expanded:
        return None

    if not quota_allows(
        hits,
        abbr_counter,
        pron_counter,
        tier_counter,
        max_per_abbr=max_per_abbr,
        pron_quota=pron_quota,
        tier_quota=tier_quota,
    ):
        return None

    primary = choose_primary_hit(hits)
    return {
        "text_expanded": text_expanded,
        "abbreviations": hits,
        "primary_abbr": primary["abbr"],
        "primary_pron_type": primary["pron_type"],
        "primary_confidence_tier": primary["confidence_tier"],
    }


def manifest_path(path: Path, out_dir: Path, path_mode: str) -> str:
    resolved = path.resolve()
    if path_mode == "relative":
        return str(os.path.relpath(resolved, out_dir.resolve()))
    return str(resolved)


def dedup_key_for_segment(seg: Segment, video_id: str, dedup_key: str) -> Optional[str]:
    if dedup_key == "none":
        return None
    if dedup_key == "segment":
        return f"{video_id}:{seg.start:.3f}:{seg.end:.3f}"
    return seg.text.strip().lower()


def validate_dependencies(download_audio: bool) -> None:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not found in PATH. Please install yt-dlp first.")
    if download_audio and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg or disable --download-audio.")


def save_manifest(records: List[Dict], out_json: Path, out_jsonl: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_rule_audit(rows: List[Dict], audit_jsonl: Path) -> None:
    audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with audit_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_sample_review(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "uid",
        "abbr",
        "surface",
        "canonical_reading",
        "pron_type",
        "confidence_tier",
        "text",
        "audio_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
    tier_counter: Counter,
) -> Dict:
    stats = {
        "target_count": target_count,
        "final_count": final_count,
        "inspected_videos": inspected_videos,
        "kept_videos": kept_videos,
        "source_expand_failures": source_expand_failures,
        "video_info_failures": video_info_failures,
        "abbr_counter": dict(sorted(abbr_counter.items())),
        "pron_type_counter": dict(sorted(pron_counter.items())),
        "confidence_tier_counter": dict(sorted(tier_counter.items())),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def restore_resume_state(
    records: List[Dict],
    existing_video_segments: Dict[str, int],
    abbr_counter: Counter,
    pron_counter: Counter,
    tier_counter: Counter,
) -> int:
    uid = 0
    for record in records:
        uid = max(uid, int(record.get("uid", 0) or 0))
        video_id = str(record.get("video_id") or "")
        if video_id:
            existing_video_segments[video_id] += 1
        seen_abbrs = set()
        seen_pron = set()
        seen_tiers = set()
        for hit in record.get("abbreviations") or []:
            if not isinstance(hit, dict):
                continue
            abbr = str(hit.get("abbr") or "")
            pron_type = str(hit.get("pron_type") or "")
            confidence_tier = str(hit.get("confidence_tier") or "")
            if abbr and abbr not in seen_abbrs:
                abbr_counter[abbr] += 1
                seen_abbrs.add(abbr)
            if pron_type and pron_type not in seen_pron:
                pron_counter[pron_type] += 1
                seen_pron.add(pron_type)
            if confidence_tier and confidence_tier not in seen_tiers:
                tier_counter[confidence_tier] += 1
                seen_tiers.add(confidence_tier)
    return uid


def build_manifest(args: argparse.Namespace) -> int:
    global CMD_TIMEOUT_SEC, YTDLP_PROXY, YTDLP_RETRIES, YTDLP_RETRY_BACKOFF_SEC
    CMD_TIMEOUT_SEC = float(args.cmd_timeout)
    YTDLP_PROXY = (args.proxy or "").strip()
    YTDLP_RETRIES = max(0, int(args.expand_retries))
    YTDLP_RETRY_BACKOFF_SEC = max(0.0, float(args.expand_backoff))
    if YTDLP_PROXY:
        print(f"[INFO] yt-dlp proxy={YTDLP_PROXY}", file=sys.stderr, flush=True)

    validate_dependencies(download_audio=args.download_audio)

    sources = read_seed_sources(Path(args.seed_file))
    if not sources:
        raise RuntimeError(f"No valid sources found in {args.seed_file}")
    rules_by_abbr, rules_by_variant = load_abbreviation_rules(Path(args.rule_file))
    pron_quota, tier_quota = build_quota_plan(rules_by_abbr, args.target_count)

    out_dir = Path(args.out_dir)
    subtitle_dir = out_dir / "subtitles"
    full_audio_dir = out_dir / "full_audio"
    clips_dir = out_dir / "clips_wav"
    out_json = out_dir / "manifest.json"
    out_jsonl = out_dir / "manifest.jsonl"
    audit_jsonl = out_dir / "rule_audit.jsonl"
    review_tsv = out_dir / "sample_review.tsv"
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
    tier_counter: Counter = Counter()

    if args.resume and out_json.exists():
        try:
            loaded = json.loads(out_json.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = loaded
        except Exception:
            records = []
        uid = restore_resume_state(records, existing_video_segments, abbr_counter, pron_counter, tier_counter)
        for record in records:
            if args.dedup_key == "text":
                text = record.get("text")
                if isinstance(text, str):
                    seen_dedup_keys.add(text.strip().lower())
            elif args.dedup_key == "segment":
                video_id = str(record.get("video_id") or "")
                start = record.get("start")
                end = record.get("end")
                if video_id and isinstance(start, (float, int)) and isinstance(end, (float, int)):
                    seen_dedup_keys.add(f"{video_id}:{float(start):.3f}:{float(end):.3f}")
        if audit_jsonl.exists():
            with audit_jsonl.open("r", encoding="utf-8") as f:
                audit_rows = [json.loads(line) for line in f if line.strip()]
        if review_tsv.exists():
            with review_tsv.open("r", encoding="utf-8", newline="") as f:
                review_rows = list(csv.DictReader(f, delimiter="\t"))
        print(f"[INFO] resume loaded records={len(records)}", file=sys.stderr, flush=True)

    if uid >= args.target_count:
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
            tier_counter=tier_counter,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    for source in sources:
        if uid >= args.target_count:
            break
        print(f"[INFO] source={source.url}", file=sys.stderr, flush=True)
        video_urls = expand_video_urls(
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
            info, info_err = ytdlp_dump_json(video_url, flat_playlist=False, playlist_end=None)
            if not info:
                video_info_failures += 1
                if info_err:
                    print(f"[WARN] video info failed url={video_url} err={info_err}", file=sys.stderr, flush=True)
                continue

            subtitle_lang = find_manual_sub_lang(info, args.lang)
            if not subtitle_lang:
                continue
            video_id = str(info.get("id") or "")
            if not video_id:
                continue
            if existing_video_segments[video_id] >= args.max_segments_per_video:
                continue

            subtitle_path = download_manual_subtitle(video_url, video_id, subtitle_dir, subtitle_lang)
            if subtitle_path is None or not subtitle_path.exists():
                continue
            cues = parse_subtitle_file(subtitle_path)
            if not cues:
                continue
            segments = cues_to_segments(
                cues,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                max_gap=args.max_gap,
            )
            if not segments:
                continue

            full_audio_path: Optional[Path] = None
            if args.download_audio:
                full_audio_path = ensure_full_audio(
                    video_url=video_url,
                    video_id=video_id,
                    full_audio_dir=full_audio_dir,
                    sample_rate=args.sample_rate,
                )
                if full_audio_path is None:
                    continue

            added_this_video = 0
            segments.sort(key=lambda seg: (len(extract_abbreviation_candidates(seg.text)), seg.start))
            for seg in segments:
                if uid >= args.target_count:
                    break
                if added_this_video >= args.max_segments_per_video:
                    break
                if existing_video_segments[video_id] >= args.max_segments_per_video:
                    break

                dedup_key = dedup_key_for_segment(seg, video_id, args.dedup_key)
                if dedup_key is not None and dedup_key in seen_dedup_keys:
                    continue

                score = score_segment_for_abbreviation(
                    seg,
                    rules_by_variant,
                    abbr_counter,
                    pron_counter,
                    tier_counter,
                    max_per_abbr=args.max_per_abbr,
                    max_abbrs_per_sentence=args.max_abbrs_per_sentence,
                    pron_quota=pron_quota,
                    tier_quota=tier_quota,
                )
                if score is None:
                    continue

                clip_name = f"{video_id}_{int(seg.start * 1000):09d}_{int(seg.end * 1000):09d}.wav"
                clip_path = clips_dir / clip_name
                if args.download_audio:
                    assert full_audio_path is not None
                    ok = cut_clip(
                        full_audio=full_audio_path,
                        clip_out=clip_path,
                        start=seg.start,
                        end=seg.end,
                        sample_rate=args.sample_rate,
                    )
                    if not ok:
                        continue

                uid += 1
                audio_manifest_path = manifest_path(clip_path, out_dir, args.path_mode) if args.download_audio else ""
                record = {
                    "uid": uid,
                    "URL": str(info.get("webpage_url") or video_url),
                    "video_id": video_id,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "audio_path": audio_manifest_path,
                    "subtitle_path": manifest_path(subtitle_path, out_dir, args.path_mode),
                    "text": seg.text,
                    "text_expanded": score["text_expanded"],
                    "abbreviations": score["abbreviations"],
                    "primary_abbr": score["primary_abbr"],
                    "primary_pron_type": score["primary_pron_type"],
                    "primary_confidence_tier": score["primary_confidence_tier"],
                    "source_domain": source.domain,
                    "source_note": source.note,
                }
                records.append(record)

                seen_abbrs = set()
                seen_pron = set()
                seen_tiers = set()
                for hit in score["abbreviations"]:
                    if hit["abbr"] not in seen_abbrs:
                        abbr_counter[hit["abbr"]] += 1
                        seen_abbrs.add(hit["abbr"])
                    if hit["pron_type"] not in seen_pron:
                        pron_counter[hit["pron_type"]] += 1
                        seen_pron.add(hit["pron_type"])
                    if hit["confidence_tier"] not in seen_tiers:
                        tier_counter[hit["confidence_tier"]] += 1
                        seen_tiers.add(hit["confidence_tier"])

                    audit_rows.append(
                        {
                            "uid": uid,
                            "video_id": video_id,
                            "surface": hit["surface"],
                            "abbr": hit["abbr"],
                            "canonical_reading": hit["canonical_reading"],
                            "pron_type": hit["pron_type"],
                            "confidence_tier": hit["confidence_tier"],
                            "rule_id": hit["rule_id"],
                            "context_text": seg.text,
                        }
                    )
                    review_rows.append(
                        {
                            "uid": uid,
                            "abbr": hit["abbr"],
                            "surface": hit["surface"],
                            "canonical_reading": hit["canonical_reading"],
                            "pron_type": hit["pron_type"],
                            "confidence_tier": hit["confidence_tier"],
                            "text": seg.text,
                            "audio_path": audio_manifest_path,
                        }
                    )

                if dedup_key is not None:
                    seen_dedup_keys.add(dedup_key)
                added_this_video += 1
                existing_video_segments[video_id] += 1

                if args.save_every > 0 and uid % args.save_every == 0:
                    save_manifest(records, out_json, out_jsonl)
                    save_rule_audit(audit_rows, audit_jsonl)
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
                        tier_counter=tier_counter,
                    )
                    print(f"[INFO] checkpoint saved records={len(records)}", file=sys.stderr, flush=True)

            if added_this_video > 0:
                kept_videos += 1
                print(
                    f"[INFO] kept video={video_url} segments={added_this_video} total={uid}",
                    file=sys.stderr,
                    flush=True,
                )
            cleanup_full_audio(full_audio_path, keep_full_audio=args.keep_full_audio)

    save_manifest(records, out_json, out_jsonl)
    save_rule_audit(audit_rows, audit_jsonl)
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
        tier_counter=tier_counter,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine abbreviation-focused science video segments.")
    parser.add_argument("--seed-file", required=True, help="TSV source list: domain<TAB>url<TAB>note")
    parser.add_argument("--rule-file", required=True, help="TSV/CSV abbreviation pronunciation table")
    parser.add_argument("--out-dir", required=True, help="Output directory")

    parser.add_argument("--target-count", type=int, default=2000)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-videos-per-source", type=int, default=400)
    parser.add_argument("--max-segments-per-video", type=int, default=10)
    parser.add_argument("--max-per-abbr", type=int, default=3)
    parser.add_argument("--max-abbrs-per-sentence", type=int, default=2)

    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-gap", type=float, default=1.2)

    parser.add_argument("--download-audio", action="store_true", default=False)
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[16000, 22050, 24000])
    parser.add_argument("--keep-full-audio", action="store_true", default=False)
    parser.add_argument("--cmd-timeout", type=float, default=45.0)
    parser.add_argument("--proxy", default="", help="Proxy passed to yt-dlp")
    parser.add_argument("--expand-retries", type=int, default=2)
    parser.add_argument("--expand-backoff", type=float, default=2.0)
    parser.add_argument("--fail-on-source-expand-error", action="store_true", default=False)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--dedup-key", choices=["text", "segment", "none"], default="text")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return build_manifest(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
