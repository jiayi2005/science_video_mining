#!/usr/bin/env python3
"""
Build a science-focused evaluation manifest from online videos with manual subtitles.

Core constraints handled by this script:
1) Real speech data from online videos (no synthetic TTS).
2) Prefer strong labels by keeping only manual subtitles (not auto captions).
3) Domain diversity (biology, medicine, chemistry, physics, geography/geology).
4) Rare word control against GigaSpeech vocab:
   - Prefer words with freq < 10 (vocab_min_freq_1 minus vocab_min_freq_10)
   - Also keep OOV words (not in vocab_min_freq_1)
   - Each sentence keeps at most 3 such rare words.
"""

from __future__ import annotations

import argparse
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
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


WORD_RE = re.compile(r"\b[a-zA-Z']+\b")
TIMESPAN_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{2,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{2,3})"
)
SENT_END_RE = re.compile(r"[.!?。！？]$")
TAG_RE = re.compile(r"<[^>]+>")
MUSIC_CUE_RE = re.compile(r"^\[(music|applause|laughter|inaudible).*\]$", re.IGNORECASE)
ENTITY_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9-]*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")


SCIENCE_KEYWORDS: Dict[str, set] = {
    "biology": {
        "biology",
        "cell",
        "cells",
        "gene",
        "genes",
        "dna",
        "rna",
        "protein",
        "genome",
        "species",
        "evolution",
        "bacteria",
        "virus",
        "ecology",
        "neuron",
    },
    "medicine": {
        "medicine",
        "medical",
        "patient",
        "patients",
        "clinical",
        "diagnosis",
        "disease",
        "therapy",
        "treatment",
        "vaccine",
        "hospital",
        "cancer",
        "drug",
        "surgery",
        "trial",
    },
    "chemistry": {
        "chemistry",
        "chemical",
        "molecule",
        "molecules",
        "reaction",
        "catalyst",
        "polymer",
        "compound",
        "acid",
        "base",
        "solvent",
        "isotope",
        "synthesis",
        "covalent",
    },
    "physics": {
        "physics",
        "quantum",
        "photon",
        "electron",
        "neutron",
        "gravity",
        "relativity",
        "particle",
        "plasma",
        "wave",
        "thermodynamics",
        "force",
        "energy",
    },
    "geography": {
        "geography",
        "geology",
        "tectonic",
        "climate",
        "glacier",
        "ocean",
        "earthquake",
        "volcano",
        "sediment",
        "latitude",
        "longitude",
        "atmosphere",
        "continent",
        "hydrology",
    },
}


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


def load_vocab_set(vocab_path: str) -> set:
    vocab_set = set()
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    vocab_set.add(parts[0].lower())
    return vocab_set


CMD_TIMEOUT_SEC = 45.0
YTDLP_PROXY = ""
YTDLP_RETRIES = 2
YTDLP_RETRY_BACKOFF_SEC = 2.0


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


def summarize_stderr(stderr: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", (stderr or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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

        current_duration = cur_end - cur_start
        if current_duration >= max_duration or SENT_END_RE.search(cue.text):
            flush()

    flush()
    return segments


def tokenize(text: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)]


def lowercase_word_tokens(tokens: Sequence[str]) -> List[str]:
    # Restrict vocab/OOV scoring to fully lowercase words to avoid CamelCase name drift.
    return [t.lower() for t in tokens if t.islower()]


def detect_domains(tokens: Sequence[str]) -> List[str]:
    token_set = set(tokens)
    hits: List[Tuple[str, int]] = []
    for domain, kws in SCIENCE_KEYWORDS.items():
        score = len(token_set.intersection(kws))
        if score > 0:
            hits.append((domain, score))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in hits]


def extract_entities(text: str) -> List[str]:
    entities = []
    for m in ENTITY_RE.finditer(text):
        ent = m.group(0).strip()
        if len(ent) < 2:
            continue
        if ent.lower() in {"i", "we", "the"}:
            continue
        entities.append(ent)
    dedup = []
    seen = set()
    for ent in entities:
        key = ent.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(ent)
    return dedup


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
    retries = max(0, int(retries))
    backoff_sec = max(0.0, float(backoff_sec))

    cmd = ytdlp_base_cmd() + ["--dump-single-json", "--skip-download"]
    if flat_playlist:
        cmd.append("--flat-playlist")
    if playlist_end is not None and playlist_end > 0:
        cmd.extend(["--playlist-end", str(playlist_end)])
    cmd.append(url)

    attempts = retries + 1
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
            time.sleep(backoff_sec * (attempt + 1))
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
    # Direct video/talk URL: keep as-is.
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
    for e in entries:
        if not isinstance(e, dict):
            continue
        webpage_url = e.get("webpage_url")
        if isinstance(webpage_url, str) and webpage_url:
            urls.append(webpage_url)
            continue
        url = e.get("url")
        if isinstance(url, str) and url:
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
                urls.append(f"https://www.youtube.com/watch?v={url}")
            elif url.startswith("http"):
                urls.append(url)
        if len(urls) >= max_videos_per_source:
            break

    if not urls:
        msg = f"[WARN] expand got empty url list source={source_url}"
        print(msg, file=sys.stderr, flush=True)
        if fail_on_error:
            raise RuntimeError(msg)
        return []
    dedup: List[str] = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def find_subtitle_file(subtitle_dir: Path, video_id: str, lang: str) -> Optional[Path]:
    patterns = [
        f"{video_id}.{lang}*.vtt",
        f"{video_id}.{lang}*.srt",
        f"{video_id}*.{lang}*.vtt",
        f"{video_id}*.{lang}*.srt",
    ]
    matches: List[str] = []
    for p in patterns:
        matches.extend(glob.glob(str(subtitle_dir / p)))
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
    # Some extractors may keep original ext name. Fall back to glob.
    candidates = sorted(full_audio_dir.glob(f"{video_id}.*"))
    for c in candidates:
        if c.suffix.lower() == ".wav":
            return c
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


def score_segment(
    seg: Segment,
    vocab_min1: set,
    vocab_min10: set,
    max_rare: int,
    require_science_keyword: bool,
    require_entity: bool,
) -> Optional[Dict]:
    raw_tokens = tokenize(seg.text)
    if not raw_tokens:
        return None

    tokens = [t.lower() for t in raw_tokens]
    vocab_tokens = lowercase_word_tokens(raw_tokens)

    freq_1_9 = sorted({w for w in vocab_tokens if (w in vocab_min1 and w not in vocab_min10)})
    oov = sorted({w for w in vocab_tokens if w not in vocab_min1})
    rare_union = sorted(set(freq_1_9).union(oov))
    if not rare_union:
        return None
    if len(rare_union) > max_rare:
        return None

    domains = detect_domains(tokens)
    if require_science_keyword and not domains:
        return None

    entities = extract_entities(seg.text)
    if require_entity and not entities:
        return None

    return {
        "tokens": tokens,
        "domains": domains,
        "freq_1_9_words": freq_1_9,
        "oov_words": oov,
        "rare_words": rare_union,
        "entities": entities,
    }


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
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_stats(
    out_dir: Path,
    target_count: int,
    final_count: int,
    inspected_videos: int,
    kept_videos: int,
    domain_counter: Dict[str, int],
    source_expand_failures: int,
    video_info_failures: int,
) -> Dict:
    stats = {
        "target_count": target_count,
        "final_count": final_count,
        "inspected_videos": inspected_videos,
        "kept_videos": kept_videos,
        "source_expand_failures": source_expand_failures,
        "video_info_failures": video_info_failures,
        "domain_counter": dict(domain_counter),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


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


def build_manifest(args: argparse.Namespace) -> int:
    global CMD_TIMEOUT_SEC, YTDLP_PROXY, YTDLP_RETRIES, YTDLP_RETRY_BACKOFF_SEC
    CMD_TIMEOUT_SEC = float(args.cmd_timeout)
    YTDLP_PROXY = (args.proxy or "").strip()
    YTDLP_RETRIES = max(0, int(args.expand_retries))
    YTDLP_RETRY_BACKOFF_SEC = max(0.0, float(args.expand_backoff))
    if YTDLP_PROXY:
        print(f"[INFO] yt-dlp proxy={YTDLP_PROXY}", file=sys.stderr, flush=True)

    validate_dependencies(download_audio=args.download_audio)

    vocab_min1 = load_vocab_set(args.vocab_min1)
    vocab_min10 = load_vocab_set(args.vocab_min10)
    if not vocab_min1 or not vocab_min10:
        raise RuntimeError("Failed to load vocab files. Please verify --vocab-min1 and --vocab-min10.")

    out_dir = Path(args.out_dir)
    subtitle_dir = out_dir / "subtitles"
    full_audio_dir = out_dir / "full_audio"
    clips_dir = out_dir / "clips_wav"
    out_json = out_dir / "manifest.json"
    out_jsonl = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = read_seed_sources(Path(args.seed_file))
    if not sources:
        raise RuntimeError(f"No valid sources found in {args.seed_file}")

    records: List[Dict] = []
    uid = 0
    domain_counter: Dict[str, int] = defaultdict(int)
    inspected_videos = 0
    kept_videos = 0
    source_expand_failures = 0
    video_info_failures = 0
    seen_dedup_keys = set()
    existing_video_segments: Dict[str, int] = defaultdict(int)

    if args.resume and out_json.exists():
        try:
            loaded = json.loads(out_json.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = loaded
        except Exception:
            records = []

        if records:
            uid = max(int(r.get("uid", 0) or 0) for r in records)
            for r in records:
                d = r.get("domain")
                if isinstance(d, str) and d:
                    domain_counter[d] += 1
                vid = str(r.get("video_id") or "")
                if vid:
                    existing_video_segments[vid] += 1

                if args.dedup_key == "text":
                    t = r.get("text")
                    if isinstance(t, str):
                        seen_dedup_keys.add(t.strip().lower())
                elif args.dedup_key == "segment":
                    s = r.get("start")
                    e = r.get("end")
                    if vid and isinstance(s, (int, float)) and isinstance(e, (int, float)):
                        seen_dedup_keys.add(f"{vid}:{float(s):.3f}:{float(e):.3f}")

            print(f"[INFO] resume loaded records={len(records)}", file=sys.stderr, flush=True)

        stats_path = out_dir / "stats.json"
        if stats_path.exists():
            try:
                prev_stats = json.loads(stats_path.read_text(encoding="utf-8"))
                inspected_videos = int(prev_stats.get("inspected_videos", 0) or 0)
                kept_videos = int(prev_stats.get("kept_videos", 0) or 0)
                source_expand_failures = int(prev_stats.get("source_expand_failures", 0) or 0)
                video_info_failures = int(prev_stats.get("video_info_failures", 0) or 0)
            except Exception:
                inspected_videos = 0
                kept_videos = 0
                source_expand_failures = 0
                video_info_failures = 0

    if uid >= args.target_count:
        stats = write_stats(
            out_dir=out_dir,
            target_count=args.target_count,
            final_count=len(records),
            inspected_videos=inspected_videos,
            kept_videos=kept_videos,
            domain_counter=domain_counter,
            source_expand_failures=source_expand_failures,
            video_info_failures=video_info_failures,
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
                    print(
                        f"[WARN] video info failed url={video_url} err={info_err}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            subtitle_lang = find_manual_sub_lang(info, args.lang)
            if not subtitle_lang:
                continue
            video_id = str(info.get("id") or "")
            if not video_id:
                continue

            # Resume optimization: skip videos that already reached segment quota.
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

                score = score_segment(
                    seg=seg,
                    vocab_min1=vocab_min1,
                    vocab_min10=vocab_min10,
                    max_rare=args.max_rare_per_sentence,
                    require_science_keyword=args.require_science_keyword,
                    require_entity=args.require_entity,
                )
                if score is None:
                    continue

                if source.domain and source.domain not in {"unknown", "multi"}:
                    domain = source.domain
                elif score["domains"]:
                    domain = score["domains"][0]
                else:
                    domain = source.domain or "unknown"
                if args.max_per_domain > 0 and domain_counter[domain] >= args.max_per_domain:
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
                record = {
                    "uid": uid,
                    "URL": str(info.get("webpage_url") or video_url),
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text,
                    "audio_path": manifest_path(clip_path, out_dir, args.path_mode),
                    "video_id": video_id,
                    "subtitle_lang": subtitle_lang,
                    "domain": domain,
                    "rare_words": score["rare_words"],
                    "freq_1_9_words": score["freq_1_9_words"],
                    "oov_words": score["oov_words"],
                    "entities": score["entities"],
                    "subtitle_path": manifest_path(subtitle_path, out_dir, args.path_mode),
                    "source_domain": source.domain,
                    "source_note": source.note,
                }
                records.append(record)
                if dedup_key is not None:
                    seen_dedup_keys.add(dedup_key)
                domain_counter[domain] += 1
                added_this_video += 1
                existing_video_segments[video_id] += 1

                if args.save_every > 0 and uid % args.save_every == 0:
                    save_manifest(records, out_json=out_json, out_jsonl=out_jsonl)
                    write_stats(
                        out_dir=out_dir,
                        target_count=args.target_count,
                        final_count=len(records),
                        inspected_videos=inspected_videos,
                        kept_videos=kept_videos,
                        domain_counter=domain_counter,
                        source_expand_failures=source_expand_failures,
                        video_info_failures=video_info_failures,
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

    save_manifest(records, out_json=out_json, out_jsonl=out_jsonl)
    stats = write_stats(
        out_dir=out_dir,
        target_count=args.target_count,
        final_count=len(records),
        inspected_videos=inspected_videos,
        kept_videos=kept_videos,
        domain_counter=domain_counter,
        source_expand_failures=source_expand_failures,
        video_info_failures=video_info_failures,
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine science video segments with manual subtitles and low-frequency terms."
    )
    parser.add_argument("--seed-file", required=True, help="TSV source list: domain<TAB>url<TAB>note")
    parser.add_argument("--vocab-min1", required=True, help="GigaSpeech vocab with freq >= 1")
    parser.add_argument("--vocab-min10", required=True, help="GigaSpeech vocab with freq >= 10")
    parser.add_argument("--out-dir", required=True, help="Output directory")

    parser.add_argument("--target-count", type=int, default=10000, help="Target number of segments")
    parser.add_argument("--lang", default="en", help="Preferred subtitle language tag prefix")
    parser.add_argument("--max-videos-per-source", type=int, default=400)
    parser.add_argument("--max-segments-per-video", type=int, default=30)

    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-gap", type=float, default=1.2, help="Max silence gap when merging cues")
    parser.add_argument("--max-rare-per-sentence", type=int, default=3)

    parser.add_argument("--require-science-keyword", action="store_true", default=True)
    parser.add_argument("--no-require-science-keyword", dest="require_science_keyword", action="store_false")
    parser.add_argument("--require-entity", action="store_true", default=False)
    parser.add_argument("--max-per-domain", type=int, default=0, help="0 = unlimited")

    parser.add_argument("--download-audio", action="store_true", default=False)
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[16000, 22050, 24000])
    parser.add_argument(
        "--keep-full-audio",
        action="store_true",
        default=False,
        help="Keep downloaded full_audio wav files after clip extraction",
    )
    parser.add_argument("--cmd-timeout", type=float, default=45.0, help="Per external command timeout seconds")
    parser.add_argument("--proxy", default="", help="Proxy passed to yt-dlp, e.g. socks5://127.0.0.1:1080")
    parser.add_argument("--expand-retries", type=int, default=2, help="Retries for yt-dlp source/video metadata fetch")
    parser.add_argument("--expand-backoff", type=float, default=2.0, help="Retry backoff seconds for yt-dlp fetch")
    parser.add_argument(
        "--fail-on-source-expand-error",
        action="store_true",
        default=False,
        help="Fail fast when non-direct source URL cannot expand into video URLs",
    )
    parser.add_argument("--save-every", type=int, default=200, help="Checkpoint save frequency by kept records")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from existing manifest.json")
    parser.add_argument(
        "--path-mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="How to store audio_path/subtitle_path in manifest",
    )
    parser.add_argument(
        "--dedup-key",
        choices=["text", "segment", "none"],
        default="text",
        help="Dedup strategy across segments",
    )
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
