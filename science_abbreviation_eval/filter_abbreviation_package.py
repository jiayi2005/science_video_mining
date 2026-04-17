#!/usr/bin/env python3
import argparse
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_FORBIDDEN_CHARS = ":[]"
EXTRA_BAD_PUNCT_RE = re.compile(r"[\[\]\{\}\|<>]")
SPACE_RE = re.compile(r"\s+")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def clean_text(text: str, forbidden_chars: str) -> str:
    if not text:
        return ""
    trans = str.maketrans("", "", forbidden_chars)
    text = text.translate(trans)
    text = SPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return text


def count_extra_bad_punct(text: str) -> int:
    return len(EXTRA_BAD_PUNCT_RE.findall(text or ""))


def safe_name(value: str) -> str:
    value = SAFE_NAME_RE.sub("_", value or "").strip("._-")
    return value or "item"


def primary_hit(record: Dict) -> Dict:
    primary_abbr = record.get("primary_abbr", "")
    hits = record.get("abbreviations", []) or []
    for hit in hits:
        if hit.get("abbr") == primary_abbr:
            return hit
    return hits[0] if hits else {}


def record_score(record: Dict, forbidden_chars: str) -> Tuple:
    text = record.get("text", "")
    text_expanded = record.get("text_expanded", "")
    hits = record.get("abbreviations", []) or []
    clean = clean_text(text, forbidden_chars)
    clean_expanded = clean_text(text_expanded, forbidden_chars)
    audio_missing = not Path(record.get("audio_path", "")).exists()
    has_forbidden = clean != text or clean_expanded != text_expanded
    multi_abbr = len(hits) != 1
    extra_punct = count_extra_bad_punct(text) + count_extra_bad_punct(text_expanded)
    text_len = len(clean)
    word_count = len(clean.split())
    uid = int(record.get("uid", 10**9))
    return (
        audio_missing,
        has_forbidden,
        multi_abbr,
        extra_punct,
        abs(word_count - 12),
        text_len,
        uid,
    )


def load_records(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def choose_records(records: List[Dict], forbidden_chars: str) -> Tuple[List[Dict], Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    skipped_missing_primary = 0
    skipped_missing_audio = 0
    for record in records:
        primary_abbr = (record.get("primary_abbr") or "").strip()
        if not primary_abbr:
            skipped_missing_primary += 1
            continue
        if not Path(record.get("audio_path", "")).exists():
            skipped_missing_audio += 1
            continue
        grouped[primary_abbr].append(record)

    selected: List[Dict] = []
    score_stats = {
        "groups": len(grouped),
        "skipped_missing_primary": skipped_missing_primary,
        "skipped_missing_audio": skipped_missing_audio,
        "with_forbidden_available": 0,
        "without_forbidden_selected": 0,
    }

    for _, candidates in grouped.items():
        candidates = sorted(candidates, key=lambda rec: record_score(rec, forbidden_chars))
        best = candidates[0]
        selected.append(best)
        if any(clean_text(rec.get("text", ""), forbidden_chars) != rec.get("text", "") for rec in candidates):
            score_stats["with_forbidden_available"] += 1
        if clean_text(best.get("text", ""), forbidden_chars) == best.get("text", ""):
            score_stats["without_forbidden_selected"] += 1

    selected.sort(key=lambda rec: ((rec.get("primary_abbr") or ""), int(rec.get("uid", 10**9))))
    return selected, score_stats


def build_metadata(selected: List[Dict], forbidden_chars: str) -> List[Dict]:
    items = []
    for index, record in enumerate(selected, start=1):
        hit = primary_hit(record)
        abbr = record.get("primary_abbr") or hit.get("abbr", "")
        surface = hit.get("surface", abbr)
        audio_path = Path(record["audio_path"])
        arcname = f"audio/{index:05d}_{safe_name(abbr)}_{safe_name(record.get('video_id', 'video'))}.wav"
        items.append(
            {
                "id": index,
                "abbr": abbr,
                "surface": surface,
                "text": clean_text(record.get("text", ""), forbidden_chars),
                "text_expanded": clean_text(record.get("text_expanded", ""), forbidden_chars),
                "audio_file": arcname,
                "original_uid": record.get("uid"),
                "video_id": record.get("video_id"),
                "URL": record.get("URL"),
                "start": record.get("start"),
                "end": record.get("end"),
                "source_domain": record.get("source_domain"),
                "source_note": record.get("source_note"),
                "primary_pron_type": record.get("primary_pron_type"),
                "primary_confidence_tier": record.get("primary_confidence_tier"),
                "match_mode": record.get("match_mode"),
                "abbreviation": {
                    "abbr": hit.get("abbr", abbr),
                    "surface": surface,
                    "span_start": hit.get("span_start"),
                    "span_end": hit.get("span_end"),
                    "pron_type": hit.get("pron_type"),
                    "confidence_tier": hit.get("confidence_tier"),
                    "canonical_reading": hit.get("canonical_reading", ""),
                    "rule_id": hit.get("rule_id", ""),
                    "is_rule_confirmed": hit.get("is_rule_confirmed", False),
                },
                "original_audio_path": str(audio_path),
            }
        )
    return items


def write_zip(selected: List[Dict], metadata_items: List[Dict], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for record, item in zip(selected, metadata_items):
            zf.write(record["audio_path"], arcname=item["audio_file"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--forbidden-chars", default=DEFAULT_FORBIDDEN_CHARS)
    parser.add_argument("--zip-name", default="audio.zip")
    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_manifest)
    selected, score_stats = choose_records(records, args.forbidden_chars)
    metadata_items = build_metadata(selected, args.forbidden_chars)

    zip_path = output_dir / args.zip_name
    metadata_path = output_dir / "metadata.json"
    report_path = output_dir / "selection_report.json"

    write_zip(selected, metadata_items, zip_path)

    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip(),
        "input_manifest": str(input_manifest),
        "num_items": len(metadata_items),
        "forbidden_chars_removed": args.forbidden_chars,
        "items": metadata_items,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "input_records": len(records),
        "selected_records": len(selected),
        "unique_primary_abbr": len({item["abbr"] for item in metadata_items}),
        "score_stats": score_stats,
        "zip_path": str(zip_path),
        "metadata_path": str(metadata_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
