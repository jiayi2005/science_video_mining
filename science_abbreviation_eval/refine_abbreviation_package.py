#!/usr/bin/env python3
import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path


def load_denylist(path: Path) -> set[str]:
    vals = set()
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals.add(line)
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-metadata', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--denylist-file', required=True)
    parser.add_argument('--zip-name', default='audio.zip')
    args = parser.parse_args()

    input_metadata = Path(args.input_metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    denylist = load_denylist(Path(args.denylist_file))

    meta = json.loads(input_metadata.read_text(encoding='utf-8'))
    items = meta['items']

    kept = []
    removed = []
    for item in items:
        abbr = (item.get('abbr') or '').strip()
        if abbr in denylist:
            removed.append(item)
        else:
            kept.append(item)

    for idx, item in enumerate(kept, start=1):
        old_audio = Path(item['audio_file']).name
        new_audio = f"audio/{idx:05d}_{Path(old_audio).name}"
        item['id'] = idx
        item['audio_file'] = new_audio

    zip_path = output_dir / args.zip_name
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for item in kept:
            src = item['original_audio_path']
            zf.write(src, arcname=item['audio_file'])

    out_meta = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z').strip(),
        'source_metadata': str(input_metadata),
        'num_items': len(kept),
        'denylist_file': str(Path(args.denylist_file)),
        'removed_abbr_count': len({x['abbr'] for x in removed}),
        'items': kept,
    }
    (output_dir / 'metadata.json').write_text(json.dumps(out_meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    report = {
        'input_items': len(items),
        'kept_items': len(kept),
        'removed_items': len(removed),
        'removed_unique_abbr': len({x['abbr'] for x in removed}),
        'removed_examples': sorted({x['abbr'] for x in removed})[:100],
        'zip_path': str(zip_path),
        'metadata_path': str(output_dir / 'metadata.json'),
    }
    (output_dir / 'selection_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
