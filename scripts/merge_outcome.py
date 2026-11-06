#!/usr/bin/env python3
"""
Merge the outcome of evaluate_embeddings.py from multiple directories.

Each input directory is expected to contain
embedding_evaluation_summary.csv and/or embedding_evaluation_summary.json
(written by evaluate_embeddings.py --outdir <dir>).

Directories may be given as glob patterns (e.g. 'exp_*', 'results/**/eval');
quote them to let this script expand them instead of the shell.

Examples:
  python scripts/merge_outcome.py --csv exp_A exp_B 'exp_C*' -o merged.csv
  python scripts/merge_outcome.py --json 'results/*/eval' -o merged.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

SUMMARY_STEM = "embedding_evaluation_summary"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge evaluate_embeddings.py summaries from multiple directories."
    )
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--csv", action="store_true", help="Merge the CSV summaries (default)")
    fmt.add_argument("--json", action="store_true", help="Merge the JSON summaries")
    p.add_argument("dirs", nargs="+", help="Output directories of evaluate_embeddings.py (glob patterns allowed)")
    p.add_argument("-o", "--output", help="Output file (default: merged_embedding_evaluation_summary.<csv|json>)")
    return p.parse_args(argv)


def expand_dirs(patterns: List[str]) -> List[Path]:
    """Expand glob patterns into a de-duplicated, ordered list of directories."""
    dirs: List[Path] = []
    seen = set()
    for pat in patterns:
        matches = sorted(glob.glob(pat, recursive=True)) if glob.has_magic(pat) else [pat]
        if not matches:
            print(f"Warning: no match for '{pat}'", file=sys.stderr)
        for m in matches:
            path = Path(m)
            if not path.is_dir():
                if not glob.has_magic(pat):
                    print(f"Warning: not a directory: {m}", file=sys.stderr)
                continue
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                dirs.append(path)
    return dirs


def merge_csv(dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        f = d / f"{SUMMARY_STEM}.csv"
        if not f.exists():
            print(f"Warning: {f} not found, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(f)
        df.insert(0, "source_dir", str(d))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    # outer concat so differing metric columns (e.g. purity_k*) are kept
    return pd.concat(frames, ignore_index=True, sort=False)


def merge_json(dirs: List[Path]) -> dict:
    merged = {}
    for d in dirs:
        f = d / f"{SUMMARY_STEM}.json"
        if not f.exists():
            print(f"Warning: {f} not found, skipping", file=sys.stderr)
            continue
        with open(f) as fh:
            merged[str(d)] = json.load(fh)
    return merged


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    use_json = args.json
    ext = "json" if use_json else "csv"
    output = Path(args.output or f"merged_{SUMMARY_STEM}.{ext}")

    dirs = expand_dirs(args.dirs)
    if not dirs:
        print("ERROR: no input directories found.", file=sys.stderr)
        return 2
    print(f"Merging {ext.upper()} summaries from {len(dirs)} directories:")
    for d in dirs:
        print(f"  - {d}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if use_json:
        merged = merge_json(dirs)
        if not merged:
            print("ERROR: no JSON summaries found.", file=sys.stderr)
            return 1
        with open(output, "w") as fh:
            json.dump(merged, fh, indent=2, default=str)
        print(f"\nMerged {len(merged)} JSON summaries -> {output}")
    else:
        df = merge_csv(dirs)
        if df.empty:
            print("ERROR: no CSV summaries found.", file=sys.stderr)
            return 1
        df.to_csv(output, index=False)
        print(f"\nMerged {len(df)} rows from {df['source_dir'].nunique()} directories -> {output}")
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
