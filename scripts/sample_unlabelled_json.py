#!/usr/bin/env python3
"""
Subsample a COCO-style unlabelled JSON (typically ``images`` + empty ``annotations``).

Writes a new JSON with a random fraction or exact count of the images. Preserves other
top-level keys (e.g. ``info``). If ``annotations`` is non-empty, keeps only rows whose
``image_id`` is in the sampled image id set.

Example:
  python scripts/sample_unlabelled_json.py path/to/unlabelled.json
  python scripts/sample_unlabelled_json.py path/to/unlabelled.json --fraction 0.5
  python scripts/sample_unlabelled_json.py path/to/unlabelled.json --fraction 0 -o path/to/unlabelled_0.json
  python scripts/sample_unlabelled_json.py path/to/unlabelled.json --count 1 -o path/to/unlabelled_0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from unlabelled_subset import subset_unlabelled_coco


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        type=Path,
        help="Path to unlabelled.json (or any COCO dict with 'images')",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path (default: same dir as input, "
            "unlabelled_{pct}.json where pct = round(fraction*100) for --fraction; "
            "unlabelled_{count}.json for --count)"
        ),
    )
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=None,
        help="Fraction of images to keep (default: 0.5 if --count not used)",
    )
    g.add_argument(
        "-n",
        "--count",
        type=int,
        default=None,
        help="Exact number of images to keep (mutually exclusive with --fraction)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible sampling (default: 0)",
    )
    args = p.parse_args()

    fraction = args.fraction
    count = args.count
    if count is None and fraction is None:
        fraction = 0.5

    inp = args.input.resolve()
    if not inp.is_file():
        print(f"ERROR: not a file: {inp}", file=sys.stderr)
        return 1

    out_path = args.output
    if out_path is None:
        if count is not None:
            out_path = inp.parent / f"unlabelled_{count}.json"
        else:
            assert fraction is not None
            pct = int(round(fraction * 100))
            out_path = inp.parent / f"unlabelled_{pct}.json"
    else:
        out_path = out_path.resolve()

    with open(inp, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("ERROR: root JSON must be an object", file=sys.stderr)
        return 1

    try:
        if count is not None:
            new_data = subset_unlabelled_coco(data, args.seed, count=count)
            mode = f"count={count}"
        else:
            new_data = subset_unlabelled_coco(data, args.seed, fraction=fraction)
            mode = f"fraction={fraction}"
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    n_old = len(data.get("images") or [])
    n_new = len(new_data.get("images") or [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path} ({n_new}/{n_old} images, {mode}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
