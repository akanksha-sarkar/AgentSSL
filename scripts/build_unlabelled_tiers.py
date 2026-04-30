#!/usr/bin/env python3
"""
For each ``seed*`` under a split root, read ``annotations/unlabelled/unlabelled.json`` and write:

- ``unlabelled_0.json`` — exactly **1** unlabeled image (tier name; not zero images)
- ``unlabelled_10.json`` — 10% of images
- ``unlabelled_50.json`` — 50% of images

RNG seed per folder is parsed from the directory name (``seed0`` → 0, ``seed26`` → 26) so runs are reproducible.

Example:
  python scripts/build_unlabelled_tiers.py
  python scripts/build_unlabelled_tiers.py --root /path/to/dtd/aSSL_backbone_unsupervised/k6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from unlabelled_subset import subset_unlabelled_coco

_DEFAULT_ROOT = Path("/home/as2637/agentSSL/dtd/aSSL_backbone_unsupervised/k6")


def _rng_seed_from_seed_dir(seed_dir: Path) -> int:
    m = re.fullmatch(r"seed(\d+)", seed_dir.name)
    if not m:
        raise ValueError(f"expected directory name like seed0, got {seed_dir.name!r}")
    return int(m.group(1))


def _write(out_path: Path, data: object) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help=f"Split directory containing seed* folders (default: {_DEFAULT_ROOT})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not write files",
    )
    args = p.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    seed_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed"))
    if not seed_dirs:
        print(f"ERROR: no seed* directories under {root}", file=sys.stderr)
        return 1

    n_ok = 0
    for seed_dir in seed_dirs:
        try:
            rng_seed = _rng_seed_from_seed_dir(seed_dir)
        except ValueError as e:
            print(f"SKIP {seed_dir}: {e}", file=sys.stderr)
            continue

        src = seed_dir / "annotations" / "unlabelled" / "unlabelled.json"
        if not src.is_file():
            print(f"SKIP {seed_dir}: missing {src}", file=sys.stderr)
            continue

        out_dir = src.parent
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"SKIP {seed_dir}: root JSON is not an object", file=sys.stderr)
            continue

        n_img = len(data.get("images") or [])
        plans = [
            ("unlabelled_0.json", lambda: subset_unlabelled_coco(data, rng_seed, count=1)),
            ("unlabelled_10.json", lambda: subset_unlabelled_coco(data, rng_seed, fraction=0.1)),
            ("unlabelled_50.json", lambda: subset_unlabelled_coco(data, rng_seed, fraction=0.5)),
        ]

        for name, fn in plans:
            dest = out_dir / name
            try:
                subset = fn()
            except ValueError as e:
                print(f"ERROR {seed_dir} {name}: {e}", file=sys.stderr)
                return 1
            k = len(subset.get("images") or [])
            if args.dry_run:
                print(f"would write {dest} ({k}/{n_img} images, rng_seed={rng_seed})")
            else:
                _write(dest, subset)
                print(f"wrote {dest} ({k}/{n_img} images, rng_seed={rng_seed})")
        n_ok += 1

    if n_ok == 0:
        print("ERROR: no seed directories processed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
