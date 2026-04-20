#!/usr/bin/env python3
"""
Summarize COCO train.json (num classes) and unlabelled.json (datapoints) for KITTI-style splits.

Expected layout (under --split-dir):

  seed0/annotations/train/train.json
  seed0/annotations/unlabelled/unlabelled.json
  seed26/...
  ...

Usage:
  python scripts/kitti_split_counts.py
  python scripts/kitti_split_counts.py --split-dir /home/as2637/agentSSL/kitti/splits/k5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def train_stats(data: dict) -> dict:
    cats = data.get("categories") or []
    anns = data.get("annotations") or []
    imgs = data.get("images") or []
    cat_ids = sorted(c["id"] for c in cats if isinstance(c, dict) and "id" in c)
    ann_cats = sorted({a["category_id"] for a in anns if isinstance(a, dict) and "category_id" in a})
    return {
        "num_classes": len(cats),
        "num_images": len(imgs),
        "num_annotations": len(anns),
        "category_ids_declared": cat_ids,
        "unique_category_ids_in_annotations": ann_cats,
    }


def unlabelled_datapoints(data) -> int | None:
    if isinstance(data, dict) and "images" in data:
        return len(data["images"])
    if isinstance(data, list):
        return len(data)
    return None


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    default_split = repo / "kitti/splits/k5"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=default_split,
        help=f"Directory containing seed*/annotations/ (default: {default_split})",
    )
    args = ap.parse_args()

    split_dir: Path = args.split_dir.resolve()
    if not split_dir.is_dir():
        print(f"ERROR: not a directory: {split_dir}", file=sys.stderr)
        return 1

    seeds = sorted(
        p for p in split_dir.iterdir()
        if p.is_dir() and p.name.startswith("seed")
    )
    if not seeds:
        print(f"No seed* directories under {split_dir}", file=sys.stderr)
        return 1

    print(f"Split directory: {split_dir}")
    print()

    for seed_dir in seeds:
        train_p = seed_dir / "annotations" / "train" / "train.json"
        unl_p = seed_dir / "annotations" / "unlabelled" / "unlabelled.json"

        print(f"=== {seed_dir.name} ===")
        if not train_p.is_file():
            print(f"  MISSING: {train_p}")
        else:
            t = _load(train_p)
            if not isinstance(t, dict):
                print(f"  train.json: unexpected top-level type {type(t)}")
            else:
                s = train_stats(t)
                print(f"  train.json: {train_p}")
                print(f"    num_classes: {s['num_classes']}")
                print(f"    num_images: {s['num_images']}")
                print(f"    num_annotations: {s['num_annotations']}")
                if s["num_classes"]:
                    print(f"    category ids (declared): {s['category_ids_declared'][:20]}"
                          + (" ..." if len(s["category_ids_declared"]) > 20 else ""))
                if s["unique_category_ids_in_annotations"] != s["category_ids_declared"]:
                    print(f"    unique category_id in anns: {s['unique_category_ids_in_annotations']}")

        if not unl_p.is_file():
            print(f"  MISSING: {unl_p}")
        else:
            u = _load(unl_p)
            n = unlabelled_datapoints(u)
            print(f"  unlabelled.json: {unl_p}")
            print(f"    datapoints: {n}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
