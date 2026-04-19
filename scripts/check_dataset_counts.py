#!/usr/bin/env python3
"""
Report class count and datapoint counts from COCO-style train / unlabelled JSON files.

Usage:
  python scripts/check_dataset_counts.py \\
    --train clevr_count/aSSL_backbone/k10/seed0/annotations/train/train.json \\
    --unlabelled clevr_count/aSSL_backbone/k10/seed0/annotations/unlabelled/unlabelled.json

Or with defaults (clevr_count k10 seed0 relative to repo root):
  python scripts/check_dataset_counts.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_unlabelled(data) -> dict:
    if isinstance(data, dict) and "images" in data:
        return {"format": "coco_images", "datapoints": len(data["images"])}
    if isinstance(data, list):
        return {"format": "list", "datapoints": len(data)}
    return {"format": type(data).__name__, "datapoints": None}


def _analyze_train(data) -> dict:
    if not isinstance(data, dict):
        return {
            "format": type(data).__name__,
            "num_classes": None,
            "num_images": None,
            "num_annotations": None,
            "unique_category_ids_in_anns": None,
        }

    cats = data.get("categories") or []
    anns = data.get("annotations") or []
    imgs = data.get("images") or []

    num_classes = len(cats)
    cat_ids_declared = sorted(c["id"] for c in cats if isinstance(c, dict) and "id" in c)
    ann_cat_ids = [a["category_id"] for a in anns if isinstance(a, dict) and "category_id" in a]
    unique_in_anns = sorted(set(ann_cat_ids))

    return {
        "format": "coco",
        "num_classes": num_classes,
        "category_id_range_declared": (min(cat_ids_declared), max(cat_ids_declared))
        if cat_ids_declared
        else None,
        "num_images": len(imgs),
        "num_annotations": len(anns),
        "unique_category_ids_in_annotations": unique_in_anns,
        "num_unique_category_ids_in_annotations": len(unique_in_anns),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    default_train = repo / "clevr_count/aSSL_backbone/k10/seed0/annotations/train/train.json"
    default_unl = repo / "clevr_count/aSSL_backbone/k10/seed0/annotations/unlabelled/unlabelled.json"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train",
        type=Path,
        default=default_train,
        help=f"Path to train.json (default: {default_train})",
    )
    p.add_argument(
        "--unlabelled",
        type=Path,
        default=default_unl,
        help=f"Path to unlabelled.json (default: {default_unl})",
    )
    args = p.parse_args()

    for label, path in ("train", args.train), ("unlabelled", args.unlabelled):
        if not path.is_file():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            return 1

    print("Repo root (script parent):", repo)
    print()

    train_path = args.train.resolve()
    unl_path = args.unlabelled.resolve()

    print(f"Train annotation: {train_path}")
    train_data = _load_json(train_path)
    train_info = _analyze_train(train_data)
    for k, v in train_info.items():
        print(f"  {k}: {v}")
    print()

    print(f"Unlabelled annotation: {unl_path}")
    unl_data = _load_json(unl_path)
    unl_info = _count_unlabelled(unl_data)
    for k, v in unl_info.items():
        print(f"  {k}: {v}")
    st = os.stat(unl_path)
    print(f"  file_size_bytes: {st.st_size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
