#!/usr/bin/env python3
"""
Sanity-check Semi-iNat–style files under setup/datasets/inat/.../annotations/:
- train.json: every category has kingdom + phylum; one annotation per image.
- unlabelled.json: COCO images only; expected keys are id + file_name (no taxonomy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _check_train(path: Path) -> list[str]:
    errs: list[str] = []
    with open(path) as f:
        d = json.load(f)
    cats = d.get("categories") or []
    for c in cats:
        for k in ("kingdom", "phylum"):
            if k not in c or not c[k]:
                errs.append(f"category id={c.get('id')!r} missing/empty {k!r}")
    n_img = len(d.get("images") or [])
    n_ann = len(d.get("annotations") or [])
    if n_img != n_ann:
        errs.append(f"train: len(images)={n_img} != len(annotations)={n_ann}")
    return errs


def _check_unlabelled(path: Path) -> list[str]:
    errs: list[str] = []
    with open(path) as f:
        d = json.load(f)
    for bad in ("annotations", "categories"):
        if d.get(bad):
            errs.append(f"unlabelled: non-empty {bad!r} (expected absent or empty)")
    imgs = d.get("images") or []
    extra_keys: set[str] = set()
    for im in imgs:
        extra_keys |= set(im.keys()) - {"id", "file_name"}
    if extra_keys:
        errs.append(
            f"unlabelled: unexpected image keys (expected id, file_name only): {sorted(extra_keys)}"
        )
    if not imgs:
        errs.append("unlabelled: no images")
    return errs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "setup" / "datasets" / "inat" / "k10" / "seed0" / "annotations",
        help="Directory containing train/ and unlabelled/ subfolders",
    )
    args = p.parse_args()
    train_j = args.root / "train" / "train.json"
    unl_j = args.root / "unlabelled" / "unlabelled.json"
    all_errs: list[str] = []
    if not train_j.is_file():
        all_errs.append(f"missing {train_j}")
    else:
        all_errs.extend(_check_train(train_j))
    if not unl_j.is_file():
        all_errs.append(f"missing {unl_j}")
    else:
        all_errs.extend(_check_unlabelled(unl_j))

    if all_errs:
        for e in all_errs:
            print(e, file=sys.stderr)
        return 1
    with open(train_j) as f:
        tc = len(json.load(f).get("categories", []))
    with open(unl_j) as f:
        n_unl = len(json.load(f).get("images", []))
    print(
        f"OK: {train_j} has {tc} categories (kingdom+phylum on each); "
        f"{unl_j} has {n_unl} unlabeled image entries (id+file_name only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
