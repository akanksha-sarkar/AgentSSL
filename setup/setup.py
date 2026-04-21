#!/usr/bin/env python3
"""
Create an experiment directory and copy annotations from setup/datasets (no symlinks).

Optionally copy evaluate.py + binds from setup/evaluate/{metric}/{learning}/ and setup/binds,
with NUM_CLASSES filled from setup/dataset.json.

Layout:
  {root}/{dataset}/{setting}/{split}/{seed}/

Annotations source:
  setup/datasets/{setup_subdir}/{split}/{seed}/annotations/

Example:
  python setup/setup.py --dataset dtd --setting aSSL_backbone_1804 --split k3 --seed 0 \\
    --metric unsupervised_metric --learning SSL

Creates:
  .../seed0/annotations/  (copy)
  .../seed0/evaluate.py   (from setup/evaluate/unsupervised_metric/SSL/evaluate.py, NUM_CLASSES set)
  .../seed0/binds/        (copy of setup/binds)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_dir() -> Path:
    return Path(__file__).resolve().parent


def _normalize_split(split: str) -> str:
    s = split.strip()
    if not s:
        raise ValueError("split must be non-empty")
    if s.startswith("k") and len(s) > 1 and s[1:].isdigit():
        return s
    if s.isdigit():
        return f"k{s}"
    return s


def _normalize_seed(seed: str) -> str:
    s = str(seed).strip()
    if not s:
        raise ValueError("seed must be non-empty")
    if s.startswith("seed"):
        return s
    return f"seed{s}"


def _load_dataset_meta(dataset_json: Path, dataset_key: str) -> dict:
    if not dataset_json.is_file():
        return {}
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    entry = data.get(dataset_key)
    if entry is None:
        return {}
    if isinstance(entry, dict):
        return entry
    return {}


def _setup_subdir(meta: dict, dataset_key: str) -> str:
    return str(meta.get("setup_subdir", dataset_key))


def _annotations_source(
    setup_dir: Path, setup_subdir: str, split: str, seed_dir: str
) -> Path:
    return setup_dir / "datasets" / setup_subdir / split / seed_dir / "annotations"


def _remove_path(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def _copy_annotations(source: Path, dest: Path, *, force: bool) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Annotations source is not a directory: {source}")

    if dest.exists() or dest.is_symlink():
        if not force:
            raise FileExistsError(
                f"{dest} already exists; use --force to replace"
            )
        _remove_path(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, symlinks=False)
    print(f"annotations: copied {source} -> {dest}")


def _substitute_num_classes(py_source: str, num_classes: int) -> str:
    out, n = re.subn(
        r"(?m)^NUM_CLASSES\s*=\s*.*$",
        f"NUM_CLASSES = {num_classes}",
        py_source,
    )
    if n == 0:
        raise ValueError(
            "Template evaluate.py must contain a line matching: NUM_CLASSES = ..."
        )
    return out


def _copy_tree_merge(src: Path, dst: Path, *, force: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Not a directory: {src}")
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} already exists; use --force to replace")
        _remove_path(dst)
    shutil.copytree(src, dst, symlinks=False)
    print(f"copied {src} -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", "-d", required=True, help="Dataset key (see setup/dataset.json)")
    ap.add_argument("--setting", "-s", required=True, help="Experiment setting (e.g. aSSL_backbone_1804)")
    ap.add_argument("--split", required=True, help="Split id: k3 or 3")
    ap.add_argument("--seed", required=True, help="Seed: 0 or seed0")
    ap.add_argument(
        "--metric",
        choices=["noisy_val", "unsupervised_metric"],
        default=None,
        help="Which evaluate template under setup/evaluate/{metric}/",
    )
    ap.add_argument(
        "--learning",
        choices=["SL", "SSL"],
        default=None,
        help="SL vs SSL template folder",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help=f"Repo / experiment root (default: {_repo_root()})",
    )
    ap.add_argument(
        "--check-dataset-json",
        action="store_true",
        help="Require --dataset to exist in setup/dataset.json",
    )
    ap.add_argument(
        "--no-annotations",
        action="store_true",
        help="Only create the experiment folder; do not copy annotations",
    )
    ap.add_argument(
        "--no-binds",
        action="store_true",
        help="With --metric/--learning: do not copy setup/binds",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Replace existing annotations / evaluate.py / binds if present",
    )
    args = ap.parse_args()

    if (args.metric is None) ^ (args.learning is None):
        print(
            "ERROR: provide both --metric and --learning, or neither.",
            file=sys.stderr,
        )
        return 1

    root = (args.root or _repo_root()).resolve()
    split = _normalize_split(args.split)
    seed_dir = _normalize_seed(args.seed)
    setup_dir = _setup_dir()
    dataset_json = setup_dir / "dataset.json"

    if args.check_dataset_json:
        if not dataset_json.is_file():
            print(f"ERROR: {dataset_json} not found", file=sys.stderr)
            return 1
        meta_all = json.loads(dataset_json.read_text(encoding="utf-8"))
        if args.dataset not in meta_all:
            print(
                f"ERROR: dataset {args.dataset!r} not in {dataset_json}. Keys: {list(meta_all)}",
                file=sys.stderr,
            )
            return 1

    meta = _load_dataset_meta(dataset_json, args.dataset)
    setup_subdir = _setup_subdir(meta, args.dataset)

    target = root / args.dataset / args.setting / split / seed_dir
    target.mkdir(parents=True, exist_ok=True)
    print(f"Created: {target}")

    if args.metric is not None and args.learning is not None:
        if not dataset_json.is_file():
            print(f"ERROR: {dataset_json} required for NUM_CLASSES", file=sys.stderr)
            return 1
        meta_all = json.loads(dataset_json.read_text(encoding="utf-8"))
        if args.dataset not in meta_all:
            print(
                f"ERROR: dataset {args.dataset!r} not in {dataset_json}",
                file=sys.stderr,
            )
            return 1
        entry = meta_all[args.dataset]
        if not isinstance(entry, dict) or "num_classes" not in entry:
            print(
                f"ERROR: {args.dataset!r} must have num_classes in {dataset_json}",
                file=sys.stderr,
            )
            return 1
        num_classes = int(entry["num_classes"])

        tmpl_eval = (
            setup_dir / "evaluate" / args.metric / args.learning / "evaluate.py"
        )
        if not tmpl_eval.is_file():
            print(f"ERROR: template not found: {tmpl_eval}", file=sys.stderr)
            return 1

        text = tmpl_eval.read_text(encoding="utf-8")
        text = _substitute_num_classes(text, num_classes)

        dest_eval = target / "evaluate.py"
        if dest_eval.exists() and not args.force:
            print(f"ERROR: {dest_eval} exists; use --force", file=sys.stderr)
            return 1
        dest_eval.write_text(text, encoding="utf-8")
        print(f"evaluate.py: wrote {dest_eval} (NUM_CLASSES = {num_classes})")

        if not args.no_binds:
            src_binds = setup_dir / "binds"
            dst_binds = target / "binds"
            try:
                _copy_tree_merge(src_binds, dst_binds, force=args.force)
            except FileNotFoundError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
            except FileExistsError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1

    if args.no_annotations:
        return 0

    source = _annotations_source(setup_dir, setup_subdir, split, seed_dir)
    dest = target / "annotations"

    try:
        _copy_annotations(source, dest, force=args.force)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            f"Expected annotations under setup/datasets/{setup_subdir}/{split}/{seed_dir}/annotations",
            file=sys.stderr,
        )
        return 1
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
