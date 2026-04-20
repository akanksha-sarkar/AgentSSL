"""
Select journal entries with train metric >= a threshold, then run evaluate.py on each.

Reuses the same paths and evaluate invocation as validate_run_best.py / validate_run_all.py.

Usage:
    python validate_run_threshold.py \\
        --log-dir /share/j_sun/as2637/logs/clevr_count/aSSL_noisy_val/k10/seed0/aira/1000 \\
        --min-metric 0.34

    python validate_run_threshold.py --log-dir ... --min-metric 0.34 --out /path/to/results.json

Results are saved incrementally to JSON keyed by node id (same as validate_run_all.py).
Programs and per-eval JSON files go under <log-dir>/threshold_prog_eval_programs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validate_run_best import load_journal, parse_log_dir, run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument(
        "--min-metric",
        type=float,
        default=0.34,
        help="Evaluate non-buggy programs with metric >= this value (default: 0.34)",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Output JSON (default: <log-dir>/threshold_ge_<min>_eval_results.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching entries and exit without running evaluate.py",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    min_m = args.min_metric

    setting, k, seed, run_id, root_dir = parse_log_dir(log_dir)
    print(f"setting: {setting}, k: {k}, seed: {seed}, run_id: {run_id}")
    print(f"root_dir: {root_dir}")
    print(f"Selecting non-buggy programs with metric >= {min_m}")

    entries = load_journal(log_dir)
    selected = []
    for e in entries:
        if e.get("is_buggy", True):
            continue
        code = e.get("code") or ""
        if not code.strip():
            continue
        m = e.get("metric")
        if m is None or not isinstance(m, (int, float)):
            continue
        if float(m) < min_m:
            continue
        selected.append(e)

    # Stable order: by step, then id
    selected.sort(key=lambda e: (e.get("step", 0), e.get("id", "")))

    print(
        f"Journal has {len(entries)} entries; "
        f"{len(selected)} non-buggy with code and metric >= {min_m}",
    )

    if not selected:
        print("Nothing to evaluate.")
        return

    tag = str(min_m).replace(".", "_")
    out_path = args.out
    if out_path is None:
        out_path = log_dir / f"threshold_ge_{tag}_eval_results.json"
    else:
        out_path = Path(out_path).resolve()

    if args.dry_run:
        for e in selected:
            print(
                f"  step={e.get('step')} id={e.get('id')} "
                f"metric={e.get('metric')}",
            )
        print(f"\nDry run: would write results to {out_path}")
        return

    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    prog_dir = log_dir / "threshold_prog_eval_programs"
    prog_dir.mkdir(exist_ok=True)

    for entry in selected:
        node_id = entry.get("id", "")
        step = entry.get("step", 0)
        metric = entry.get("metric")
        code = entry.get("code", "")
        key = node_id if node_id else f"step_{step}"

        if key in all_results:
            print(f"Step {step} id={node_id}: already in results, skipping.")
            continue

        program_path = prog_dir / f"program_step{step:04d}_{node_id[:8]}.py"
        with open(program_path, "w") as f:
            f.write(code)

        out_file = prog_dir / f"eval_step{step:04d}_{node_id[:8]}.json"
        metrics = run_eval(program_path, root_dir, setting, out_file)
        metrics["step"] = step
        metrics["id"] = node_id
        metrics["train_metric"] = metric
        metrics["min_metric_threshold"] = min_m
        all_results[key] = metrics
        print(f"Step {step} id={str(node_id)[:8]}: {metrics}")

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
