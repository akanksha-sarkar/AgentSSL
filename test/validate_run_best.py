"""
Evaluate the best program found so far at every N-th step milestone.

For each milestone k*N (k=1,2,3,...), finds the non-buggy entry with the
highest metric among all steps <= k*N, then runs evaluate.py on it (skipping
if already evaluated or if the best program hasn't changed since the last
milestone).

Results are saved to <log-dir>/best_prog_eval_results.json, keyed by
"best_at_step_<milestone>".

Usage:
    python validate_run_best.py \
        --log-dir /share/j_sun/as2637/logs/clevr_count/aSSL_noisy_val/k10/seed26/aira/1000 \
        --every 5

    python validate_run_best.py --log-dir ... --every 10 --out /path/to/out.json

Path conventions (same as validate_run_all.py):
    log_dir:  /share/j_sun/as2637/logs/clevr_count/{setting}/{k}/{seed}/aira/{run_id}
    root_dir: /home/as2637/agentSSL/clevr_count/{setting}/{k}/{seed}
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


TEST_DIR = Path(__file__).parent
ROOT_BASE = Path("/home/as2637/agentSSL/clevr_count")
EVALUATE_PY = TEST_DIR / "evaluate.py"


def parse_log_dir(log_dir: Path):
    parts = log_dir.parts
    try:
        clevr_count_idx = next(i for i, p in enumerate(parts) if p == "clevr_count")
    except StopIteration:
        raise ValueError(f"Could not find 'clevr_count' in path: {log_dir}")
    setting = parts[clevr_count_idx + 1]
    k       = parts[clevr_count_idx + 2]
    seed    = parts[clevr_count_idx + 3]
    run_id  = parts[clevr_count_idx + 5]
    root_dir = ROOT_BASE / setting / k / seed
    return setting, k, seed, run_id, root_dir


def load_journal(log_dir: Path):
    candidates = [
        log_dir / "checkpoint" / "journal.jsonl",
        log_dir / "json" / "JOURNAL.jsonl",
    ]
    for path in candidates:
        if path.exists():
            entries = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries
    raise FileNotFoundError(f"No journal found in {log_dir}")


def run_eval(program_path: Path, root_dir: Path, setting: str, out_file: Path) -> dict:
    cmd = [
        sys.executable, str(EVALUATE_PY),
        "--program-path", str(program_path),
        "--root-dir",     str(root_dir),
        "--setting",      setting,
        "--out-file",     str(out_file),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(TEST_DIR), capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: evaluate.py exited with code {result.returncode}", flush=True)
        return {"error": f"exit_code={result.returncode}"}
    if out_file.exists():
        with open(out_file) as f:
            return json.load(f)
    return {}


def best_entry_up_to_step(candidates, max_step):
    """Return the non-buggy entry with the highest metric among steps <= max_step."""
    pool = [e for e in candidates if e["step"] <= max_step]
    if not pool:
        return None
    return max(pool, key=lambda e: (e["metric"] is not None, e["metric"] or 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--every", type=int, default=5,
                        help="Evaluate best program at every N-th step (default: 5)")
    parser.add_argument("--out", default=None,
                        help="Output JSON file (default: <log-dir>/best_prog_eval_results.json)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    every   = args.every

    setting, k, seed, run_id, root_dir = parse_log_dir(log_dir)
    print(f"setting: {setting}, k: {k}, seed: {seed}, run_id: {run_id}")
    print(f"root_dir: {root_dir}")
    print(f"Evaluating best program at every {every} steps")

    entries = load_journal(log_dir)

    # All non-buggy entries with code, sorted by step
    candidates = sorted(
        [e for e in entries if not e.get("is_buggy", True) and e.get("code")],
        key=lambda e: e["step"],
    )
    print(f"Journal has {len(entries)} entries; {len(candidates)} non-buggy with code")

    if not candidates:
        print("No candidates to evaluate.")
        return

    max_step_in_journal = max(e["step"] for e in entries)

    out_path = Path(args.out) if args.out else log_dir / "best_prog_eval_results.json"

    # Load existing results to allow incremental re-runs
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    prog_dir = log_dir / "best_prog_eval_programs"
    prog_dir.mkdir(exist_ok=True)

    # Build milestones: every N steps up to max_step_in_journal
    milestones = list(range(every, max_step_in_journal + every, every))

    last_evaluated_id = None  # avoid re-evaluating unchanged best

    for milestone in milestones:
        key = f"best_at_step_{milestone}"

        best = best_entry_up_to_step(candidates, milestone)
        if best is None:
            print(f"Milestone {milestone}: no non-buggy programs yet, skipping.")
            continue

        node_id = best.get("id", "")
        step    = best["step"]
        metric  = best["metric"]

        print(f"Milestone {milestone}: best so far is step={step} id={node_id[:8]} metric={metric}")

        if key in all_results:
            print(f"  Already evaluated, skipping.")
            last_evaluated_id = node_id
            continue

        if node_id == last_evaluated_id:
            # Best hasn't changed; record a reference instead of re-running
            print(f"  Best program unchanged since last milestone, recording reference.")
            all_results[key] = {
                "step":         step,
                "id":           node_id,
                "train_metric": metric,
                "note":         "same best as previous milestone, no re-eval",
            }
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            continue

        program_path = prog_dir / f"program_step{step:04d}_{node_id[:8]}.py"
        with open(program_path, "w") as f:
            f.write(best["code"])

        out_file = prog_dir / f"eval_milestone{milestone:04d}_step{step:04d}_{node_id[:8]}.json"
        metrics = run_eval(program_path, root_dir, setting, out_file)
        metrics["step"]         = step
        metrics["id"]           = node_id
        metrics["train_metric"] = metric
        metrics["milestone"]    = milestone
        all_results[key] = metrics
        print(f"  Result: {metrics}")

        last_evaluated_id = node_id

        # Save incrementally after each evaluation
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
