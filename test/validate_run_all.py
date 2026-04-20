"""
Evaluate every non-buggy program in a run's journal on the test set (not just best-so-far).

Reads checkpoint/journal.jsonl, runs evaluate.py on each non-buggy node,
and saves results to all_test_results.json keyed by node id.

Usage:
    python validate_run_all.py \
        --log-dir /share/j_sun/as2637/logs/clevr_count/aSSL_0_1_metric_priorlora2_mcts/k10/seed0/aira/1111

Path conventions:
    log_dir:  /share/j_sun/as2637/logs/clevr_count/{setting}/{k}/{seed}/aide/{run_id}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out", default=None,
                        help="Output JSON file (default: <log-dir>/all_test_results.json)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    setting, k, seed, run_id, root_dir = parse_log_dir(log_dir)

    print(f"setting: {setting}, k: {k}, seed: {seed}, run_id: {run_id}")
    print(f"root_dir: {root_dir}")

    entries = load_journal(log_dir)

    # Collect all non-buggy nodes that have code
    candidates = [
        e for e in entries
        if not e.get("is_buggy", True) and e.get("code")
    ]
    print(f"Journal has {len(entries)} entries; {len(candidates)} non-buggy with code")

    out_path = Path(args.out) if args.out else log_dir / "all_test_results.json"

    # Load existing results to skip already-evaluated nodes
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    prog_dir = log_dir / "all_test_programs"
    prog_dir.mkdir(exist_ok=True)

    for entry in sorted(candidates, key=lambda e: e["step"]):
        node = entry.get("data", entry)
        node_id = node.get("id", "")
        step    = node.get("step", 0)
        metric  = node.get("metric")
        code    = node.get("code", "")

        key = node_id if node_id else f"step_{step}"

        if key in all_results:
            print(f"Step {step} id={node_id}: already evaluated, skipping.")
            continue

        program_path = prog_dir / f"program_step{step:04d}_{node_id[:8]}.py"
        with open(program_path, "w") as f:
            f.write(code)

        out_file = prog_dir / f"eval_step{step:04d}_{node_id[:8]}.json"
        metrics = run_eval(program_path, root_dir, setting, out_file)
        metrics["step"]         = step
        metrics["id"]           = node_id
        metrics["train_metric"] = metric
        all_results[key] = metrics
        print(f"Step {step} id={node_id[:8]}: {metrics}")

        # Save incrementally
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
