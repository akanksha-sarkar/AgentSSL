"""
Evaluate every non-buggy program in a run's journal on the test set (not just best-so-far).

Reads checkpoint/journal.jsonl, runs evaluate.py on each non-buggy node,
and saves results to all_test_results.json keyed by node id.

Usage:
    python validate_run_all.py \
        --log-dir /share/j_sun/as2637/logs/resisc45/aSL/k1/seed0/aide/1000

Path conventions:
    log_dir:  /share/j_sun/as2637/logs/resisc45/{setting}/{k}/{seed}/aide/{run_id}
    root_dir: /home/as2637/agentSSL/resisc45/{setting}/{k}/{seed}
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


TEST_DIR = Path(__file__).parent


def infer_dataset_from_log_dir(log_dir: Path) -> str:
    """
    Infer dataset name from a log_dir like:
      /share/j_sun/as2637/logs/<dataset>/<setting>/<k>/<seed>/aira/<run_id>
    """
    parts = log_dir.parts
    try:
        logs_idx = next(i for i, p in enumerate(parts) if p == "logs")
    except StopIteration:
        raise ValueError(f"Could not infer dataset (no 'logs' segment) from: {log_dir}")
    if logs_idx + 1 >= len(parts):
        raise ValueError(f"Could not infer dataset (nothing after 'logs') from: {log_dir}")
    return parts[logs_idx + 1]


def parse_log_dir(log_dir: Path, root_base: Path, args: argparse.Namespace):
    parts = log_dir.parts
    try:
        dataset_idx = next(i for i, p in enumerate(parts) if p == args.dataset)
    except StopIteration:
        raise ValueError(f"Could not find '{args.dataset}' in path: {log_dir}")
    setting = parts[dataset_idx + 1]
    k       = parts[dataset_idx + 2]
    seed    = parts[dataset_idx + 3]
    run_id  = parts[dataset_idx + 5]
    root_dir = root_base / setting / k / seed
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


def infer_learning_from_setting(setting: str) -> str:
    # aSL* (but not aSSL*) -> SL, else -> SSL
    if setting.startswith("aSL") and not setting.startswith("aSSL"):
        return "SL"
    return "SSL"


def run_eval(
    program_path: Path,
    root_dir: Path,
    out_file: Path,
    evaluate_py: Path,
    *,
    dataset: str,
    learning: str,
    eval_type: str,
) -> dict:
    cmd = [
        sys.executable, str(evaluate_py),
        "--program-path", str(program_path),
        "--root-dir",     str(root_dir),
        "--learning",     learning,
        "--dataset",      dataset,
        "--eval_type",    eval_type,
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
    parser.add_argument("--out", default=None, help="Output JSON file (default: <log-dir>/all_test_results.json)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (if omitted, inferred from --log-dir, e.g. .../logs/dtd/...)",
    )
    parser.add_argument("--eval-type", default="test", choices=["test", "val"], help="Which split to evaluate")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    if args.dataset is None:
        args.dataset = infer_dataset_from_log_dir(log_dir)

    BASE = Path("/home/as2637/agentSSL")
    root_base = BASE / args.dataset
    evaluate_py = Path("/share/j_sun/as2637/logs/eval/evaluate_program.py")

    setting, k, seed, run_id, root_dir = parse_log_dir(log_dir, root_base, args)
    learning = infer_learning_from_setting(setting)

    print(f"dataset: {args.dataset}")
    print(f"setting: {setting}, k: {k}, seed: {seed}, run_id: {run_id}")
    print(f"root_dir: {root_dir}")
    print(f"root_base: {root_base}")
    print(f"evaluate_py: {evaluate_py}")
    print(f"learning: {learning}")
    print(f"eval_type: {args.eval_type}")

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
        metrics = run_eval(
            program_path,
            root_dir,
            out_file,
            evaluate_py,
            dataset=args.dataset,
            learning=learning,
            eval_type=args.eval_type,
        )
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
