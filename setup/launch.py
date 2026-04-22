#!/usr/bin/env python3

import argparse
import itertools
import os
import subprocess
from pathlib import Path


def launch_job(dataset, shot, seed, eval_method, script, dry_run=False):
    job_name = f"{dataset}_{shot}_{seed}_{eval_method}"

    log_dir = Path("logs") / "sweep" / dataset / str(shot) / f"seed{seed}" / eval_method
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        "-J", job_name,
        "-o", str(log_dir / "exp.out"),
        "-e", str(log_dir / "exp.err"),
        script,
        dataset,
        str(shot),
        str(seed),
        eval_method,
    ]

    print(f"🚀 Launching: {job_name}")
    print(f"   Command: {' '.join(cmd)}")

    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_int_list(values):
    return [int(v) for v in values]

def get_shots(dataset):
    if dataset == "clevr_count":
        return [10, 20]
    elif dataset == "dtd":
        return [3, 6]
    elif dataset == "resisc45":
        return [1, 2]
    elif dataset == "kitti":
        return [5, 10]
    elif dataset == "sun397":
        return [3, 6]
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    args = parser.parse_args()
    datasets = ["kitti"]
    eval_methods = ["unsupervised"]

    launched = 0
    for dataset in datasets:
        shots = get_shots(dataset)
        for shot in shots:
            for eval_method in eval_methods:
                save_path = Path("setup", "warmstart", dataset, eval_method, f"k{shot}", "results.json")
                if os.path.exists(save_path):
                    print(f"Results file already exists: {save_path}")
                    continue  
                launch_job(
                    dataset=dataset,
                    shot=shot,
                    seed=0,
                    eval_method=eval_method,
                    script="setup/warmstart.sub",
                    dry_run=args.dry_run,
                )
                launched += 1

    print("\n----------------------------------")
    print(f"✅ Total launched: {launched}")


if __name__ == "__main__":
    main()