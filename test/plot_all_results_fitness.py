"""
Plot per-program fitness, val accuracy, and test accuracy for a single run.

Each point is one program (journal node), not the best-so-far.
Joins journal.jsonl with all_val_results.json and all_test_results.json by node id.

Usage:
    python plot_all_results_fitness.py --log-dir /share/j_sun/as2637/logs/dtd/aSSL_0_1_metric_priorlora2_mcts/k6/seed26/aira/1111
    python plot_all_results_fitness.py --log-dir ... --out my_plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Palette ────────────────────────────────────────────────────────────────────
C_FITNESS       = "#7A90CC"   # dusty cornflower blue
C_FITNESS_BUGGY = "#C2CDE8"   # pale blue for buggy
C_VAL           = "#A67FB5"   # muted mauve-purple
C_TEST          = "#C97B50"   # muted terracotta
C_BEST_STAR     = "#E8C46A"   # soft gold star
C_BEST_TEXT     = "#9A6F1E"   # warm brown-gold text
C_BG            = "#F6F4F1"   # warm off-white
C_GRID          = "#E4E0DA"   # warm light gray


def load_journal(log_dir: Path):
    for path in [log_dir / "checkpoint" / "journal.jsonl",
                 log_dir / "json" / "JOURNAL.jsonl"]:
        if path.exists():
            entries = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries
    raise FileNotFoundError(f"No journal found in {log_dir}")


def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log_dir  = Path(args.log_dir).resolve()
    out_path = Path(args.out) if args.out else log_dir / "all_results_fitness_plot.png"

    entries      = load_journal(log_dir)
    val_results  = load_json(log_dir / "all_val_results.json")
    test_results = load_json(log_dir / "all_test_results.json")

    # Build per-program records joined by node id.
    # Include any entry that has fitness OR a test/val result.
    records = []
    for e in entries:
        node    = e.get("data", e)
        node_id = node.get("id", "")
        step    = node.get("step", 0)
        if step == 0:
            continue

        mi      = node.get("metric_info") or {}
        fitness = mi.get("fitness")
        if fitness is None:
            fitness = node.get("metric")

        val_entry  = val_results.get(node_id, {})
        test_entry = test_results.get(node_id, {})

        val_acc  = val_entry.get("acc")
        test_acc = test_entry.get("test_acc")

        # Skip entries with nothing to plot
        if fitness is None and val_acc is None and test_acc is None:
            continue

        records.append({
            "step":     step,
            "fitness":  fitness,
            "val_acc":  val_acc,
            "test_acc": test_acc,
            "buggy":    node.get("is_buggy", False),
        })

    records.sort(key=lambda r: r["step"])

    if not records:
        print("No data to plot.")
        return

    steps        = [r["step"]    for r in records]
    fitness_vals = [r["fitness"] for r in records]
    val_accs     = [r["val_acc"] for r in records]
    test_accs    = [r["test_acc"]for r in records]
    buggy        = [r["buggy"]   for r in records]

    valid_steps   = [s for s, b in zip(steps, buggy) if not b]
    valid_fitness = [a for a, b in zip(fitness_vals, buggy) if not b]
    buggy_steps   = [s for s, b in zip(steps, buggy) if b]
    buggy_fitness = [a for a, b in zip(fitness_vals, buggy) if b]

    val_s  = [s for s, v in zip(steps, val_accs)  if v is not None]
    val_v  = [v for v       in val_accs             if v is not None]
    test_s = [s for s, v in zip(steps, test_accs) if v is not None]
    test_v = [v for v       in test_accs            if v is not None]

    # ── Figure setup ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(colors="#555555", labelsize=10)
    ax.yaxis.label.set_color("#555555")
    ax.xaxis.label.set_color("#555555")

    ax.grid(True, color=C_GRID, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    # ── Fitness ──────────────────────────────────────────────────────────────────
    if buggy_steps:
        ax.scatter(buggy_steps, buggy_fitness, s=20, marker="x", color=C_FITNESS_BUGGY,
                   alpha=0.5, linewidths=1, zorder=2, label="Fitness (buggy)")
    if valid_steps:
        ax.plot(valid_steps, valid_fitness, "o-", color=C_FITNESS,
                markersize=5, linewidth=1.5, alpha=0.75, zorder=3, label="Fitness")

    # ── Val acc ──────────────────────────────────────────────────────────────────
    if val_s:
        ax.plot(val_s, val_v, "^-", color=C_VAL,
                markersize=6, linewidth=1.8, alpha=0.9, zorder=4, label="Val acc")

    # ── Test acc ─────────────────────────────────────────────────────────────────
    if test_s:
        ax.plot(test_s, test_v, "s-", color=C_TEST,
                markersize=6, linewidth=1.8, alpha=0.9, zorder=4, label="Test acc")

        best_idx = max(range(len(test_v)), key=lambda i: test_v[i])
        best_s, best_v = test_s[best_idx], test_v[best_idx]

        ax.scatter([best_s], [best_v], s=350, marker="*", color=C_BEST_STAR,
                   edgecolors="#888800", linewidths=0.8, zorder=7)
        ax.annotate(
            f"best test: {best_v:.4f}",
            xy=(best_s, best_v),
            xytext=(10, 8), textcoords="offset points",
            fontsize=9.5, fontweight="bold", color=C_BEST_TEXT,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=C_BG)],
        )

    # ── Title & labels ───────────────────────────────────────────────────────────
    parts = log_dir.parts
    try:
        idx   = next(i for i, p in enumerate(parts) if p == "resisc45")
        title = "/".join(parts[idx:])
    except StopIteration:
        title = str(log_dir)

    ax.set_title(title, fontsize=11, color="#333333", pad=12)
    ax.set_xlabel("Agent Step", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(bottom=0)

    legend = ax.legend(loc="lower right", fontsize=9.5, framealpha=0.85,
                       edgecolor="#CCCCCC", fancybox=True)
    legend.get_frame().set_facecolor(C_BG)

    n_val  = len(val_s)
    n_test = len(test_s)
    fig.text(0.5, 0.01,
             f"Each point = one program  |  {n_val} val evals  |  {n_test} test evals",
             ha="center", fontsize=9, color="#888888")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
