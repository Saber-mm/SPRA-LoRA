#!/usr/bin/env python
"""Bar chart of average accuracy across GLUE datasets at a fixed round.

For each alpha, each baseline becomes one bar whose height is the mean across
datasets of the per-dataset (per-seed-mean) accuracy at a designated round.
Per-alpha eval rounds are configured in EVAL_ROUND. One figure per alpha.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ROOT, collect, avg_across_tasks, order, METHOD_COLOR

ap = argparse.ArgumentParser()
ap.add_argument("--alphas", nargs="+", default=["0.4", "0.8", "2.0"])
ap.add_argument("--clients", default="40", help="client count to plot (runs/glue/<task>_n<clients>)")
ap.add_argument("--out", default=os.path.join(ROOT, "plots", "glue", "bar"))
cli = ap.parse_args()

METRIC = "acc"
OUT_DIR = cli.out
# Round at which to evaluate each alpha (0-based round index from the logs;
# plot_eval_curves.py draws round r at x = r + 5).
EVAL_ROUND = {"0.4": 145, "0.8": 145, "2.0": 145}  # 145 = last round

os.makedirs(OUT_DIR, exist_ok=True)


def per_method_avg(alpha):
    """Return {method: avg acc across datasets} at EVAL_ROUND[alpha]."""
    if alpha not in EVAL_ROUND:
        raise ValueError(f"No eval round configured for alpha={alpha}")
    return avg_across_tasks(collect(alpha, cli.clients), EVAL_ROUND[alpha])


per_alpha_avg = {}
for alpha in cli.alphas:
    avg = per_method_avg(alpha)
    if not avg:
        print(f"[skip] no data for alpha={alpha}")
        continue
    per_alpha_avg[alpha] = avg

    methods = order(list(avg.keys()), avg)
    heights = [avg[m] for m in methods]
    colors = [METHOD_COLOR.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(methods))
    bars = ax.bar(xs, heights, width=0.9, color=colors, edgecolor="white", linewidth=1.0)

    for x, h in zip(xs, heights):
        ax.text(x, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=12, rotation=45)

    display_round = EVAL_ROUND[alpha]
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=13)
    ax.set_ylabel("average test accuracy", fontsize=16)
    ax.set_title(
        rf"Average accuracy across GLUE datasets "
        rf"(data heterogeneity ($\alpha$)={alpha})",
        fontsize=15, pad=12,
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(55, 90)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    base = os.path.join(
        OUT_DIR,
        f"bar_avg_across_datasets_n{cli.clients}_alpha{alpha}_r{display_round}_{METRIC}",
    )
    for ext in (".png", ".pdf"):
        fig.savefig(base + ext, dpi=300, bbox_inches="tight")
        print(f"wrote {base + ext}")
    plt.close(fig)


# Combined grouped-bar plot: one cluster per alpha, one bar per method.
if len(per_alpha_avg) >= 2:
    alphas = list(per_alpha_avg.keys())
    all_methods = set().union(*[set(d.keys()) for d in per_alpha_avg.values()])
    # Sort by mean across alphas (where present), with PINNED first
    method_means = {
        m: float(np.mean([d[m] for d in per_alpha_avg.values() if m in d]))
        for m in all_methods
    }
    methods = order(list(all_methods), method_means)

    n_methods = len(methods)
    n_alphas = len(alphas)
    width = 0.9 / n_methods
    method_colors = [METHOD_COLOR.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_alphas), 6))
    centers = np.arange(n_alphas)
    for j, method in enumerate(methods):
        offsets = (j - (n_methods - 1) / 2) * width
        heights = [per_alpha_avg[a].get(method, 0.0) for a in alphas]
        ax.bar(
            centers + offsets, heights, width,
            color=method_colors[j], edgecolor="white", linewidth=0.8,
            label=method,
        )
        for x, h in zip(centers + offsets, heights):
            if h > 0:
                ax.text(x, h + 0.3, f"{h:.1f}", ha="center", va="bottom",
                        fontsize=9, rotation=45)

    ax.set_xticks(centers)
    ax.set_xticklabels([rf"$\alpha$={a}" for a in alphas], fontsize=14)
    ax.set_ylabel("average test accuracy", fontsize=16)
    ax.set_title(
        r"Average accuracy across GLUE datasets per "
        r"data heterogeneity ($\alpha$) setting",
        fontsize=15, pad=12,
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(55, 93)  # headroom so the legend clears the rotated labels
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ncol = (n_methods + 2) // 3
    ax.legend(frameon=True, loc="upper center", fontsize=11, ncol=ncol)
    fig.tight_layout()

    base = os.path.join(OUT_DIR, f"bar_avg_across_datasets_grouped_alpha_n{cli.clients}_{METRIC}")
    for ext in (".png", ".pdf"):
        fig.savefig(base + ext, dpi=300, bbox_inches="tight")
        print(f"wrote {base + ext}")
    plt.close(fig)
