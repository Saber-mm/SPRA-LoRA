#!/usr/bin/env python
"""Bar chart of average accuracy across GLUE datasets, per client count.

Mirrors plot_bar_avg_across_datasets.py but groups by number of clients
(n40, n60, n80) instead of alpha. Per-setting eval round is configured in
EVAL_ROUND. Writes one figure per client count plus one grouped figure
(clusters = client counts, bars colored by method).
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ROOT, collect, avg_across_tasks, order, METHOD_COLOR

ap = argparse.ArgumentParser()
ap.add_argument("--clients", nargs="+", default=["40", "60", "80"])
ap.add_argument("--alpha", default="0.8")
ap.add_argument("--out", default=os.path.join(ROOT, "plots", "glue", "bar"))
cli = ap.parse_args()

METRIC = "acc"
OUT_DIR = cli.out
# Round at which to evaluate each client count (0-based round index from the logs).
EVAL_ROUND = {"20": 80, "40": 145, "60": 145, "80": 145}  # 145 = last round

os.makedirs(OUT_DIR, exist_ok=True)


def per_method_avg(n_clients):
    """Return {method: avg acc across datasets} at EVAL_ROUND[n_clients]."""
    if n_clients not in EVAL_ROUND:
        raise ValueError(f"No eval round configured for n_clients={n_clients}")
    return avg_across_tasks(collect(cli.alpha, n_clients), EVAL_ROUND[n_clients])


per_setting_avg = {}
for n in cli.clients:
    avg = per_method_avg(n)
    if not avg:
        print(f"[skip] no data for n_clients={n}")
        continue
    per_setting_avg[n] = avg

    methods = order(list(avg.keys()), avg)
    heights = [avg[m] for m in methods]
    colors = [METHOD_COLOR.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(methods))
    ax.bar(xs, heights, width=0.9, color=colors, edgecolor="white", linewidth=1.0)

    for x, h in zip(xs, heights):
        ax.text(x, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=12, rotation=45)

    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=13)
    ax.set_ylabel("average test accuracy", fontsize=16)
    ax.set_title(
        rf"Average accuracy across GLUE datasets (n_clients={n})",
        fontsize=15, pad=12,
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(55, 90)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    base = os.path.join(
        OUT_DIR, f"bar_avg_across_datasets_n{n}_{METRIC}",
    )
    for ext in (".png", ".pdf"):
        fig.savefig(base + ext, dpi=300, bbox_inches="tight")
        print(f"wrote {base + ext}")
    plt.close(fig)


# Combined grouped-bar plot: one cluster per client count, one bar per method.
if len(per_setting_avg) >= 2:
    settings = list(per_setting_avg.keys())
    all_methods = set().union(*[set(d.keys()) for d in per_setting_avg.values()])
    method_means = {
        m: float(np.mean([d[m] for d in per_setting_avg.values() if m in d]))
        for m in all_methods
    }
    methods = order(list(all_methods), method_means)

    n_methods = len(methods)
    n_settings = len(settings)
    width = 0.9 / n_methods
    method_colors = [METHOD_COLOR.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_settings), 6))
    centers = np.arange(n_settings)
    for j, method in enumerate(methods):
        offsets = (j - (n_methods - 1) / 2) * width
        heights = [per_setting_avg[s].get(method, 0.0) for s in settings]
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
    ax.set_xticklabels([f"n={s}" for s in settings], fontsize=14)
    ax.set_ylabel("average test accuracy", fontsize=16)
    ax.set_title(
        "Average accuracy across GLUE datasets per number of clients",
        fontsize=15, pad=12,
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(55, 93)  # headroom so the legend clears the rotated labels
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ncol = (n_methods + 2) // 3
    ax.legend(frameon=True, loc="upper center", fontsize=11, ncol=ncol)
    fig.tight_layout()

    base = os.path.join(OUT_DIR, f"bar_avg_across_datasets_grouped_clients_{METRIC}")
    for ext in (".png", ".pdf"):
        fig.savefig(base + ext, dpi=300, bbox_inches="tight")
        print(f"wrote {base + ext}")
    plt.close(fig)
