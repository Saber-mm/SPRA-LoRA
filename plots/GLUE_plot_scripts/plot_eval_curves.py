"""Eval-accuracy curves for the runs/glue sweep, in the old plots/alpha0.8_acc
style (viridis, markers, dashed grid, ±1 std-over-seeds band, png+pdf @ 300 dpi).

One figure per (client count, alpha, task), plus one per (client count, alpha)
for the 7-task mean (avg7). Also prints final / last-5 tables (mean ± standard
error over seeds of the 7-task average).

    python plot_scripts/plot_eval_curves.py [--clients 40 60 80] [--alphas 0.8] [--out plots/glue/curves] [--no-band]
"""
import argparse, glob, json, os, re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ROOT, LABELS, METHOD_COLOR, METHOD_MARKER, order  # LABELS: (folder, lr is "map") -> display label
RUNS = os.path.join(ROOT, "runs", "glue")
MIXED_RANKS = {"4,8,16,32,64,128", "8,16,32,64,128,256"}
TASKS = ["qnli", "MRPC", "cola", "MNLI", "RTE", "sst2", "QQP"]
TASK_LABEL = {"qnli": "QNLI", "MRPC": "MRPC", "cola": "CoLA", "MNLI": "MNLI",
              "RTE": "RTE", "sst2": "SST-2", "QQP": "QQP"}
MARKERS = ["o", "s", "D", "^", "v", "p", "X", "P", "*", "h", "<", "d"]

LINESTYLE = {"SPRA-LoRA": ":", "SPRA-LoRA (uniform LR)": "--"}


def load(n_clients):
    """curves[(method_key, alpha, task)][seed] = {round: acc} for one client count."""
    curves = defaultdict(dict)
    for d in glob.glob(f"{RUNS}/*_n{n_clients}/*/*/"):
        a = json.load(open(d + "args.json"))
        if a.get("ranks") not in MIXED_RANKS:
            continue
        txt = "".join(open(f).read() for f in glob.glob(d + "*.out"))
        if "Done." not in txt:
            continue
        acc = {int(m[1]): float(m[2]) for m in re.finditer(r"Round (\d+): acc=([\d.]+)", txt)}
        folder, task = d.split("/")[-3], d.split("/")[-4].replace(f"_n{n_clients}", "")
        key = (folder, a["lr"] == "map")
        curves[(key, float(a["dirichlet_alpha"]), task)][a["seed"]] = acc
    return curves


def seed_matrix(curves, key, alpha, tasks):
    """(rounds, [n_seeds, n_rounds]) of per-seed task-mean acc; seeds present for every task."""
    by_task = [curves.get((key, alpha, t), {}) for t in tasks]
    if not all(by_task):
        return None, None
    seeds = sorted(set.intersection(*(set(bt) for bt in by_task)))
    if not seeds:
        return None, None
    rounds = sorted(set.intersection(*(set(bt[s]) for bt in by_task for s in seeds)))
    m = np.array([[np.mean([bt[s][r] for bt in by_task]) for r in rounds] for s in seeds])
    return np.array(rounds), m


def plot_one(series, out_base, band, title=None):
    """series: {display: (rounds, seeds x rounds)}"""
    fig, ax = plt.subplots(figsize=(10, 6))
    # Fixed order, color and marker per method (plot_utils), so a method looks
    # the same in every plot.
    visible = order(list(series), {l: series[l][1].mean(0)[-1] for l in series})

    for i, label in enumerate(visible):
        rounds, m = series[label]
        mean = m.mean(0)
        sd = m.std(0, ddof=1) if len(m) > 1 else np.zeros_like(mean)
        c = METHOD_COLOR.get(label, "gray")
        x = rounds + 5
        ax.plot(x, mean,
                marker=METHOD_MARKER.get(label, MARKERS[i % len(MARKERS)]),
                markevery=max(1, len(rounds) // 12),
                markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.7,
                linestyle=LINESTYLE.get(label, "-"), color=c, label=label,
                linewidth=2.2, markersize=11)
        if band:
            ax.fill_between(x, mean - sd, mean + sd, color=c, alpha=0.18, linewidth=0)

    if title:
        ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("communication round", fontsize=16)
    ax.set_ylabel("test accuracy", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax.grid(True, which="both", linestyle="--", alpha=1.0)
    # Legend lists methods by final-round mean accuracy, best first (colors and
    # markers stay fixed per method).
    handles, labels = ax.get_legend_handles_labels()
    final = {l: series[l][1].mean(0)[-1] for l in labels}
    ranked = sorted(zip(handles, labels), key=lambda hl: final[hl[1]], reverse=True)
    ax.legend([h for h, _ in ranked], [l for _, l in ranked],
              frameon=True, loc="lower right", fontsize=13 if len(visible) > 8 else 15,
              ncol=2 if len(visible) > 8 else 1)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(out_base + ext, dpi=300, bbox_inches="tight")
    print("  wrote", out_base + ".{png,pdf}")
    plt.close(fig)


def print_tables(curves, n_clients, alphas):
    """Final round and mean of the last 5 evals (rounds 125-145)."""
    for name, pick in [("final round (145)", lambda m: m[:, -1]),
                       ("mean of last 5 evals (rounds 125-145)", lambda m: m[:, -5:].mean(1))]:
        print(f"\n== n={n_clients}, {name}: avg over 7 tasks, mean ± s.e. over seeds ==")
        print(f"{'method':28s}" + "".join(f"{f'α={a:g}':>16s}" for a in alphas))
        for key, label in LABELS.items():
            vals = []
            for a in alphas:
                _, m = seed_matrix(curves, key, a, TASKS)
                if m is None:
                    vals.append(f"{'—':>16s}")
                    continue
                sc = pick(m)
                se = np.std(sc, ddof=1) / np.sqrt(len(sc)) if len(sc) > 1 else float("nan")
                vals.append(f"{np.mean(sc):8.2f} ± {se:4.2f} ({len(sc)})")
            print(f"{label:28s}" + "".join(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clients", nargs="+", default=["40", "60", "80"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.8])
    p.add_argument("--out", default=os.path.join(ROOT, "plots", "glue", "curves"))
    p.add_argument("--no-band", action="store_true")
    args = p.parse_args()
    band = not args.no_band
    sfx = "_acc_band" if band else "_acc"

    for n in args.clients:
        curves = load(n)
        for alpha in args.alphas:
            d = os.path.join(args.out, f"n{n}_alpha{alpha:g}_acc")
            for name, tasks in [(t, [t]) for t in TASKS] + [("avg7", TASKS)]:
                series = {}
                for key, label in LABELS.items():
                    x, m = seed_matrix(curves, key, alpha, tasks)
                    if x is not None:
                        series[label] = (x, m)
                if series:
                    os.makedirs(d, exist_ok=True)
                    what = ("Average test accuracy across GLUE datasets" if name == "avg7"
                            else f"Test accuracy on {TASK_LABEL[name]}")
                    title = rf"{what} ($\alpha$={alpha:g}, clients={n})"
                    plot_one(series, os.path.join(d, f"{name}_n{n}_alpha{alpha:g}{sfx}"), band, title)
        print_tables(curves, n, args.alphas)


if __name__ == "__main__":
    main()
