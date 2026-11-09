"""Shared loader for the runs/glue sweep, used by the bar-chart scripts.

Same run filter and method labels as plot_eval_curves.py: mixed-rank runs only,
finished runs only ("Done." in the SLURM .out), Frank-LoRA split by lr type.
"""
import glob
import json
import os
import re

import matplotlib.cm as cm
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # LA-LORA/
RUNS = os.path.join(ROOT, "runs", "glue")
MIXED_RANKS = {"4,8,16,32,64,128", "8,16,32,64,128,256"}

# (method folder, lr is "map") -> display label
LABELS = {
    ("Frank-LoRA-r2", True): "SPRA-LoRA (2r)",
    ("Frank-LoRA", True): "SPRA-LoRA",
    ("Frank-LoRA", False): "SPRA-LoRA (uniform LR)",
    ("FlexLoRA", False): "FlexLoRA",
    ("FLoRA", False): "FLoRA",
    ("FSLoRA", False): "FSLoRA",
    ("FFA", False): "FFA",
    # The old param-averaging "HetLoRA" runs are not plotted; the delta-averaging
    # runs (folder HetLoRA-delta) are what we call HetLoRA.
    ("HetLoRA-delta", False): "HetLoRA",
    ("RoLoRA", False): "RoLoRA",
    ("LA-LoRA", False): "LA-LoRA",
    # Personalized: server A + each client's own B, mean over sampled clients.
    ("FedSA", False): "FedSA",
}
PINNED = ["SPRA-LoRA (2r)", "SPRA-LoRA", "SPRA-LoRA (uniform LR)"]

# Fixed method order, color and marker for every plot, taken from the
# n=40, alpha=0.8 avg7 curve plot (methods ranked by final accuracy there).
METHOD_ORDER = [
    "SPRA-LoRA (2r)", "SPRA-LoRA", "SPRA-LoRA (uniform LR)", "FlexLoRA", "FLoRA",
    "HetLoRA", "FFA", "LA-LoRA", "FSLoRA", "RoLoRA", "FedSA",
]
METHOD_COLOR = dict(zip(METHOD_ORDER, cm.viridis(np.linspace(0.0, 1.05, len(METHOD_ORDER)))))
METHOD_MARKER = {
    "SPRA-LoRA (2r)": "s", "SPRA-LoRA": "s", "SPRA-LoRA (uniform LR)": "s",
    "FlexLoRA": "^", "FLoRA": "v", "HetLoRA": "p", "FFA": "X", "LA-LoRA": "P",
    "FSLoRA": ">", "RoLoRA": "h", "FedSA": "<",
}

ACC_RE = re.compile(r"Round (\d+): acc=([\d.]+)")


def collect(alpha, n_clients):
    """Return {task: {label: {round: [acc, one per seed]}}} for one
    (dirichlet alpha, client count) setting."""
    data = {}
    for run_dir in glob.glob(os.path.join(RUNS, f"*_n{n_clients}", "*", "*", "")):
        args_path = os.path.join(run_dir, "args.json")
        if not os.path.exists(args_path):
            continue
        with open(args_path) as f:
            a = json.load(f)
        if a.get("ranks") not in MIXED_RANKS:
            continue
        if float(a["dirichlet_alpha"]) != float(alpha):
            continue
        txt = ""
        for out in glob.glob(os.path.join(run_dir, "*.out")):
            with open(out) as f:
                txt += f.read()
        if "Done." not in txt:
            continue
        label = LABELS.get((a.get("label", a["alg"]), a["lr"] == "map"))
        if label is None:
            continue
        per_round = data.setdefault(a["task"], {}).setdefault(label, {})
        for m in ACC_RE.finditer(txt):
            per_round.setdefault(int(m[1]), []).append(float(m[2]))
    return data


def value_at_round(per_round, target_r):
    """Seed-mean accuracy at target_r, or at the closest earlier round with data.
    Returns None if there is no round <= target_r."""
    rounds = [r for r in per_round if r <= target_r]
    if not rounds:
        return None
    vals = per_round[max(rounds)]
    return sum(vals) / len(vals)


def avg_across_tasks(data, target_r):
    """{label: mean over tasks of the seed-mean acc at target_r}; keeps
    methods that have data for at least 2 tasks."""
    by_method = {}
    for task, curves in data.items():
        for label, per_round in curves.items():
            v = value_at_round(per_round, target_r)
            if v is not None:
                by_method.setdefault(label, []).append(v)
    return {m: sum(vs) / len(vs) for m, vs in by_method.items() if len(vs) >= 2}


def order(methods, values=None):
    """Methods in the fixed METHOD_ORDER; any unknown ones go last, by value."""
    known = [m for m in METHOD_ORDER if m in methods]
    rest = sorted((m for m in methods if m not in METHOD_ORDER),
                  key=lambda m: values[m] if values else 0, reverse=True)
    return known + rest
