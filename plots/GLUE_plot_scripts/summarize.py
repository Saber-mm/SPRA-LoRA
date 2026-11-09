#!/usr/bin/env python3
"""Summarize LA-LORA FL experiment results from a runs directory.

Usage:
    python plot_scripts/summarize.py --dir runs/glue/sst2_n40 --max-rounds 100
    python plot_scripts/summarize.py --dir runs/glue --finished-only --n 40 --alpha 0.8 --latex

Directory structure expected:
    <dir>/<method>/<run_name>/*.out                        (single dataset)
    <dir>/[subdirs/]<dataset>/<method>/<run_name>/*.out    (multiple datasets)

Config is read from the run's args.json when present; falls back to
parsing run_name when absent so old runs still work.

Output is one combined table per (n, alpha) group: rows = methods, columns =
datasets, plus a GPU-hours column and an Avg column. Each cell is the mean over
seeds of the best config/lr for that seed, +- the standard error over seeds.

--r2-dir adds a second root of doubled-rank Frank-LoRA runs. Its datasets are
matched to the main ones by directory label (which carries the dataset name and
the _nX suffix), and alpha comes from each run's args.json, so results land in
the table for their own n and alpha as extra "doubled ranks" rows:

    python plot_scripts/summarize.py --dir runs/glue --last --r2-dir <doubled-rank runs dir>

print_dataset_table() below still renders the old per-config / per-seed / per-lr
breakdown, but nothing calls it now that only combined tables are printed.
"""

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict


# ---------------------------------------------------------------------------
# Log / file readers
# ---------------------------------------------------------------------------

def read_out_file(run_dir):
    """Return ({round: acc}, runtime_h) from .out / .txt log files in run_dir.

    Round/acc lines appear in both file types; ``Total wall time`` is only in .txt.
    """
    values = {}
    runtime_h = None
    for fname in os.listdir(run_dir):
        if not (fname.endswith(".out") or fname.endswith(".txt")):
            continue
        path = os.path.join(run_dir, fname)
        try:
            with open(path) as f:
                for line in f:
                    m = re.search(r"Round (\d+): acc=([\d.]+)", line)
                    if m:
                        values[int(m.group(1))] = float(m.group(2))
                    m_t = re.search(r"Total wall time: ([\d.]+) h", line)
                    if m_t:
                        runtime_h = float(m_t.group(1))
        except Exception:
            pass
    return values, runtime_h


def best_entry(values):
    if not values:
        return None, None
    step = max(values, key=values.__getitem__)
    return step, values[step]


def last_entry(values):
    if not values:
        return None, None
    step = max(values)
    return step, values[step]


def check_done(run_dir):
    for fname in os.listdir(run_dir):
        if not fname.endswith(".out"):
            continue
        try:
            with open(os.path.join(run_dir, fname)) as f:
                if "Done." in f.read():
                    return True
        except Exception:
            pass
    return False


# Old args.json keys / method names -> current ones, so runs from before the
# rename land in the same rows and configs as new ones.
_KEY_ALIASES = {
    "num_workers": "num_clients", "selection": "client_frac", "K": "local_steps",
    "epoch": "rounds", "alpha_value": "dirichlet_alpha", "cnn": "model",
}
_METHOD_ALIASES = {"FFA+": "Frank-LoRA"}
# Old param-averaging HetLoRA runs are left out of every table; the
# delta-averaging runs (label HetLoRA-delta) are reported as "HetLoRA".
_HIDDEN_METHODS = {"HetLoRA"}


def canonical_method(name):
    return _METHOD_ALIASES.get(name, name)


def read_args_json(run_dir):
    """Return dict from args.json in run_dir (old keys renamed), or None if missing/unreadable."""
    path = os.path.join(run_dir, "args.json")
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    return {_KEY_ALIASES.get(k, k): v for k, v in d.items()}


# ---------------------------------------------------------------------------
# Hyperparameter extraction
# ---------------------------------------------------------------------------

def parse_hyperparams(run_name):
    """Return (rw, lrr, rank, cfg, lr) extracted from run_name (fallback path)."""
    m_lr = re.search(r"_lr([\d.e+-]+)(.*)", run_name)
    lr = m_lr.group(1) if m_lr else "?"
    suffix = m_lr.group(2) if m_lr else ""

    lrr = bool(re.search(r"_lrr(?:_|$)", suffix))

    rank = ""
    m_r = re.match(r"_(r\d+)", suffix)
    if m_r:
        rank = m_r.group(1)

    stem = re.sub(r"_lr[\d.e+-]+.*", "", run_name)
    parts = stem.split("_")
    start = next((i for i, p in enumerate(parts) if "=" in p), len(parts))
    hyperparam_parts = parts[start:]

    rw = ""
    cfg_parts = []
    for p in hyperparam_parts:
        if p.startswith("rw="):
            rw = p[len("rw="):]
        else:
            cfg_parts.append(p)

    cfg = "_".join(cfg_parts)
    return rw, lrr, rank, cfg, lr


# Fields used to build a config key (seed and lr are intentionally excluded).
_CFG_FIELDS = (
    "lora_alpha", "ranks", "num_clients", "client_frac", "local_steps",
    "dirichlet_alpha", "lora_scaling", "rank_weight",
)


def _norm_alpha(a):
    """Canonicalize dirichlet_alpha so 2, 2.0, "2", "2.0" all map to "2"."""
    if a is None or a == "":
        return ""
    try:
        f = float(a)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(a)


def cfg_key_from_args(d):
    """Hashable config key (excludes seed and lr) from args.json dict."""
    return tuple(
        _norm_alpha(d.get(f, "")) if f == "dirichlet_alpha" else str(d.get(f, ""))
        for f in _CFG_FIELDS
    )


def cfg_label_from_args(d):
    """Human-readable config label (excludes seed and lr)."""
    parts = []
    if "lora_alpha" in d:
        parts.append(f"la={d['lora_alpha']}")
    if "ranks" in d:
        max_r = max(int(x) for x in str(d["ranks"]).split(","))
        parts.append(f"r{max_r}")
    if "num_clients" in d:
        parts.append(f"n{d['num_clients']}")
    if "client_frac" in d:
        parts.append(f"q{d['client_frac']}")
    if "local_steps" in d:
        parts.append(f"K{d['local_steps']}")
    if "dirichlet_alpha" in d:
        parts.append(f"alpha{_norm_alpha(d['dirichlet_alpha'])}")
    if d.get("lora_scaling"):
        parts.append(f"ls={d['lora_scaling']}")
    return "_".join(parts)


def row_key_from_args(d):
    """Return (method, rw, lrr, lr_tag) from args.json dict.

    lr_tag is non-empty for special lr values (e.g. "map") that should appear
    as their own method row rather than as a swept hyperparameter.
    """
    method = d.get("label") or canonical_method(d.get("alg", "?"))
    rw = str(d.get("rank_weight", "")) if d.get("rank_weight") else ""
    lrr = bool(d.get("lr_rescale", False))
    lr = str(d.get("lr", ""))
    lr_tag = lr if lr == "map" else ""
    return (method, rw, lrr, lr_tag)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_val(val, step, finished, max_rounds=None):
    if val is None:
        return "n/a"
    short = max_rounds is not None and step < max_rounds
    suffix = "~" if short else ("" if finished else "*")
    return f"{val:.2f}{suffix}"


def fmt_mean(mean, std, n_seeds, finished, max_rounds=None, avg_step=None):
    """Format aggregated value. Single seed: plain value; multi-seed: mean±std (Ns)."""
    if mean is None:
        return "n/a"
    suffix = "" if finished else "*"
    if n_seeds == 1:
        short = max_rounds is not None and avg_step is not None and avg_step < max_rounds
        suffix = "~" if short else suffix
        return f"{mean:.2f}{suffix}"
    std_str = f"±{std:.2f}" if std is not None else ""
    return f"{mean:.2f}{std_str} ({n_seeds}s){suffix}"


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def is_dataset_dir(path):
    """True if path is a leaf dataset dir (method/run_name/*.out exists under it)."""
    try:
        for method in os.listdir(path):
            method_dir = os.path.join(path, method)
            if not os.path.isdir(method_dir):
                continue
            for run_name in os.listdir(method_dir):
                run_dir = os.path.join(method_dir, run_name)
                if not os.path.isdir(run_dir):
                    continue
                if any(f.endswith(".out") for f in os.listdir(run_dir)):
                    return True
    except PermissionError:
        pass
    return False


_DATASET_ORDER = {d: i for i, d in enumerate([
    "qnli", "MRPC", "cola", "MNLI", "RTE", "sst2", "QQP",
])}

_DATASET_DISPLAY = {
    "qnli": "QNLI", "MRPC": "MRPC", "cola": "CoLA", "MNLI": "MNLI",
    "RTE": "RTE", "sst2": "SST-2", "QQP": "QQP",
}


def _dir_sort_key(name):
    base = re.sub(r"_n\d+$", "", name)
    rank = _DATASET_ORDER.get(base, _DATASET_ORDER.get(name, len(_DATASET_ORDER)))
    return (rank, name)


def find_dataset_dirs(root):
    """Return [(label, abs_path)] for all leaf dataset dirs under root."""
    if is_dataset_dir(root):
        return [(".", root)]
    results = []
    try:
        for entry in sorted(os.listdir(root), key=_dir_sort_key):
            path = os.path.join(root, entry)
            if not os.path.isdir(path):
                continue
            for rel, abs_path in find_dataset_dirs(path):
                label = entry if rel == "." else os.path.join(entry, rel)
                results.append((label, abs_path))
    except PermissionError:
        pass
    return results


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def process_dataset_dir(root, max_rounds=None, use_last=False, finished_only=False):
    """
    Scan a leaf dataset dir and return:
      seed_best    : {(row, cfg_key, seed): (best_val, best_step, finished, best_lr, runtime_h)}
                     where best_lr is chosen by averaging over seeds first
      lr_data      : {(row, cfg_key, seed): {lr: (val, step, finished, runtime_h)}}
      cfg_labels   : {cfg_key: human-readable label}
      cfg_alphas   : {cfg_key: dirichlet_alpha string}
      configs      : sorted list of cfg_keys
      row_keys     : sorted list of row keys
      incomplete   : list of run_dir paths that are not finished
    """
    # lr_data[(row, cfg_key, seed)][lr] = (val, step, finished, runtime_h)
    lr_data = defaultdict(dict)
    cfg_labels = {}
    cfg_alphas = {}
    row_keys = set()
    cfg_keys = set()
    incomplete = []

    for method in sorted(os.listdir(root)):
        method_dir = os.path.join(root, method)
        if not os.path.isdir(method_dir):
            continue
        for run_name in sorted(os.listdir(method_dir)):
            run_dir = os.path.join(method_dir, run_name)
            if not os.path.isdir(run_dir):
                continue

            step_values, runtime_h = read_out_file(run_dir)
            if max_rounds is not None:
                step_values = {s: v for s, v in step_values.items() if s <= max_rounds}
            step, val = (last_entry if use_last else best_entry)(step_values)
            finished = check_done(run_dir)
            if not finished:
                incomplete.append(run_dir)
                if finished_only:
                    continue

            d = read_args_json(run_dir)
            if d is not None:
                row = row_key_from_args(d)
                cfg_key = cfg_key_from_args(d)
                cfg_label = cfg_label_from_args(d)
                seed = d.get("seed", 42)
                lr = str(d.get("lr", "?"))
            else:
                # Fallback: parse everything from the folder name.
                rw, lrr, rank, cfg, lr = parse_hyperparams(run_name)
                lr_tag = lr if lr == "map" else ""
                row = (canonical_method(method), rw, lrr, lr_tag)
                # Pad to _CFG_FIELDS length so cfg_keys are comparable.
                cfg_key = (cfg, rank) + ("",) * (len(_CFG_FIELDS) - 2)  # pad to _CFG_FIELDS length
                cfg_label = f"{cfg}[{rank}]" if rank else cfg
                seed = 42

            if row[0] in _HIDDEN_METHODS:
                continue
            row_keys.add(row)
            cfg_keys.add(cfg_key)
            cfg_labels[cfg_key] = cfg_label
            if d is not None and "dirichlet_alpha" in d:
                cfg_alphas[cfg_key] = _norm_alpha(d["dirichlet_alpha"])

            run_key = (row, cfg_key, seed)
            prev = lr_data[run_key].get(lr)
            if val is not None and (prev is None or prev[0] is None or val > prev[0]):
                lr_data[run_key][lr] = (val, step, finished, runtime_h)
            elif prev is None:
                lr_data[run_key][lr] = (val, step, finished, runtime_h)

    # Find best lr per (row, cfg_key) by averaging over seeds first.
    lr_vals_by_cfg = defaultdict(lambda: defaultdict(list))
    for (row, cfg_key, seed), lrs in lr_data.items():
        for lr, (val, step, fin, rt) in lrs.items():
            if val is not None:
                lr_vals_by_cfg[(row, cfg_key)][lr].append(val)

    best_lr_for_cfg = {}
    for (row, cfg_key), lr_vals in lr_vals_by_cfg.items():
        if not lr_vals:
            continue
        best_lr = max(lr_vals, key=lambda lr: sum(lr_vals[lr]) / len(lr_vals[lr]))
        best_lr_for_cfg[(row, cfg_key)] = best_lr

    # Reduce lr_data -> best-per-seed using the globally best lr.
    seed_best = {}
    for (row, cfg_key, seed), lrs in lr_data.items():
        best_lr = best_lr_for_cfg.get((row, cfg_key))
        if best_lr is not None and best_lr in lrs:
            val, step, fin, rt = lrs[best_lr]
            seed_best[(row, cfg_key, seed)] = (val, step, fin, best_lr, rt)
        else:
            # This seed has no data for the best lr; record as missing but preserve finish state.
            any_unfinished = any(not fin for _, _, fin, _ in lrs.values())
            seed_best[(row, cfg_key, seed)] = (None, None, not any_unfinished, best_lr, None)

    return seed_best, lr_data, cfg_labels, cfg_alphas, sorted(cfg_keys), sorted(row_keys), incomplete


def seed_summary(seed_best, row, cfg_key):
    """Return {seed: (val, step, fin, best_lr)} for a (row, cfg_key) pair."""
    return {
        seed: entry
        for (r, c, seed), entry in seed_best.items()
        if r == row and c == cfg_key
    }


def agg_seeds(per_seed):
    """Aggregate {seed: (val, step, fin, lr, runtime_h)} -> (mean, std, n, all_fin, avg_step, avg_runtime_h)."""
    vals = [(v, s) for v, s, _, _, _ in per_seed.values() if v is not None]
    if not vals:
        return None, None, 0, False, None, None
    vs = [v for v, _ in vals]
    ss = [s for _, s in vals]
    mean = sum(vs) / len(vs)
    std = statistics.stdev(vs) if len(vs) > 1 else None
    avg_step = round(sum(ss) / len(ss))
    all_fin = all(fin for _, _, fin, _, _ in per_seed.values())
    rts = [rt for _, _, _, _, rt in per_seed.values() if rt is not None]
    avg_rt = sum(rts) / len(rts) if rts else None
    return mean, std, len(vs), all_fin, avg_step, avg_rt


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

# Display ordering: (method, lr_tag) -> (sort index, display label).
# Rows that don't match fall back to default sort and the original label format.
_ROW_DISPLAY = [
    (("FFA",      ""),    "FFA"),
    (("HetLoRA-delta", ""), "HetLoRA"),
    (("FlexLoRA", ""),    "FlexLoRA"),
    (("FLoRA",    ""),    "FLoRA"),
    (("FSLoRA",   ""),    "FSLoRA"),
    (("FedSA",    ""),    "FedSA"),
    (("RoLoRA",   ""),    "RoLoRA"),
    (("LA-LoRA",  ""),    "LA-LoRA"),
    (("Frank-LoRA",   ""),    "SPRA-LoRA (uniform LR)"),
    (("Frank-LoRA",   "map"), "SPRA-LoRA"),
    (("Frank-LoRA-r2", "map"), "SPRA-LoRA (doubled ranks)"),
]
_ROW_INDEX = {k: i for i, (k, _) in enumerate(_ROW_DISPLAY)}
_ROW_LABEL_OVERRIDE = {k: v for k, v in _ROW_DISPLAY}


def _row_label(m, rw, lrr, lr_tag=""):
    override = _ROW_LABEL_OVERRIDE.get((m, lr_tag))
    if override and not rw and not lrr:
        return override
    parts = [m]
    if rw:
        parts.append(f"rw={rw}")
    if lrr:
        parts.append("lrr")
    if lr_tag:
        parts.append(lr_tag)
    return " ".join(parts)


def _row_sort_key(row):
    m, rw, lrr, lr_tag = row
    idx = _ROW_INDEX.get((m, lr_tag), len(_ROW_INDEX))
    return (idx, m, rw, lrr, lr_tag)


def print_dataset_table(seed_best, lr_data, cfg_labels, configs, row_keys, max_rounds=None, tsv=False):
    """Print per-config tables for a single dataset."""
    INDENT = "    "
    rows = sorted(row_keys, key=_row_sort_key)
    row_labels = [_row_label(*row) for row in rows]

    if tsv:
        print("\t".join(["config", "method", "mean", "std", "n_seeds", "runtime_h", "best_lr"]))
        for cfg_key in configs:
            cfg_label = cfg_labels.get(cfg_key, str(cfg_key))
            for row, label in zip(rows, row_labels):
                per_seed = seed_summary(seed_best, row, cfg_key)
                mean, std, n, all_fin, avg_step, avg_rt = agg_seeds(per_seed)
                suffix = "" if all_fin else "*"
                mean_str = f"{mean:.2f}{suffix}" if mean is not None else ""
                std_str = f"{std:.2f}" if std is not None else ""
                rt_str = f"{avg_rt:.2f}" if avg_rt is not None else ""
                best_lrs = sorted({lr for _, _, _, lr, _ in per_seed.values() if lr is not None})
                print("\t".join([cfg_label, label, mean_str, std_str, str(n), rt_str, ",".join(best_lrs)]))
        print()
        return

    row_w = max((len(l) for l in row_labels), default=6)
    row_w = max(row_w, len("method"))
    val_w = len("000.00±0.00 (00s)*")
    rt_w = len("runtime")
    cfg_w = max((len(cfg_labels.get(k, "")) for k in configs), default=8)
    cfg_w = max(cfg_w, len("best lr"))

    for cfg_key in configs:
        cfg_label = cfg_labels.get(cfg_key, str(cfg_key))
        header = f"{'method':<{row_w}}  {'value':>{val_w}}  {'runtime':>{rt_w}}  {'best lr':<{cfg_w}}"
        sep = "-" * len(header)
        print()
        print(cfg_label)
        print(sep)
        print(header)
        print(sep)

        for row, label in zip(rows, row_labels):
            per_seed = seed_summary(seed_best, row, cfg_key)
            mean, std, n, all_fin, avg_step, avg_rt = agg_seeds(per_seed)
            val_cell = fmt_mean(mean, std, n, all_fin, max_rounds, avg_step)
            rt_cell = f"{avg_rt:.2f}h" if avg_rt is not None else "—"
            best_lrs = sorted({lr for _, _, _, lr, _ in per_seed.values() if lr is not None})
            lr_cell = ",".join(best_lrs) if best_lrs else ""
            print(f"{label:<{row_w}}  {val_cell:>{val_w}}  {rt_cell:>{rt_w}}  {lr_cell:<{cfg_w}}")

        print(sep)

        # Per-seed breakdown: one table per seed, rows = methods, cols = lr values
        all_seed_lr = {
            (r, seed, lr): entry
            for (r, c, seed), lr_dict in lr_data.items()
            if c == cfg_key
            for lr, entry in lr_dict.items()
        }
        if not all_seed_lr:
            continue
        all_seeds = sorted({seed for (_, seed, _) in all_seed_lr})
        all_lrs = sorted({lr for (_, _, lr) in all_seed_lr})
        if len(all_seeds) <= 1 and len(all_lrs) <= 1:
            continue
        bk_row_w = max(len(l) for l in row_labels)
        bk_row_w = max(bk_row_w, len("method"))
        bk_cell_w = max(len("000.00*"), max(len(lr) for lr in all_lrs))
        for seed in all_seeds:
            print()
            print(f"{INDENT}seed={seed}")
            header_cells = [f"{'method':<{bk_row_w}}"] + [f"  {lr:>{bk_cell_w}}" for lr in all_lrs]
            tbl_header = INDENT + "".join(header_cells)
            tbl_sep = INDENT + "-" * (len(tbl_header) - len(INDENT))
            print(tbl_sep)
            print(tbl_header)
            print(tbl_sep)
            for row, label in zip(rows, row_labels):
                row_cells = [f"{label:<{bk_row_w}}"]
                for lr in all_lrs:
                    entry = all_seed_lr.get((row, seed, lr))
                    if entry is None:
                        cell = "—"
                    else:
                        val, step, fin, _rt = entry
                        cell = fmt_val(val, step, fin, max_rounds) if val is not None else "n/a"
                    row_cells.append(f"  {cell:>{bk_cell_w}}")
                print(INDENT + "".join(row_cells))
            print(tbl_sep)


def best_per_row(seed_best, row_keys, cfg_alphas=None, alpha_filter=None):
    """Return {row: (mean, all_fin, mean_runtime_h, sem, n_seeds)} taking best config per seed, then averaging.

    sem is the standard error of that mean (stdev over seeds / sqrt(n_seeds)),
    so it shrinks as seeds are added; None when there is only one seed.
    """
    row_seed_vals = defaultdict(dict)
    row_seed_rts = defaultdict(dict)
    row_seed_fin = defaultdict(lambda: defaultdict(lambda: True))

    for (row, cfg_key, seed), (val, step, fin, lr, rt) in seed_best.items():
        if alpha_filter is not None and cfg_alphas is not None:
            if cfg_alphas.get(cfg_key) != alpha_filter:
                continue
        prev = row_seed_vals[row].get(seed)
        if val is not None and (prev is None or val > prev):
            row_seed_vals[row][seed] = val
            row_seed_rts[row][seed] = rt
        if not fin:
            row_seed_fin[row][seed] = False

    result = {}
    for row in row_keys:
        if row not in row_seed_vals or not row_seed_vals[row]:
            result[row] = (None, True, None, None, 0)
            continue
        vals = [v for v in row_seed_vals[row].values() if v is not None]
        mean = sum(vals) / len(vals) if vals else None
        all_fin = all(row_seed_fin[row][s] for s in row_seed_vals[row])
        rts = [r for r in row_seed_rts[row].values() if r is not None]
        mean_rt = sum(rts) / len(rts) if rts else None
        sem = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else None
        result[row] = (mean, all_fin, mean_rt, sem, len(vals))
    return result


# (method, lr_tag) pairs pulled out of an --r2-dir, and the suffix that keeps them
# on their own rows instead of merging into the same method's rows from the main
# --dir. Uniform-LR runs in the r2 tree are ignored — only the mapped-LR variant.
_R2_ROWS = {("Frank-LoRA", "map")}
_R2_SUFFIX = "-r2"


def merge_r2_result(main, r2):
    """Return a copy of ``main`` with ``r2``'s Frank-LoRA runs added as extra rows.

    Both arguments are process_dataset_dir() tuples for the *same* dataset (same
    name, hence same n) from two different roots. Only _R2_ROWS are taken; their
    method name gets _R2_SUFFIX appended so they sort and label as their own
    rows. alpha comes along inside each run's config, so the per-alpha combined
    tables pick these up automatically.
    """
    seed_best, lr_data, cfg_labels, cfg_alphas, configs, row_keys, incomplete = main
    r2_seed_best, _, r2_cfg_labels, r2_cfg_alphas, _, _, r2_incomplete = r2

    seed_best = dict(seed_best)
    cfg_labels = dict(cfg_labels)
    cfg_alphas = dict(cfg_alphas)
    row_keys = list(row_keys)
    cfg_keys = set(configs)

    for (row, cfg_key, seed), entry in r2_seed_best.items():
        if (row[0], row[3]) not in _R2_ROWS:
            continue
        new_row = (row[0] + _R2_SUFFIX,) + tuple(row[1:])
        seed_best[(new_row, cfg_key, seed)] = entry
        if new_row not in row_keys:
            row_keys.append(new_row)
        cfg_keys.add(cfg_key)
        if cfg_key in r2_cfg_labels:
            cfg_labels[cfg_key] = r2_cfg_labels[cfg_key]
        if cfg_key in r2_cfg_alphas:
            cfg_alphas[cfg_key] = r2_cfg_alphas[cfg_key]

    return (seed_best, lr_data, cfg_labels, cfg_alphas, sorted(cfg_keys),
            sorted(row_keys), incomplete + r2_incomplete)


def avg_stderr(per_dataset_stats):
    """Standard error of the mean over datasets.

    per_dataset_stats: [(sem, n_seeds)] for every dataset contributing to the
    average, with sem=None for datasets that only have one seed.

    The mean over D dataset means has variance sum(sem^2)/D^2. Single-seed
    datasets have no sem of their own; they borrow the seed stdev pooled across
    the other datasets, so a dataset crossing 1 -> 2 seeds doesn't make the
    error bar jump. D is the full count of datasets in the average, matching
    the mean itself.
    """
    if not per_dataset_stats:
        return None
    known = [(sem, n) for sem, n in per_dataset_stats if sem is not None]
    if not known:
        return None
    # sem = sd / sqrt(n), so recover each sd to pool them on a common scale.
    pooled_sd = sum(sem * n ** 0.5 for sem, n in known) / len(known)
    var = sum(sem ** 2 if sem is not None else pooled_sd ** 2 / max(n, 1)
              for sem, n in per_dataset_stats)
    return var ** 0.5 / len(per_dataset_stats)


_LATEX_ROW_GROUPS = [
    [("FFA", ""), ("HetLoRA-delta", ""), ("FlexLoRA", ""), ("FLoRA", ""), ("FSLoRA", ""),
     ("FedSA", ""), ("RoLoRA", ""), ("LA-LoRA", "")],
    [("Frank-LoRA", ""), ("Frank-LoRA", "map"), ("Frank-LoRA-r2", "map")],
]
_LATEX_DISPLAY = {
    ("FFA", ""): r"\algname{FFA} [\citeyear{FFA}]",
    ("HetLoRA-delta", ""): r"\algname{HetLoRA} [\citeyear{HetLoRA}]",
    ("FlexLoRA", ""): r"\algname{FlexLoRA} [\citeyear{Flexlora}]",
    ("FLoRA", ""): r"\algname{FLoRA} [\citeyear{flora}]",
    ("FSLoRA", ""): r"\algname{FSLoRA} [\citeyear{Fslora}]",
    ("FedSA", ""): r"\algname{FedSA} [\citeyear{Fedsa}]",
    ("RoLoRA", ""): r"\algname{RoLoRA} [\citeyear{RoLoRA}]",
    ("LA-LoRA", ""): r"\algname{LA-LoRA} [\citeyear{LALoRA}]",
    ("Frank-LoRA", ""): r"\algname{SPRA-LoRA} (uniform LR)",
    ("Frank-LoRA", "map"): r"\rowcolor{lightgray}\algname{SPRA-LoRA}",
    ("Frank-LoRA-r2", "map"): r"\rowcolor{lightgray}\algname{SPRA-LoRA} (doubled ranks)",
}
# Rows whose numbers should be \textbf bolded (when they lead their column).
_LATEX_HIGHLIGHT = {("Frank-LoRA", "map"), ("Frank-LoRA-r2", "map")}


def _latex_cell(val, all_fin, sem=None, bold=False, underline=False):
    if val is None:
        return "—"
    s = f"{val:.2f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    if underline:
        s = rf"\underline{{{s}}}"
    if sem is not None:
        s = s + rf"$_{{\pm {sem:.2f}}}$"
    return s


def _alpha_label_tag(alpha):
    """Format alpha for LaTeX label, e.g. 0.4 -> '04', 2.0 -> '20'."""
    if alpha is None or alpha == "":
        return "x"
    try:
        af = float(alpha)
        if abs(af * 10 - round(af * 10)) < 1e-9:
            return f"{int(round(af * 10)):02d}"
    except (ValueError, TypeError):
        pass
    return str(alpha).replace(".", "")


_BASELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "runs", "evals")
_BASELINE_RE = re.compile(r"Round -1 \(baseline\): acc=([\d.]+)")


def baseline_accs(display_labels):
    """Pretrained-model (no training) accuracy per dataset column, from the
    runs/evals logs written by baseline_eval.py; None where there is no log."""
    task_for = {v: k for k, v in _DATASET_DISPLAY.items()}
    out = []
    for label in display_labels:
        task = task_for.get(label, label)
        acc = None
        for log in glob.glob(os.path.join(_BASELINE_DIR, task, "*", "*.txt")):
            with open(log) as f:
                hits = _BASELINE_RE.findall(f.read())
            if hits:
                acc = float(hits[-1])
        out.append(acc)
    return out


def _second_best(vals):
    """Second-highest distinct value at the printed precision (2 dp), or None."""
    distinct = sorted({round(v, 2) for v in vals}, reverse=True)
    return distinct[1] if len(distinct) > 1 else None


def _same_2dp(val, target):
    return target is not None and round(val, 2) == target


def print_combined_table_latex(display_labels, all_rows, per_dataset, row_avg, row_avg_rt, row_all_fin, textbf=False, alpha=None, n=None):
    bpr_lookup = {row: [bpr.get(row) for bpr in per_dataset] for row in all_rows}

    # Per-column max across all rows, for conditional bolding of highlighted rows.
    col_maxes = []
    col_seconds = []
    for di in range(len(per_dataset)):
        col_vals = [bpr_lookup[r][di][0] for r in all_rows
                    if bpr_lookup[r][di] is not None and bpr_lookup[r][di][0] is not None]
        col_maxes.append(max(col_vals) if col_vals else None)
        col_seconds.append(_second_best(col_vals))
    avg_vals = [row_avg(r) for r in all_rows if row_avg(r) is not None]
    avg_max = max(avg_vals) if avg_vals else None
    avg_second = _second_best(avg_vals)

    def emit(row):
        m, rw, lrr, lr_tag = row
        label = _LATEX_DISPLAY.get((m, lr_tag), _row_label(*row))
        bold_row = textbf and (m, lr_tag) in _LATEX_HIGHLIGHT
        avg_rt = row_avg_rt(row)
        rt_cell = f"{avg_rt:.2f}h" if avg_rt is not None else ""
        cells = [label, rt_cell]
        avg_terms = []
        for di, entry in enumerate(bpr_lookup[row]):
            if entry is None or entry[0] is None:
                cells.append("—")
            else:
                val, fin, _, sem, n_seeds = entry
                is_best = col_maxes[di] is not None and abs(val - col_maxes[di]) < 1e-9
                is_second = textbf and _same_2dp(val, col_seconds[di])
                cells.append(_latex_cell(val, fin, sem=sem, bold=bold_row and is_best,
                                         underline=is_second))
                avg_terms.append((sem, n_seeds))
        avg = row_avg(row)
        avg_sem = avg_stderr(avg_terms)
        avg_is_best = avg is not None and avg_max is not None and abs(avg - avg_max) < 1e-9
        avg_is_second = textbf and avg is not None and _same_2dp(avg, avg_second)
        cells.append(_latex_cell(avg, row_all_fin(row), sem=avg_sem, bold=bold_row and avg_is_best,
                                 underline=avg_is_second))
        print(" & ".join(cells) + r" \\")

    n_data = len(display_labels)
    col_spec = "c|c|" + "c" * n_data + "|c"
    tag = _alpha_label_tag(alpha)
    alpha_str = (f"{float(alpha):.1f}" if alpha is not None and alpha != "" else "?")

    print(r"\begin{table*}[t!]")
    print(r"\centering")
    print(r"\setlength\tabcolsep{2pt}")
    print(r"\begin{minipage}[c]{1\textwidth}")
    print(r"\centering")
    n_tag = f"_n{n}" if n is not None else ""
    n_cap = n if n is not None else "?"
    print(r"\caption{Test accuracy averaged over 5 runs on GLUE benchmark with Dirichlet distribution "
          rf"($\mathbf{{\alpha={alpha_str}}}$), \texttt{{RoBERTa}} model, \texttt{{AdamW}} optimizer, "
          rf"150 communication rounds, and $\mathbf{{n={n_cap}}}$ clients. Clients' ranks follow "
          r"\textbf{\emph{uniform}} distribution over $\{\texttt{4}, \texttt{8}, \dots, \texttt{128}\}$.}")
    print(rf"\label{{tab:iid_alpha_{tag}{n_tag}}}")
    print(r"\resizebox{\textwidth}{!}{")
    print(rf"\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")
    header = (r"\bf{Method}& Run time & "
              + " & ".join(rf"\bf{{{d}}}" for d in display_labels)
              + r"& \bf{Avg} % & VPF")
    print(header)
    print(r"\\ ")
    print(r"\toprule")
    base = baseline_accs(display_labels)
    if all(v is not None for v in base):
        base_cells = [f"{v:.2f}" for v in base] + [f"{sum(base) / len(base):.2f}"]
    else:
        base_cells = ["" if v is None else f"{v:.2f}" for v in base] + [""]
    print(r"\algname{Base Pretrained Model}& --- & " + " & ".join(base_cells) + r" \\")
    print(r"\midrule")

    present = set(all_rows)
    for gi, group in enumerate(_LATEX_ROW_GROUPS):
        if gi > 0:
            print(r"\midrule")
        for key in group:
            row = next((r for r in all_rows if (r[0], r[3]) == key), None)
            if row is not None:
                emit(row)
                present.discard(row)
    # Any leftover rows not in groups
    for row in all_rows:
        if row in present:
            emit(row)

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{minipage}\hfill")
    print(rf"\label{{table:iid_alpha_{tag}{n_tag}}}")
    print(r"\end{table*}")


def _md_cell(val, all_fin, sem=None, bold=False):
    if val is None:
        return "—"
    s = f"{val:.2f}"
    if sem is not None:
        s += f"±{sem:.2f}"
    if not all_fin:
        s += "*"
    return f"**{s}**" if bold else s


def print_combined_table_markdown(display_labels, all_rows, row_labels, per_dataset,
                                  row_avg, row_avg_rt, row_all_fin, textbf=False):
    """Print the combined table as a GitHub-flavored markdown pipe table."""
    # Per-column max across all rows, for conditional bolding of highlighted rows.
    col_maxes = []
    for di in range(len(per_dataset)):
        col_vals = [per_dataset[di][r][0] for r in all_rows
                    if per_dataset[di].get(r) is not None and per_dataset[di][r][0] is not None]
        col_maxes.append(max(col_vals) if col_vals else None)
    avg_vals = [row_avg(r) for r in all_rows if row_avg(r) is not None]
    avg_max = max(avg_vals) if avg_vals else None

    print("| " + " | ".join(["Method", "GPU hours"] + display_labels + ["Avg"]) + " |")
    print("| " + " | ".join([":---"] + ["---:"] * (len(display_labels) + 2)) + " |")

    for row, label in zip(all_rows, row_labels):
        bold_row = textbf and (row[0], row[3]) in _LATEX_HIGHLIGHT
        avg_rt = row_avg_rt(row)
        cells = [label, f"{avg_rt:.2f}h" if avg_rt is not None else "—"]
        avg_terms = []
        for di, bpr in enumerate(per_dataset):
            entry = bpr.get(row)
            if entry is None or entry[0] is None:
                cells.append("—")
                continue
            val, all_fin, _, sem, n_seeds = entry
            is_best = col_maxes[di] is not None and abs(val - col_maxes[di]) < 1e-9
            cells.append(_md_cell(val, all_fin, sem, bold=bold_row and is_best))
            avg_terms.append((sem, n_seeds))
        avg = row_avg(row)
        avg_is_best = avg is not None and avg_max is not None and abs(avg - avg_max) < 1e-9
        cells.append(_md_cell(avg, row_all_fin(row), avg_stderr(avg_terms),
                              bold=bold_row and avg_is_best))
        print("| " + " | ".join(cells) + " |")


def print_combined_table(dataset_labels, results_list, tsv=False, alpha_filter=None, latex=False, textbf=False, n=None, markdown=False):
    """Print combined table: rows = methods, columns = datasets + avg."""
    all_rows = sorted({row for sb, _, _, _, _, rk, _ in results_list for row in rk}, key=_row_sort_key)
    row_labels = [_row_label(*row) for row in all_rows]

    per_dataset = [
        best_per_row(sb, rk, ca, alpha_filter)
        for sb, _, _, ca, _, rk, _ in results_list
    ]

    def row_avg(row):
        vals = [bpr[row][0] for bpr in per_dataset if row in bpr and bpr[row][0] is not None]
        return sum(vals) / len(vals) if vals else None

    def row_all_fin(row):
        return all(bpr[row][1] for bpr in per_dataset if row in bpr and bpr[row][0] is not None)

    def row_avg_rt(row):
        rts = [bpr[row][2] for bpr in per_dataset if row in bpr and bpr[row][2] is not None]
        return sum(rts) / len(rts) if rts else None

    display_labels = [_DATASET_DISPLAY.get(os.path.basename(d), os.path.basename(d)) for d in dataset_labels]

    if latex:
        print_combined_table_latex(display_labels, all_rows, per_dataset, row_avg, row_avg_rt, row_all_fin, textbf=textbf, alpha=alpha_filter, n=n)
        return

    if markdown:
        print_combined_table_markdown(display_labels, all_rows, row_labels, per_dataset,
                                      row_avg, row_avg_rt, row_all_fin, textbf=textbf)
        return

    if tsv:
        print("\t".join(["Method", "GPU hours"] + display_labels + ["Avg"]))
        for row, label in zip(all_rows, row_labels):
            avg_rt = row_avg_rt(row)
            cells = [label, f"{avg_rt:.2f}" if avg_rt is not None else ""]
            for bpr in per_dataset:
                entry = bpr.get(row)
                if entry is None or entry[0] is None:
                    cells.append("")
                else:
                    val, all_fin, _, _, _ = entry
                    suffix = "" if all_fin else "*"
                    cells.append(f"{val:.2f}{suffix}")
            avg = row_avg(row)
            cells.append(f"{avg:.2f}" if avg is not None else "")
            print("\t".join(cells))
        return

    row_w = max((len(l) for l in row_labels), default=6)
    row_w = max(row_w, len("Method"))
    col_w = max((len(d) for d in display_labels), default=8)
    col_w = max(col_w, len("00.00±0.00*"))
    avg_w = max(col_w, len("Avg"))
    rt_w = max(len("GPU hours"), len("00.00h"))

    header_parts = [f"{'Method':<{row_w}}", f"{'GPU hours':>{rt_w}}"]
    for ds in display_labels:
        header_parts.append(f"{ds:>{col_w}}")
    header_parts.append(f"{'Avg':>{avg_w}}")
    header = "  ".join(header_parts)
    sep = "-" * len(header)

    print()
    print("=== Combined ===")
    print(sep)
    print(header)
    print(sep)

    for row, label in zip(all_rows, row_labels):
        avg_rt = row_avg_rt(row)
        rt_cell = f"{avg_rt:.2f}h" if avg_rt is not None else "—"
        cells = [f"{label:<{row_w}}", rt_cell.rjust(rt_w)]
        avg_terms = []
        for bpr in per_dataset:
            entry = bpr.get(row)
            if entry is None or entry[0] is None:
                cells.append(f"{'—':>{col_w}}")
            else:
                val, all_fin, _, sem, n_seeds = entry
                suffix = "" if all_fin else "*"
                sem_str = f"±{sem:.2f}" if sem is not None else ""
                cells.append(f"{val:.2f}{sem_str}{suffix}".rjust(col_w))
                avg_terms.append((sem, n_seeds))
        avg = row_avg(row)
        suffix = "" if row_all_fin(row) else "*"
        avg_sem = avg_stderr(avg_terms)
        avg_sem_str = f"±{avg_sem:.2f}" if avg_sem is not None else ""
        cells.append((f"{avg:.2f}{avg_sem_str}{suffix}" if avg is not None else "—").rjust(avg_w))
        print("  ".join(cells))

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Summarize LA-LORA FL results.")
    parser.add_argument("--dir", required=True,
                        help="Target directory (single dataset or parent of multiple datasets)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Only consider rounds up to this value (inclusive)")
    parser.add_argument("--tsv", action="store_true",
                        help="Output tables with tab separators (for copy-paste into spreadsheets)")
    parser.add_argument("--latex", action="store_true",
                        help="Print combined tables as LaTeX (tabular)")
    parser.add_argument("--markdown", "--md", action="store_true", dest="markdown",
                        help="Print combined tables as markdown pipe tables")
    parser.add_argument("--textbf", action="store_true",
                        help="With --latex or --markdown, bold the highlighted row's numbers")
    parser.add_argument("--last", action="store_true",
                        help="Report last-round accuracy instead of best-round accuracy")
    parser.add_argument("--finished-only", action="store_true",
                        help="Leave out runs without 'Done.' in their .out")
    parser.add_argument("--n", nargs="+", default=None,
                        help="Only print tables for these client counts (e.g. --n 40 60 80)")
    parser.add_argument("--alpha", nargs="+", default=None,
                        help="Only print tables for these Dirichlet alphas (e.g. --alpha 0.8)")
    parser.add_argument("--r2-dir", default=None,
                        help="Extra root of doubled-rank Frank-LoRA runs; its results are matched "
                             "to the main tables by dataset name (hence n) and alpha, and added "
                             "as separate 'doubled ranks' rows")
    args = parser.parse_args()

    chosen = [f for f, on in (("--tsv", args.tsv), ("--latex", args.latex),
                              ("--markdown", args.markdown)) if on]
    if len(chosen) > 1:
        print(f"Pick one output format, got: {', '.join(chosen)}")
        raise SystemExit(1)

    root = args.dir
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        raise SystemExit(1)
    if args.r2_dir is not None and not os.path.isdir(args.r2_dir):
        print(f"Not a directory: {args.r2_dir}")
        raise SystemExit(1)

    dataset_dirs = find_dataset_dirs(root)
    if args.n is not None:
        # Drop other client counts before reading any logs (reading is the slow part).
        dataset_dirs = [(l, p) for l, p in dataset_dirs
                        if (m := re.search(r"_n(\d+)$", os.path.basename(p))) and m.group(1) in args.n]
    if not dataset_dirs:
        print("No results found.")
        return

    # A single leaf dataset dir reports itself as "."; name it after the directory
    # so it flows through the same combined-table path as the multi-dataset case.
    if len(dataset_dirs) == 1 and dataset_dirs[0][0] == ".":
        dataset_dirs = [(os.path.basename(os.path.abspath(root)), dataset_dirs[0][1])]
    # --latex / --markdown output is meant to be pasted somewhere, so keep it to
    # the tables alone.
    quiet = args.latex or args.markdown

    if not quiet:
        print(f"dir: {os.path.abspath(root)}")
        if args.r2_dir:
            print(f"r2 dir: {os.path.abspath(args.r2_dir)}")
    results_list = []
    dataset_labels = []
    all_incomplete = []
    for label, path in dataset_dirs:
        result = process_dataset_dir(path, args.max_rounds, use_last=args.last,
                                     finished_only=args.finished_only)
        results_list.append(result)
        dataset_labels.append(label)

    # Doubled-rank runs: match by dataset label, which carries both the dataset
    # name and the _nX suffix, then merge in as separate rows.
    if args.r2_dir:
        r2_by_label = {
            label: process_dataset_dir(path, args.max_rounds, use_last=args.last,
                                       finished_only=args.finished_only)
            for label, path in find_dataset_dirs(args.r2_dir)
        }
        matched = set()
        for i, label in enumerate(dataset_labels):
            r2 = r2_by_label.get(label)
            if r2 is None or not r2[0]:
                continue
            results_list[i] = merge_r2_result(results_list[i], r2)
            matched.add(label)
        unmatched = sorted(set(r2_by_label) - matched)
        if unmatched and not quiet and not args.tsv:
            print(f"r2 datasets with no match in --dir ({len(unmatched)}): {', '.join(unmatched)}")

    for result in results_list:
        all_incomplete.extend(result[6])

    if all_incomplete and not quiet:
        print()
        print(f"Incomplete runs ({len(all_incomplete)}):")
        for run_dir in sorted(all_incomplete):
            print(f"  {run_dir}")

    non_empty = [(l, r) for l, r in zip(dataset_labels, results_list) if r[0]]  # r[0] = seed_best
    if non_empty:
        # Group datasets by their _nX suffix (num_clients); each group prints its own combined table.
        def _extract_n(label):
            basename = os.path.basename(label) if label else label
            m = re.search(r"_n(\d+)$", basename) if basename else None
            return m.group(1) if m else None

        def _strip_n(label):
            if not label:
                return label
            head, tail = os.path.split(label)
            new_tail = re.sub(r"_n\d+$", "", tail)
            return os.path.join(head, new_tail) if head else new_tail

        n_groups = defaultdict(list)
        for label, result in non_empty:
            n_groups[_extract_n(label)].append((label, result))

        group_keys = sorted(n_groups.keys(), key=lambda x: (x is None, int(x) if x is not None else 0))

        if args.n is not None:
            group_keys = [n for n in group_keys if n in args.n]
        alpha_keep = None if args.alpha is None else {_norm_alpha(a) for a in args.alpha}

        for n in group_keys:
            items = n_groups[n]
            if n is not None:
                combined_labels = [_strip_n(l) for l, _ in items]
            else:
                combined_labels = [l for l, _ in items]
            combined_results = [r for _, r in items]

            # Collect all alpha values seen in this group.
            all_alphas = sorted({
                alpha
                for _, _, _, ca, _, _, _ in combined_results
                for alpha in ca.values()
            })

            combined_legend = ("(*) = in progress  (mean over seeds, best over configs and lr)"
                               "  (± = standard error over seeds)")

            def _title(alpha):
                tags = []
                if n is not None:
                    tags.append(f"n={n}")
                if alpha is not None:
                    tags.append(f"alpha={alpha}")
                tag_str = f" ({', '.join(tags)})" if tags else ""
                # Markdown wants a plain ATX heading, not the ### ... ### banner.
                if args.markdown:
                    return f"### Combined{tag_str}"
                return f"### Combined{tag_str} ###"

            def _emit(alpha):
                print()
                if not args.latex:
                    print(_title(alpha))
                if args.markdown:
                    print()
                print_combined_table(combined_labels, combined_results, tsv=args.tsv,
                                     alpha_filter=alpha, latex=args.latex,
                                     markdown=args.markdown, textbf=args.textbf, n=n)
                if not args.tsv and not args.latex and not args.markdown:
                    print(combined_legend)

            if alpha_keep is not None:
                all_alphas = [a for a in all_alphas if a in alpha_keep]
                if not all_alphas:
                    continue
            if len(all_alphas) <= 1:
                _emit(all_alphas[0] if all_alphas else None)
            else:
                for alpha in all_alphas:
                    _emit(alpha)


if __name__ == "__main__":
    main()
