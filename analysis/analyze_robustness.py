#!/usr/bin/env python3
"""
Robustness checks on aggregate MEG results.

Checks:
1. Leave-one-task-out (LOO-task): drop each task, recompute agg MEG over 7.
2. Median vs mean aggregation across tasks.
3. Leave-one-seed-out jackknife: SE for aggregate MEG.
4. Summary table.
"""

import json
import os
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = "/Users/summerann/Desktop/neurips-experiments/results/full-v2"

TASKS = {
    # task_id: (direction, s_bad, s_ref, sentinel)
    # sentinel: value that indicates an invalid/failed run (mapped to Q=0)
    "bench-maxcut-g200":        ("maximize", 3139.0,    3517.0,    None),
    "bench-circle-packing-n20": ("maximize", 0.5039,    2.0,       None),
    "bench-difference-bases":   ("minimize", 14.0,      1.0,       0),
    "bench-flat-poly-deg50":    ("minimize", 2.0489,    1.0,       None),
    "bench-tsp-100":            ("minimize", 50835.0,   7517.0,    0),
    "bench-lj-n41":             ("minimize", -136.0,    -198.061,  1e30),
    "bench-erdos-overlap":      ("minimize", 0.2939,    0.2321,    None),
    "bench-molecule-qed":       ("maximize", 0.60,      1.0,       None),
}

PROTOCOLS = [
    "single-shot", "self-refine", "best-of-n", "vgs",
    "homo-chain", "cross-chain", "magicore", "debate", "moa", "hpe",
]

CONTROLS = ["self-refine", "best-of-n", "vgs"]

# Seeds per protocol
def seeds_for(proto):
    if proto == "moa":
        return list(range(1, 11))
    return list(range(1, 6))


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_hidden_score(task_id, proto_id, seed):
    """Load hiddenScore from JSON file. Returns None if file is missing."""
    fname = f"{task_id}_{proto_id}_s{seed}.json"
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)["hiddenScore"]


def is_sentinel(score, sentinel):
    """Check if a score is a sentinel (invalid/failed run)."""
    if sentinel is None:
        return False
    if sentinel == 0:
        return score == 0
    return score >= sentinel * 0.5  # catches 1e30-scale values


def q_normalize(score, direction, s_bad, s_ref, sentinel=None):
    if is_sentinel(score, sentinel):
        return 0.0  # invalid run → Q=0 (penalised, matches analyze_bench.py)
    if direction == "minimize":
        q = (s_bad - score) / (s_bad - s_ref)
    else:
        q = (score - s_bad) / (s_ref - s_bad)
    return float(np.clip(q, 0.0, 1.0))


def load_all_scores():
    """Return dict: scores[task][proto] = list of (seed, Q) tuples for available seeds."""
    scores = {}
    missing = []
    for task_id, (direction, s_bad, s_ref, sentinel) in TASKS.items():
        scores[task_id] = {}
        for proto in PROTOCOLS:
            seed_qs = []
            for seed in seeds_for(proto):
                raw = load_hidden_score(task_id, proto, seed)
                if raw is None:
                    missing.append(f"{task_id}_{proto}_s{seed}")
                else:
                    seed_qs.append((seed, q_normalize(raw, direction, s_bad, s_ref, sentinel)))
            scores[task_id][proto] = seed_qs
    return scores, missing


def mean_q_from_pairs(seed_q_pairs, exclude_seed=None):
    """Compute mean Q from list of (seed, q) pairs, optionally excluding one seed."""
    vals = [q for s, q in seed_q_pairs if s != exclude_seed]
    if len(vals) == 0:
        return np.nan
    return np.mean(vals)


def compute_meg(scores, task_list=None, exclude_seed_map=None):
    """
    Compute per-task mean_Q for each protocol, then MEG, then aggregate.

    task_list: subset of tasks to use (default: all).
    exclude_seed_map: dict proto -> seed_number to exclude (actual seed value, not index).
    Returns (agg_meg dict, per_task_meg dict).
    """
    if task_list is None:
        task_list = list(TASKS.keys())

    # per-task mean Q
    mean_q = {}  # task -> proto -> float
    for task in task_list:
        mean_q[task] = {}
        for proto in PROTOCOLS:
            excl = None
            if exclude_seed_map is not None and proto in exclude_seed_map:
                excl = exclude_seed_map[proto]
            mean_q[task][proto] = mean_q_from_pairs(scores[task][proto], exclude_seed=excl)

    # per-task control baseline
    control_base = {}
    for task in task_list:
        ctrl_vals = [mean_q[task][c] for c in CONTROLS if not np.isnan(mean_q[task][c])]
        control_base[task] = max(ctrl_vals) if ctrl_vals else np.nan

    # per-task MEG
    per_task_meg = {}  # proto -> list of per-task MEGs
    for proto in PROTOCOLS:
        per_task_meg[proto] = []
        for task in task_list:
            val = mean_q[task][proto] - control_base[task]
            per_task_meg[proto].append(val)

    # aggregate MEG (unweighted mean across tasks)
    agg_meg = {}
    for proto in PROTOCOLS:
        vals = [v for v in per_task_meg[proto] if not np.isnan(v)]
        agg_meg[proto] = np.mean(vals) if vals else np.nan

    return agg_meg, per_task_meg


def compute_meg_median(scores, task_list=None):
    """Same as compute_meg but use median across tasks instead of mean."""
    if task_list is None:
        task_list = list(TASKS.keys())

    mean_q = {}
    for task in task_list:
        mean_q[task] = {}
        for proto in PROTOCOLS:
            mean_q[task][proto] = mean_q_from_pairs(scores[task][proto])

    control_base = {}
    for task in task_list:
        ctrl_vals = [mean_q[task][c] for c in CONTROLS if not np.isnan(mean_q[task][c])]
        control_base[task] = max(ctrl_vals) if ctrl_vals else np.nan

    per_task_meg = {}
    for proto in PROTOCOLS:
        per_task_meg[proto] = []
        for task in task_list:
            per_task_meg[proto].append(mean_q[task][proto] - control_base[task])

    agg_meg_median = {}
    for proto in PROTOCOLS:
        vals = [v for v in per_task_meg[proto] if not np.isnan(v)]
        agg_meg_median[proto] = float(np.median(vals)) if vals else np.nan

    return agg_meg_median


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    scores, missing = load_all_scores()
    task_list = list(TASKS.keys())

    if missing:
        print(f"\n  WARNING: {len(missing)} missing file(s):")
        for m in missing:
            print(f"    {m}")

    # Count available seeds per (task, proto)
    print("\n  Available seeds per (task, protocol):")
    header = f"  {'Task':30s}"
    for p in PROTOCOLS:
        header += f"  {p[:7]:>7s}"
    print(header)
    for task in task_list:
        row = f"  {task:30s}"
        for p in PROTOCOLS:
            row += f"  {len(scores[task][p]):>7d}"
        print(row)

    # ── Baseline aggregate MEG ──────────────────────────────────────────────
    baseline_meg, baseline_per_task = compute_meg(scores)

    print("\n" + "="*80)
    print("BASELINE AGGREGATE MEG (mean across 8 tasks)")
    print("="*80)
    rank = sorted(PROTOCOLS, key=lambda p: baseline_meg[p], reverse=True)
    for i, p in enumerate(rank, 1):
        print(f"  {i:2d}. {p:15s}  AggMEG = {baseline_meg[p]:+.4f}")

    # ── 1. Leave-one-task-out ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("1. LEAVE-ONE-TASK-OUT")
    print("="*80)

    loo_results = {}  # proto -> list of agg MEGs (one per dropped task)
    loo_by_task = {}  # drop_task -> agg_meg dict
    for drop_task in task_list:
        remaining = [t for t in task_list if t != drop_task]
        meg_loo, _ = compute_meg(scores, task_list=remaining)
        loo_by_task[drop_task] = meg_loo
        for proto in PROTOCOLS:
            loo_results.setdefault(proto, []).append(meg_loo[proto])

    print(f"\n  {'Protocol':15s}  {'Min AggMEG':>10s}  {'Max AggMEG':>10s}  {'Range':>10s}  {'Baseline':>10s}")
    print("  " + "-"*62)
    for p in rank:
        vals = loo_results[p]
        lo, hi = min(vals), max(vals)
        print(f"  {p:15s}  {lo:+10.4f}  {hi:+10.4f}  {hi-lo:10.4f}  {baseline_meg[p]:+10.4f}")

    # Check if ranking changes
    print("\n  Ranking stability (LOO-task):")
    baseline_rank = {p: i for i, p in enumerate(rank)}
    any_change = False
    for drop_task in task_list:
        meg_loo = loo_by_task[drop_task]
        loo_rank = sorted(PROTOCOLS, key=lambda p: meg_loo[p], reverse=True)
        loo_rank_map = {p: i for i, p in enumerate(loo_rank)}
        changes = [(p, baseline_rank[p]+1, loo_rank_map[p]+1)
                    for p in PROTOCOLS if baseline_rank[p] != loo_rank_map[p]]
        if changes:
            any_change = True
            print(f"    Drop {drop_task}:")
            for p, old, new in sorted(changes, key=lambda x: x[1]):
                print(f"      {p:15s}  {old} -> {new}")

    if not any_change:
        print("    No ranking changes under any single-task drop.")

    # ── 2. Median vs Mean ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("2. MEDIAN vs MEAN AGGREGATION")
    print("="*80)
    median_meg = compute_meg_median(scores)

    median_rank = sorted(PROTOCOLS, key=lambda p: median_meg[p], reverse=True)
    print(f"\n  {'Protocol':15s}  {'Mean AggMEG':>12s}  {'Median AggMEG':>14s}  {'Rank(mean)':>11s}  {'Rank(med)':>10s}")
    print("  " + "-"*68)
    for p in rank:
        mr = median_rank.index(p) + 1
        br = rank.index(p) + 1
        print(f"  {p:15s}  {baseline_meg[p]:+12.4f}  {median_meg[p]:+14.4f}  {br:>11d}  {mr:>10d}")

    # ── 3. Leave-one-seed-out jackknife ────────────────────────────────────
    print("\n" + "="*80)
    print("3. LEAVE-ONE-SEED-OUT JACKKNIFE")
    print("="*80)

    jackknife_se = {}
    for proto in PROTOCOLS:
        # Get all seeds that actually exist for this protocol across tasks
        available_seeds = seeds_for(proto)
        n_seeds = len(available_seeds)
        theta_i = []
        for seed_to_drop in available_seeds:
            sd = {proto: seed_to_drop}
            meg_jack, _ = compute_meg(scores, exclude_seed_map=sd)
            theta_i.append(meg_jack[proto])

        theta_i = np.array(theta_i)
        n = len(theta_i)
        theta_bar = np.mean(theta_i)
        jack_var = ((n - 1) / n) * np.sum((theta_i - theta_bar)**2)
        jack_se = np.sqrt(jack_var)
        jackknife_se[proto] = jack_se

    print(f"\n  {'Protocol':15s}  {'AggMEG':>10s}  {'Jack SE':>10s}  {'Seeds':>6s}  {'AggMEG +/- 2SE':>20s}")
    print("  " + "-"*65)
    for p in rank:
        n = len(seeds_for(p))
        lo = baseline_meg[p] - 2*jackknife_se[p]
        hi = baseline_meg[p] + 2*jackknife_se[p]
        print(f"  {p:15s}  {baseline_meg[p]:+10.4f}  {jackknife_se[p]:10.4f}  {n:6d}  [{lo:+.4f}, {hi:+.4f}]")

    # ── 4. Summary table ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("4. SUMMARY TABLE")
    print("="*80)

    print(f"\n  {'Protocol':15s} | {'Mean AggMEG':>12s} | {'Median AggMEG':>14s} | {'LOO-task range':>22s} | {'Jackknife SE':>13s}")
    print("  " + "-"*87)
    for p in rank:
        vals = loo_results[p]
        lo, hi = min(vals), max(vals)
        loo_str = f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"  {p:15s} | {baseline_meg[p]:+12.4f} | {median_meg[p]:+14.4f} | {loo_str:>22s} | {jackknife_se[p]:13.4f}")

    # ── Per-task MEG detail for reference ─────────────────────────────────
    print("\n" + "="*80)
    print("PER-TASK MEG DETAIL (for reference)")
    print("="*80)
    header = f"  {'Task':30s}"
    for p in rank:
        header += f"  {p:>12s}"
    print(header)
    print("  " + "-"*(30 + 14*len(PROTOCOLS)))
    for ti, task in enumerate(task_list):
        row = f"  {task:30s}"
        for p in rank:
            row += f"  {baseline_per_task[p][ti]:+12.4f}"
        print(row)

    # ── Per-task mean_Q for reference ─────────────────────────────────────
    print("\n" + "="*80)
    print("PER-TASK MEAN Q (for reference)")
    print("="*80)
    header = f"  {'Task':30s}"
    for p in rank:
        header += f"  {p:>12s}"
    print(header)
    print("  " + "-"*(30 + 14*len(PROTOCOLS)))
    for task in task_list:
        row = f"  {task:30s}"
        for p in rank:
            mq = mean_q_from_pairs(scores[task][p])
            row += f"  {mq:12.4f}"
        print(row)

    print()


if __name__ == "__main__":
    main()
