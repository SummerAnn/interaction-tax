#!/usr/bin/env python3
"""
Convergence diagnostics: Pairwise proposal distances by condition and task.

Two key comparisons:
  1. MoA-diverse proposers vs MoA-same-model proposers
     — Does model-family diversity produce more dissimilar intermediate proposals?
  2. MoA-nosynth (independent, diverse) vs Debate-mixed (post-interaction, diverse)
     — Does interaction reduce proposal diversity even for mixed-model teams?

Distance metrics:
  DiffBases  : Jaccard distance on integer sets  (1 - |A∩B|/|A∪B|)
  Erdős      : Cosine distance on value arrays   (after interpolation to len=200)
  MolQED     : Normalized edit distance on SMILES strings

Protocol → artifact mapping:
  moa              : allArtifacts[:-1] = proposers,  [-1] = synthesizer (skip)
  moa-same-model   : allArtifacts[:-1] = proposers,  [-1] = synthesizer (skip)
  moa-nosynth      : allArtifacts[:]   = proposers,  no synthesizer
  best-of-3        : allArtifacts[:]   = proposers,  no synthesizer
  debate-mixed     : allArtifacts[:2]  = debaters,   [2] = synthesizer (skip)
  debate           : allArtifacts[:2]  = debaters,   [2] = synthesizer (skip)

Output:
  — Table of mean pairwise distance ± 95% CI by (task, condition)
  — Bootstrap significance for key pairwise comparisons
  — LaTeX table rows for paper
"""

import json
import os
import numpy as np
from itertools import combinations

DATA_DIR = "/Users/summerann/Desktop/neurips-experiments/results/full-v2"
N_BOOTSTRAP = 10_000
RNG_SEED = 42

# ─── Task configuration ───────────────────────────────────────────────────────

TASKS = {
    "bench-difference-bases": ("DiffBases", "set",    "jaccard"),
    "bench-erdos-overlap":    ("Erdős",     "values", "cosine"),
    "bench-molecule-qed":     ("MolQED",    "smiles", "levenshtein"),
}

# Protocols: (label, artifact_slice, has_synthesis, seeds)
PROTOCOLS = {
    "moa":           ("MoA-diverse",          "drop_last", True,  range(1, 11)),
    "moa-same-model":("MoA-same-model",        "drop_last", True,  range(1, 11)),
    "moa-nosynth":   ("MoA-nosynth (diverse)", "all",       False, range(1, 11)),
    "best-of-3":     ("Best-of-3 (same-model)","all",       False, range(1, 11)),
    "debate-mixed":  ("Debate-mixed",          "first_two", True,  range(1, 6)),
    "debate":        ("Debate (same-model)",   "first_two", True,  range(1, 6)),
}

# ─── Distance functions ───────────────────────────────────────────────────────

def jaccard_distance(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / len(sa | sb)


def cosine_distance(a, b, target_len=200):
    """Interpolate both arrays to target_len, then compute 1 - cosine_sim."""
    def interpolate(arr, n):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == n:
            return arr
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, n)
        return np.interp(x_new, x_old, arr)

    va = interpolate(a, target_len)
    vb = interpolate(b, target_len)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    sim = np.dot(va, vb) / (norm_a * norm_b)
    return 1.0 - float(np.clip(sim, -1, 1))


def edit_distance(s1, s2):
    """Simple O(mn) Levenshtein distance."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def normalized_edit_distance(s1, s2):
    if not s1 and not s2:
        return 0.0
    return edit_distance(s1, s2) / max(len(s1), len(s2), 1)


def pairwise_distance(artifacts, dist_field, dist_fn):
    """Compute all pairwise distances among a list of artifacts."""
    vals = []
    for a in artifacts:
        sd = a.get("solutionData", {})
        v = sd.get(dist_field)
        if v is None:
            return []
        vals.append(v)

    if len(vals) < 2:
        return []

    dists = []
    for i, j in combinations(range(len(vals)), 2):
        try:
            d = dist_fn(vals[i], vals[j])
            if np.isfinite(d):
                dists.append(d)
        except Exception:
            pass
    return dists


# ─── Data loading ─────────────────────────────────────────────────────────────

def get_proposers(all_artifacts, slice_mode):
    """Extract proposer artifacts based on slice mode."""
    if slice_mode == "drop_last":
        return all_artifacts[:-1] if len(all_artifacts) > 1 else all_artifacts
    elif slice_mode == "first_two":
        return all_artifacts[:2]
    else:  # "all"
        return all_artifacts


def load_distances(task_id, dist_field, dist_fn):
    """Load pairwise distances for all protocols × seeds for one task."""
    results = {}  # proto_id → list of (seed, [distances])
    for proto_id, (label, slice_mode, _, seeds) in PROTOCOLS.items():
        per_seed = []
        for seed in seeds:
            fname = f"{task_id}_{proto_id}_s{seed}.json"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except Exception:
                continue
            all_arts = data.get("allArtifacts", [])
            if not all_arts:
                continue
            proposers = get_proposers(all_arts, slice_mode)
            dists = pairwise_distance(proposers, dist_field, dist_fn)
            if dists:
                per_seed.append((seed, dists))
        results[proto_id] = per_seed
    return results


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_mean(values, rng, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for the mean of a flat list of values."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values)
    boots = np.array([
        np.random.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    return arr.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def bootstrap_delta(vals_a, vals_b, rng, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for mean(a) - mean(b); returns (delta, ci_lo, ci_hi, p_val)."""
    if not vals_a or not vals_b:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a = np.array(vals_a)
    b = np.array(vals_b)
    obs = a.mean() - b.mean()
    boots = []
    for _ in range(n_boot):
        ba = np.random.choice(a, size=len(a), replace=True).mean()
        bb = np.random.choice(b, size=len(b), replace=True).mean()
        boots.append(ba - bb)
    boots = np.array(boots)
    ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    p_pos = np.mean(boots > 0)
    p_val = 2 * min(p_pos, 1 - p_pos)
    return obs, ci_lo, ci_hi, p_val


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.RandomState(RNG_SEED)

    # Collect all pairwise distances
    # Structure: task_id → proto_id → flat list of distances (pooled across seeds)
    all_dists = {}
    for task_id, (task_label, dist_field, dist_type) in TASKS.items():
        if dist_type == "jaccard":
            dist_fn = jaccard_distance
        elif dist_type == "cosine":
            dist_fn = cosine_distance
        else:
            dist_fn = normalized_edit_distance

        raw = load_distances(task_id, dist_field, dist_fn)
        all_dists[task_id] = {}
        for proto_id, per_seed in raw.items():
            flat = [d for _, dists in per_seed for d in dists]
            all_dists[task_id][proto_id] = flat

    # ── Table 1: Mean pairwise distance by condition and task ─────────────────

    print("=" * 80)
    print("PAIRWISE PROPOSAL DISTANCE BY CONDITION AND TASK")
    print("(higher = more dissimilar proposals)")
    print("=" * 80)

    proto_order = ["moa", "moa-same-model", "moa-nosynth", "best-of-3",
                   "debate-mixed", "debate"]

    header = f"{'Protocol':<30}"
    for _, (task_label, _, _) in TASKS.items():
        header += f"  {task_label:>12}"
    print(header)
    print("-" * 70)

    for proto_id in proto_order:
        label = PROTOCOLS[proto_id][0]
        row = f"{label:<30}"
        for task_id in TASKS:
            vals = all_dists[task_id].get(proto_id, [])
            if vals:
                m, lo, hi = bootstrap_mean(vals, rng)
                row += f"  {m:.3f}±{(hi-lo)/2:.3f}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    # ── Table 2: Key comparisons ──────────────────────────────────────────────

    print()
    print("=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)

    comparisons = [
        ("C1: Diverse vs same-model (proposers, no synthesis)",
         "moa", "moa-same-model",
         "Diverse models produce more dissimilar proposals"),
        ("C2: Diverse vs same-model (selection, no synthesis)",
         "moa-nosynth", "best-of-3",
         "Same pattern holds without synthesis step"),
        ("C3: Interaction reduces diversity (mixed-model teams)",
         "moa-nosynth", "debate-mixed",
         "Interaction (debate) reduces proposal diversity vs. independent generation"),
        ("C4: Interaction reduces diversity (same-model teams)",
         "best-of-3", "debate",
         "Same-model interaction also reduces proposal diversity"),
    ]

    for desc, proto_a, proto_b, interpretation in comparisons:
        print(f"\n{desc}")
        print(f"  (A={PROTOCOLS[proto_a][0]}, B={PROTOCOLS[proto_b][0]}, delta=A-B)")
        print(f"  {'Task':<12}  {'Delta':>8}  {'95% CI':>20}  {'p':>8}  {'sig':>4}")
        print(f"  {'-'*60}")
        deltas = []
        for task_id, (task_label, _, _) in TASKS.items():
            vals_a = all_dists[task_id].get(proto_a, [])
            vals_b = all_dists[task_id].get(proto_b, [])
            d, lo, hi, p = bootstrap_delta(vals_a, vals_b, rng)
            sig = "*" if lo > 0 else ("†" if p < 0.10 else "")
            if np.isfinite(d):
                print(f"  {task_label:<12}  {d:+8.3f}  [{lo:+.3f}, {hi:+.3f}]  {p:>8.3f}  {sig:>4}")
                deltas.append(d)
            else:
                print(f"  {task_label:<12}  {'N/A':>8}")
        if deltas:
            print(f"  {'Mean Δ':<12}  {np.mean(deltas):+8.3f}")

    # ── Table 3: Per-seed distances (for verification) ────────────────────────

    print()
    print("=" * 80)
    print("PER-SEED DISTANCE SUMMARY (MoA-diverse vs MoA-same-model)")
    print("=" * 80)
    for task_id, (task_label, dist_field, dist_type) in TASKS.items():
        if dist_type == "jaccard":
            dist_fn = jaccard_distance
        elif dist_type == "cosine":
            dist_fn = cosine_distance
        else:
            dist_fn = normalized_edit_distance

        raw = load_distances(task_id, dist_field, dist_fn)
        print(f"\n  {task_label}")
        for proto_id in ["moa", "moa-same-model", "debate-mixed"]:
            per_seed = raw.get(proto_id, [])
            if not per_seed:
                continue
            label = PROTOCOLS[proto_id][0]
            means = [np.mean(dists) for _, dists in per_seed if dists]
            if means:
                print(f"    {label:<30}: n_seeds={len(means)}, "
                      f"mean_dist={np.mean(means):.3f}, "
                      f"sd={np.std(means):.3f}")

    # ── LaTeX snippet ─────────────────────────────────────────────────────────

    print()
    print("=" * 80)
    print("LATEX TABLE ROWS (for paper)")
    print("=" * 80)
    print(r"\begin{tabular}{l r r r}")
    print(r"\toprule")
    print(r"Condition & DiffBases & Erd\H{o}s & MolQED \\")
    print(r"\midrule")
    for proto_id in proto_order:
        label = PROTOCOLS[proto_id][0]
        cells = []
        for task_id in TASKS:
            vals = all_dists[task_id].get(proto_id, [])
            if vals:
                m, lo, hi = bootstrap_mean(vals, rng)
                cells.append(f"${m:.3f}$")
            else:
                cells.append("---")
        print(f"{label} & {' & '.join(cells)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ── Summary for paper text ────────────────────────────────────────────────

    print()
    print("=" * 80)
    print("SUMMARY FOR PAPER TEXT")
    print("=" * 80)
    # C1: diverse vs same-model on moa proposers
    for task_id, (task_label, _, _) in TASKS.items():
        vals_div  = all_dists[task_id].get("moa", [])
        vals_same = all_dists[task_id].get("moa-same-model", [])
        d, lo, hi, p = bootstrap_delta(vals_div, vals_same, rng)
        if np.isfinite(d):
            print(f"  C1 {task_label}: Δ_diversity={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
    print()
    # C3: no-interaction vs interaction (diverse teams)
    for task_id, (task_label, _, _) in TASKS.items():
        vals_nosynth = all_dists[task_id].get("moa-nosynth", [])
        vals_debate  = all_dists[task_id].get("debate-mixed", [])
        d, lo, hi, p = bootstrap_delta(vals_nosynth, vals_debate, rng)
        if np.isfinite(d):
            print(f"  C3 {task_label}: Δ_interaction_cost={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")


if __name__ == "__main__":
    main()
