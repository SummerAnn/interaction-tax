#!/usr/bin/env python3
"""
Multi-synthesizer diversity analysis for the MoA synthesis row.

Protocols in results/2x2-multisynth/:
  moa-synth-gpt4o      (diversity=1, GPT-4o synthesizer)
  moa-same-synth-gpt4o (diversity=0, GPT-4o synthesizer)
  moa-synth-gemini     (diversity=1, Gemini synthesizer)
  moa-same-synth-gemini(diversity=0, Gemini synthesizer)

Plus original Claude synthesizer from results/full-v2/:
  moa            (diversity=1, Claude synthesizer)
  moa-same-model (diversity=0, Claude synthesizer)

For each synthesizer, estimate:
  Q ~ diversity + task_dummies

Primary test: diversity coefficient positive across all 3 synthesizers.
"""

import json
import os
import numpy as np

DATA_DIR_ORIG = "/Users/summerann/Desktop/neurips-experiments/results/full-v2"
DATA_DIR_NEW  = "/Users/summerann/Desktop/neurips-experiments/results/2x2-multisynth"

TASKS = {
    "bench-difference-bases": ("minimize", 14.0, 1.0),
    "bench-erdos-overlap":    ("minimize", 0.2939, 0.2321),
    "bench-molecule-qed":     ("maximize", 0.60, 1.0),
}

# New synthesizer protocols
NEW_PROTOCOLS = {
    # proto_id: (diversity, synthesizer)
    "moa-synth-gpt4o":       (1, "gpt4o"),
    "moa-same-synth-gpt4o":  (0, "gpt4o"),
    "moa-synth-gemini":      (1, "gemini"),
    "moa-same-synth-gemini": (0, "gemini"),
}

# Original Claude synthesizer protocols
CLAUDE_PROTOCOLS = {
    "moa":            (1, "claude"),
    "moa-same-model": (0, "claude"),
}

N_BOOTSTRAP = 10_000
RNG_SEED = 99


def compute_q(score, direction, s_bad, s_ref):
    if direction == "minimize":
        q = (s_bad - score) / (s_bad - s_ref)
    else:
        q = (score - s_bad) / (s_ref - s_bad)
    return max(0.0, min(1.0, q))


def ols_fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def bootstrap_div_coeff(X, Q, rng, n_boot=N_BOOTSTRAP):
    n = len(Q)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        b = ols_fit(X[idx], Q[idx])
        boots.append(b[1])
    boots = np.array(boots)
    ci_lo = np.percentile(boots, 2.5)
    ci_hi = np.percentile(boots, 97.5)
    p_pos = np.mean(boots > 0)
    p_val = 2 * min(p_pos, 1 - p_pos)
    return boots.mean(), ci_lo, ci_hi, p_val


def load_rows(protocols, data_dir):
    """Load (Q, diversity, synthesizer, task_idx) rows."""
    rows = []
    task_names = list(TASKS.keys())
    missing = []
    for task_idx, (task_id, (direction, s_bad, s_ref)) in enumerate(TASKS.items()):
        for proto_id, (div_flag, synth_label) in protocols.items():
            for seed in range(1, 11):
                fname = f"{task_id}_{proto_id}_s{seed}.json"
                fpath = os.path.join(data_dir, fname)
                if not os.path.exists(fpath):
                    missing.append(fname)
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                hs = data.get("hiddenScore")
                if hs is None:
                    missing.append(f"{fname} (null hiddenScore)")
                    continue
                q = compute_q(hs, direction, s_bad, s_ref)
                rows.append((q, div_flag, synth_label, task_idx))
    return rows, missing


def analyze_synthesizer(rows, synth_label, rng):
    """Run 2x2 diversity analysis for a single synthesizer."""
    arr = np.array([(r[0], r[1], r[3]) for r in rows if r[2] == synth_label], dtype=float)
    if len(arr) == 0:
        return None
    Q = arr[:, 0]
    diversity = arr[:, 1]
    task_idx = arr[:, 2].astype(int)

    present = np.unique(task_idx)
    task_dummies = np.zeros((len(Q), len(present) - 1))
    for i, t in enumerate(present[1:]):
        task_dummies[:, i] = (task_idx == t).astype(float)

    X = np.column_stack([np.ones(len(Q)), diversity, task_dummies])
    beta = ols_fit(X, Q)
    _, ci_lo, ci_hi, p_val = bootstrap_div_coeff(X, Q, rng)

    n_div = int(diversity.sum())
    n_same = int((1 - diversity).sum())
    mean_div  = float(Q[diversity == 1].mean())
    mean_same = float(Q[diversity == 0].mean())

    return {
        "synth": synth_label,
        "n": len(Q),
        "n_div": n_div,
        "n_same": n_same,
        "beta_div": beta[1],
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_val": p_val,
        "mean_div": mean_div,
        "mean_same": mean_same,
    }


def per_task_means(rows, synth_label):
    """Compute per-task cell means for a synthesizer."""
    task_names = list(TASKS.keys())
    task_labels = {
        "bench-difference-bases": "DiffBases",
        "bench-erdos-overlap":    "Erdős",
        "bench-molecule-qed":     "MolQED",
    }
    print(f"\n  Synthesizer: {synth_label}")
    print(f"  {'Task':<12}  {'Diverse':>8}  {'Same':>8}  {'Δ':>8}")
    for ti, task_id in enumerate(task_names):
        sub = [(r[0], r[1]) for r in rows if r[2] == synth_label and r[3] == ti]
        if not sub:
            continue
        qs = np.array([r[0] for r in sub])
        ds = np.array([r[1] for r in sub])
        mean_d = qs[ds == 1].mean() if any(ds == 1) else float("nan")
        mean_s = qs[ds == 0].mean() if any(ds == 0) else float("nan")
        print(f"  {task_labels[task_id]:<12}  {mean_d:8.3f}  {mean_s:8.3f}  {mean_d - mean_s:+8.3f}")


def main():
    rng = np.random.RandomState(RNG_SEED)

    # Load new synthesizer data
    new_rows, new_missing = load_rows(NEW_PROTOCOLS, DATA_DIR_NEW)
    # Load original Claude data
    claude_rows, claude_missing = load_rows(CLAUDE_PROTOCOLS, DATA_DIR_ORIG)

    all_rows = new_rows + claude_rows
    all_missing = new_missing + claude_missing

    if all_missing:
        print(f"WARNING: {len(all_missing)} missing files")
        for m in all_missing[:10]:
            print(f"  {m}")

    synthesizers = ["claude", "gpt4o", "gemini"]

    print("=" * 70)
    print("MULTI-SYNTHESIZER 2x2 DIVERSITY ANALYSIS")
    print("(diversity main effect, 3 tasks DiffBases+Erdős+MolQED; N=60 each synth)")
    print("=" * 70)
    print(f"\n{'Synthesizer':<12}  {'N':>4}  {'beta_div':>9}  {'95% CI':>22}  {'p':>8}  {'sig':>4}")
    print("-" * 65)

    results = {}
    for synth in synthesizers:
        r = analyze_synthesizer(all_rows, synth, rng)
        if r is None:
            print(f"  {synth:<12}  NO DATA")
            continue
        results[synth] = r
        sig = "*" if r["ci_lo"] > 0 else ""
        print(f"  {r['synth']:<12}  {r['n']:>4}  {r['beta_div']:+9.3f}  "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  {r['p_val']:>8.3f}  {sig:>4}")

    print()
    print("=" * 70)
    print("CELL MEANS BY SYNTHESIZER AND TASK")
    print("=" * 70)
    for synth in synthesizers:
        per_task_means(all_rows, synth)

    # MolQED-only check
    print()
    print("=" * 70)
    print("MolQED ONLY: diversity effect per synthesizer")
    print("=" * 70)
    for synth in synthesizers:
        sub = [(r[0], r[1]) for r in all_rows if r[2] == synth and r[3] == 2]  # task_idx=2 is MolQED
        if not sub:
            continue
        qs = np.array([r[0] for r in sub])
        ds = np.array([r[1] for r in sub])
        md = qs[ds == 1].mean() if any(ds == 1) else float("nan")
        ms = qs[ds == 0].mean() if any(ds == 0) else float("nan")
        print(f"  {synth:<12}  diverse={md:.3f}  same={ms:.3f}  Δ={md-ms:+.3f}")

    # Summary for paper
    print()
    print("=" * 70)
    print("PAPER SUMMARY")
    print("=" * 70)
    all_positive = all(r["beta_div"] > 0 for r in results.values())
    all_sig = all(r["ci_lo"] > 0 for r in results.values())
    print(f"  All 3 synthesizers: diversity coeff positive? {all_positive}")
    print(f"  All 3 synthesizers: CI excludes zero?         {all_sig}")
    for synth, r in results.items():
        print(f"  {synth}: β={r['beta_div']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] p={r['p_val']:.3f}")

    # LaTeX table row
    print()
    print("LaTeX rows (for paper table):")
    for synth, r in results.items():
        sig = "*" if r["ci_lo"] > 0 else ""
        synth_label = {"claude": "Claude Sonnet 4", "gpt4o": "GPT-4o", "gemini": "Gemini 2.5 Flash"}[synth]
        print(f"  {synth_label} & ${r['beta_div']:+.3f}$ & $[{r['ci_lo']:+.3f},\\,{r['ci_hi']:+.3f}]$ & ${r['p_val']:.3f}$ \\\\")


if __name__ == "__main__":
    main()
