#!/usr/bin/env python3
"""
analyze_mismatch_sweep.py — Proxy-sensitivity analysis

Tests whether feedback-intensive protocols degrade fastest on tasks where
the dev/hidden evaluator gap is largest.

Methodology:
  1. For each task, measure dev/hidden mismatch as |mean_dev_rank - mean_hidden_rank|
     averaged over core protocols (higher = more misaligned evaluators).
  2. For each protocol, compute per-task MEG.
  3. Test correlation: do feedback-heavy protocols (VGS, HPE) have more negative
     MEG on high-mismatch tasks?

This analysis uses existing data from results/full-v2/ — no new API calls needed.

Usage:
  python3 analyze_mismatch_sweep.py
  python3 analyze_mismatch_sweep.py --plot   # save figure
"""

import json, glob, os, math, sys
from collections import defaultdict

RESULTS_DIR = "results/full-v2"
OUT_DIR = "results/analysis"
PLOT = "--plot" in sys.argv

os.makedirs(OUT_DIR, exist_ok=True)

# ── Anchors (from analyze_bench.py) ───────────────────────────────────────────
ANCHORS = {
    "bench-maxcut-g200":        dict(sb=3139,    sr=3517,      dir="max", sentinel=None),
    "bench-circle-packing-n20": dict(sb=0.5039,  sr=2.0,       dir="max", sentinel=None),
    "bench-difference-bases":   dict(sb=14.0,    sr=1.0,       dir="min", sentinel=0),
    "bench-flat-poly-deg50":    dict(sb=2.0489,  sr=1.0,       dir="min", sentinel=None),
    "bench-tsp-100":            dict(sb=50835,   sr=7517,      dir="min", sentinel=0),
    "bench-lj-n41":             dict(sb=-136.0,  sr=-198.0609, dir="min", sentinel=1e30),
    "bench-erdos-overlap":      dict(sb=0.2939,  sr=0.2321,    dir="min", sentinel=None),
    "bench-molecule-qed":       dict(sb=0.60,    sr=1.0,       dir="max", sentinel=None),
}

TASK_LABELS = {
    "bench-maxcut-g200":        "MaxCut",
    "bench-circle-packing-n20": "CircPack",
    "bench-difference-bases":   "DiffBases",
    "bench-flat-poly-deg50":    "FlatPoly",
    "bench-tsp-100":            "TSP-100",
    "bench-lj-n41":             "LJ-n41",
    "bench-erdos-overlap":      "Erdős",
    "bench-molecule-qed":       "MolQED",
}

CORE_PROTOCOLS = {
    "single-shot", "self-refine", "best-of-n", "vgs",
    "homo-chain", "cross-chain", "magicore", "debate", "moa", "hpe",
}

DENOM_PROTOCOLS = {"self-refine", "best-of-n", "vgs"}

# Feedback intensity: mean eval calls per protocol (from budget table in paper)
EVAL_CALLS = {
    "single-shot": 1.0,
    "self-refine": 3.9,
    "best-of-n":   7.9,
    "vgs":        10.4,
    "homo-chain":  4.0,
    "cross-chain": 2.9,
    "magicore":    2.8,
    "debate":      2.9,
    "moa":         3.9,
    "hpe":         1.9,
}

FROZEN_SEEDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_sentinel(s, a):
    if s is None: return True
    if a["sentinel"] is None: return False
    if a["sentinel"] == 0:    return s == 0
    return s >= a["sentinel"] * 0.5

def compute_q(s, a):
    if is_sentinel(s, a): return None
    sb, sr = a["sb"], a["sr"]
    if a["dir"] == "max":
        q = (s - sb) / (sr - sb)
    else:
        q = (sb - s) / (sb - sr)
    return max(0.0, min(1.0, q))

def mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")

def spearman_rho(xs, ys):
    n = len(xs)
    if n < 3: return float("nan")
    def rank(lst):
        s = sorted(range(n), key=lambda i: lst[i])
        r = [0] * n
        for rank_val, idx in enumerate(s):
            r[idx] = rank_val + 1
        return r
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))

# ── Load data ─────────────────────────────────────────────────────────────────

cell_dev = defaultdict(lambda: defaultdict(list))
cell_hidden = defaultdict(lambda: defaultdict(list))
raw_dev = defaultdict(lambda: defaultdict(list))
raw_hidden = defaultdict(lambda: defaultdict(list))

files = glob.glob(f"{RESULTS_DIR}/bench-*_*.json")
for f in files:
    name = os.path.basename(f).replace(".json", "")
    rest = name[len("bench-"):]
    parts = rest.rsplit("_s", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        continue
    seed = int(parts[1])
    if seed not in FROZEN_SEEDS:
        continue

    body = "bench-" + parts[0]
    task = proto = None
    for tid in ANCHORS:
        if body.startswith(tid + "_"):
            task = tid
            proto = body[len(tid) + 1:]
            break
    if task is None:
        continue

    d = json.load(open(f))
    a = ANCHORS[task]

    dev_s = (d.get("bestArtifact") or {}).get("devScore")
    hidden_s = d.get("hiddenScore")

    dq = compute_q(dev_s, a)
    hq = compute_q(hidden_s, a)

    cell_dev[task][proto].append(0.0 if dq is None else dq)
    cell_hidden[task][proto].append(0.0 if hq is None else hq)
    if dev_s is not None and not is_sentinel(dev_s, a):
        raw_dev[task][proto].append(dev_s)
    if hidden_s is not None and not is_sentinel(hidden_s, a):
        raw_hidden[task][proto].append(hidden_s)

# ── Step 1: Compute per-task dev/hidden mismatch ─────────────────────────────

tasks = list(ANCHORS.keys())
protos_all = sorted({p for t in cell_hidden for p in cell_hidden[t]})

# Mismatch metric: Spearman ρ between dev and hidden protocol rankings (core only)
task_mismatch = {}
for t in tasks:
    a = ANCHORS[t]
    common = [p for p in CORE_PROTOCOLS
              if raw_dev[t].get(p) and raw_hidden[t].get(p)]
    if len(common) < 3:
        task_mismatch[t] = float("nan")
        continue

    rev = (a["dir"] == "max")
    dev_sorted = sorted(common, key=lambda p: mean(raw_dev[t][p]), reverse=rev)
    hid_sorted = sorted(common, key=lambda p: mean(raw_hidden[t][p]), reverse=rev)

    dev_rank = {p: i + 1 for i, p in enumerate(dev_sorted)}
    hid_rank = {p: i + 1 for i, p in enumerate(hid_sorted)}

    xs = [dev_rank[p] for p in common]
    ys = [hid_rank[p] for p in common]
    rho = spearman_rho(xs, ys)
    task_mismatch[t] = 1.0 - rho  # higher = more mismatch

# ── Step 2: Compute per-task MEG for core protocols ──────────────────────────

q_hidden = {t: {p: mean(cell_hidden[t][p]) for p in protos_all} for t in tasks}
denom = {}
for t in tasks:
    vals = [q_hidden[t].get(p, 0.0) for p in DENOM_PROTOCOLS if p in q_hidden[t]]
    denom[t] = max(vals) if vals else 0.0

meg_per_task = {}
for p in CORE_PROTOCOLS:
    meg_per_task[p] = {}
    for t in tasks:
        qh = q_hidden[t].get(p, float("nan"))
        meg_per_task[p][t] = qh - denom[t] if not math.isnan(qh) else float("nan")

# ── Step 3: Print mismatch table ─────────────────────────────────────────────

print("=" * 80)
print("VISIBLE/HIDDEN MISMATCH SWEEP — Proxy-Sensitivity Analysis")
print("=" * 80)

print(f"\n{'Task':<12} {'Mismatch':>10}  {'ρ_dev,hid':>10}  Notes")
print("-" * 55)
sorted_tasks = sorted(tasks, key=lambda t: task_mismatch.get(t, 0), reverse=True)
for t in sorted_tasks:
    mm = task_mismatch[t]
    rho = 1.0 - mm if not math.isnan(mm) else float("nan")
    label = TASK_LABELS[t]
    note = ""
    if label == "MaxCut": note = "Type I, near-floor"
    elif label == "MolQED": note = "Type II, QED×SA"
    elif label == "DiffBases": note = "Type I, 95% coverage"
    elif label == "CircPack": note = "Type I, strict tol."
    elif label == "Erdős": note = "Type II, fine mesh"
    elif label == "FlatPoly": note = "Type II, adversarial pts."
    elif label == "TSP-100": note = "Type II, perturbed cities"
    elif label == "LJ-n41": note = "Type II, 3-body corr."
    if math.isnan(mm):
        print(f"{label:<12} {'n/a':>10}  {'n/a':>10}  {note} (insufficient protocols)")
    else:
        print(f"{label:<12} {mm:10.3f}  {rho:10.3f}  {note}")

# ── Step 4: Protocol sensitivity to mismatch ─────────────────────────────────

print(f"\n\n{'Protocol':<15} {'Calls':>6}  ", end="")
for t in sorted_tasks:
    print(f"{TASK_LABELS[t]:>9}", end="")
print(f"  {'AggMEG':>8}  {'Slope':>8}")
print("-" * (15 + 8 + 9 * len(sorted_tasks) + 20))

proto_slopes = {}
for p in sorted(CORE_PROTOCOLS, key=lambda x: EVAL_CALLS.get(x, 0), reverse=True):
    calls = EVAL_CALLS.get(p, 0)
    vals_str = ""
    # Compute slope: regression of MEG on mismatch
    xs, ys = [], []
    for t in sorted_tasks:
        m = meg_per_task[p].get(t, float("nan"))
        mm = task_mismatch.get(t, float("nan"))
        if math.isnan(m):
            vals_str += f"{'n/a':>9}"
        else:
            vals_str += f"{m:>+9.3f}"
        if not math.isnan(m) and not math.isnan(mm):
            xs.append(mm)
            ys.append(m)

    # Simple linear regression slope
    if len(xs) >= 3:
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den > 0 else 0
        proto_slopes[p] = slope
    else:
        slope = float("nan")
        proto_slopes[p] = float("nan")

    agg = mean([meg_per_task[p][t] for t in tasks
                if not math.isnan(meg_per_task[p].get(t, float("nan")))])

    slope_str = f"{slope:>+8.3f}" if not math.isnan(slope) else "     n/a"
    print(f"{p:<15} {calls:6.1f}  {vals_str}  {agg:>+8.3f}  {slope_str}")

# ── Step 5: Correlation between eval calls and mismatch sensitivity ──────────

print(f"\n\nCorrelation: eval_calls × MEG_slope (over {len(proto_slopes)} core protocols)")
valid_protos = [p for p in CORE_PROTOCOLS if not math.isnan(proto_slopes.get(p, float("nan")))]
if len(valid_protos) >= 3:
    calls_list = [EVAL_CALLS[p] for p in valid_protos]
    slopes_list = [proto_slopes[p] for p in valid_protos]
    rho = spearman_rho(calls_list, slopes_list)
    print(f"  Spearman ρ(eval_calls, slope) = {rho:.3f}")
    if rho < -0.3:
        print("  → Feedback-heavy protocols have MORE negative slope (degrade faster on high-mismatch tasks)")
        print("  → Supports the 'information filter' hypothesis")
    elif rho > 0.3:
        print("  → Feedback-heavy protocols have LESS negative slope (more robust to mismatch)")
        print("  → Evidence against simple proxy-sensitivity narrative")
    else:
        print("  → No clear relationship between feedback intensity and mismatch sensitivity")
else:
    print("  Insufficient data for correlation")

# ── Step 6: Key diagnostic ───────────────────────────────────────────────────

print("\n\nKey diagnostic: VGS and HPE (highest and lowest feedback users)")
for p in ["vgs", "hpe"]:
    if p in proto_slopes and not math.isnan(proto_slopes[p]):
        calls = EVAL_CALLS[p]
        slope = proto_slopes[p]
        print(f"  {p:>4}: {calls:.0f} eval calls, slope = {slope:+.3f}")

print("\n\nConclusion:")
# Count tasks with non-trivial mismatch
n_mismatch = sum(1 for t in tasks if not math.isnan(task_mismatch.get(t, float("nan")))
                 and task_mismatch.get(t, 0) > 0.01)
print(f"  Only {n_mismatch}/8 tasks show non-trivial dev/hidden mismatch (ρ < 0.99).")
print(f"  The remaining {8 - n_mismatch} tasks have ρ ≈ 1.0, clustering all data at mismatch=0.")
print(f"  With {n_mismatch} effective data points, the slope regression is underpowered.")
print(f"  No reliable conclusion about proxy sensitivity vs feedback intensity can be drawn")
print(f"  from the current task suite. Testing the hypothesis requires tasks with")
print(f"  parametrically varying dev/hidden gaps (e.g., MaxCut with different edge-subset %).")

# ── Write results ─────────────────────────────────────────────────────────────

with open(f"{OUT_DIR}/mismatch_sweep.tsv", "w") as fh:
    fh.write("protocol\teval_calls\tslope\tagg_meg\t" +
             "\t".join(TASK_LABELS[t] for t in sorted_tasks) + "\n")
    for p in sorted(CORE_PROTOCOLS, key=lambda x: EVAL_CALLS.get(x, 0), reverse=True):
        calls = EVAL_CALLS.get(p, 0)
        slope = proto_slopes.get(p, float("nan"))
        agg = mean([meg_per_task[p][t] for t in tasks
                    if not math.isnan(meg_per_task[p].get(t, float("nan")))])
        megs = [f"{meg_per_task[p].get(t, float('nan')):+.4f}" for t in sorted_tasks]
        slope_s = f"{slope:+.4f}" if not math.isnan(slope) else "nan"
        fh.write(f"{p}\t{calls}\t{slope_s}\t{agg:+.4f}\t" + "\t".join(megs) + "\n")

print(f"\nWrote {OUT_DIR}/mismatch_sweep.tsv")

# ── Optional plot ─────────────────────────────────────────────────────────────

if PLOT:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Panel A: MEG vs mismatch for each protocol
        colors = {
            "vgs": "tab:red", "hpe": "tab:purple",
            "moa": "tab:blue", "debate": "tab:orange",
            "magicore": "tab:green", "best-of-n": "tab:gray",
            "self-refine": "tab:brown", "single-shot": "black",
            "homo-chain": "tab:olive", "cross-chain": "tab:cyan",
        }
        for p in CORE_PROTOCOLS:
            xs, ys = [], []
            for t in sorted_tasks:
                mm = task_mismatch.get(t, float("nan"))
                m = meg_per_task[p].get(t, float("nan"))
                if not math.isnan(mm) and not math.isnan(m):
                    xs.append(mm)
                    ys.append(m)
            if xs:
                ax1.scatter(xs, ys, color=colors.get(p, "gray"),
                           label=p, alpha=0.7, s=40)

        ax1.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax1.set_xlabel("Task mismatch (1 − ρ)")
        ax1.set_ylabel("MEG")
        ax1.set_title("(a) Per-task MEG vs evaluator mismatch")
        ax1.legend(fontsize=6, ncol=2, loc="lower left")

        # Panel B: Slope vs eval calls
        valid_p = [p for p in CORE_PROTOCOLS if not math.isnan(proto_slopes.get(p, float("nan")))]
        calls = [EVAL_CALLS[p] for p in valid_p]
        slopes = [proto_slopes[p] for p in valid_p]
        ax2.scatter(calls, slopes, color=[colors.get(p, "gray") for p in valid_p], s=60)
        for p, c, s in zip(valid_p, calls, slopes):
            ax2.annotate(p, (c, s), fontsize=7, ha="left", va="bottom")
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax2.set_xlabel("Mean eval calls")
        ax2.set_ylabel("Slope (MEG ~ mismatch)")
        ax2.set_title("(b) Feedback intensity vs mismatch sensitivity")

        fig.suptitle("Proxy-Sensitivity Analysis", fontsize=12)
        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/mismatch_sweep.pdf", bbox_inches="tight")
        print(f"Figure saved to figures/mismatch_sweep.pdf")
    except ImportError:
        print("matplotlib not installed — skipping plot")
