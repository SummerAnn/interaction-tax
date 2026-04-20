"""
Eval-intensity × MEG correlation: information-filter test.

If protocols act as information filters, then more interaction with the dev
evaluator should hurt MEG on tasks where dev≠hidden (proxy divergence),
but not on tasks where dev≈hidden (proxy alignment).

This analysis correlates K-used (mean eval calls per protocol) with MEG
on each task. A negative correlation on proxy-divergent tasks (MaxCut,
MolQED) but flat on proxy-aligned tasks (FlatPoly) supports the
information-filter lens without requiring new experiments.

Run: python3 analyze_eval_intensity.py
"""

import json, glob, os, math
from collections import defaultdict

RESULTS_DIR = "results/full-v2"
OUT_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

FROZEN_SEEDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

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

# Proxy divergence classification (from paper design)
PROXY_DIVERGENT = {"bench-maxcut-g200", "bench-molecule-qed", "bench-circle-packing-n20"}
PROXY_ALIGNED   = {"bench-flat-poly-deg50"}  # dev=hidden

# Only Claude backbone protocols (controls + multi-agent)
CORE_PROTOCOLS = {
    "single-shot", "self-refine", "best-of-n", "vgs",
    "homo-chain", "cross-chain", "magicore", "debate", "moa", "hpe",
}

DENOM_PROTOCOLS = {"self-refine", "best-of-n", "vgs"}


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
    return sum(lst)/len(lst) if lst else float("nan")


# ── Load data ─────────────────────────────────────────────────────────────────
# Per (task, proto): list of (Q_hidden, K_used) tuples
data = defaultdict(lambda: defaultdict(list))

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
            proto = body[len(tid)+1:]
            break
    if task is None or proto not in CORE_PROTOCOLS:
        continue

    d = json.load(open(f))
    a = ANCHORS[task]

    hidden_s = d.get("hiddenScore")
    hq = compute_q(hidden_s, a)
    hq_val = 0.0 if hq is None else hq

    bt = d.get("budgetTrace", {})
    eval_calls = bt.get("evalCalls", [])
    k_used = len(eval_calls)

    data[task][proto].append((hq_val, k_used))

# ── Compute MEG denominators ──────────────────────────────────────────────────
q_hidden = {}
for t in ANCHORS:
    q_hidden[t] = {}
    for p in CORE_PROTOCOLS:
        vals = [hq for hq, _ in data[t][p]]
        q_hidden[t][p] = mean(vals) if vals else float("nan")

denom = {}
for t in ANCHORS:
    vals = [q_hidden[t].get(p, 0.0) for p in DENOM_PROTOCOLS if p in q_hidden[t]]
    denom[t] = max(vals) if vals else 0.0

# ── Per-protocol mean K-used and MEG ──────────────────────────────────────────
def spearman_rho(xs, ys):
    n = len(xs)
    if n < 3: return float("nan"), n
    def rank(lst):
        indexed = sorted(range(n), key=lambda i: lst[i])
        r = [0]*n
        for rank_val, idx in enumerate(indexed):
            r[idx] = rank_val + 1
        return r
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    rho = 1 - 6*d2/(n*(n*n-1))
    return rho, n


print("=" * 80)
print("EVAL-INTENSITY × MEG CORRELATION (Information-Filter Test)")
print("=" * 80)
print()

# Write detailed TSV
with open(f"{OUT_DIR}/eval_intensity.tsv", "w") as fh:
    fh.write("task\tprotocol\tmean_K_used\tmean_Q_hidden\tMEG\tn_seeds\n")
    for t in ANCHORS:
        for p in sorted(CORE_PROTOCOLS):
            if not data[t][p]:
                continue
            qs = [hq for hq, _ in data[t][p]]
            ks = [k for _, k in data[t][p]]
            meg = mean(qs) - denom[t]
            fh.write(f"{TASK_LABELS[t]}\t{p}\t{mean(ks):.1f}\t{mean(qs):.4f}\t{meg:+.4f}\t{len(qs)}\n")

print("Wrote", f"{OUT_DIR}/eval_intensity.tsv")

# Per-task correlation
print()
print(f"{'Task':<12} {'rho(K,MEG)':>12} {'n_protos':>10} {'proxy_type':>14}")
print("-" * 52)

task_rhos = []
for t in ANCHORS:
    tl = TASK_LABELS[t]
    xs, ys = [], []
    for p in CORE_PROTOCOLS:
        if not data[t][p]:
            continue
        qs = [hq for hq, _ in data[t][p]]
        ks = [k for _, k in data[t][p]]
        meg = mean(qs) - denom[t]
        if not math.isnan(meg):
            xs.append(mean(ks))
            ys.append(meg)

    rho, n = spearman_rho(xs, ys)
    ptype = "divergent" if t in PROXY_DIVERGENT else ("aligned" if t in PROXY_ALIGNED else "mixed")
    task_rhos.append((tl, rho, n, ptype))
    rho_str = f"{rho:+.3f}" if not math.isnan(rho) else "  nan"
    print(f"{tl:<12} {rho_str:>12} {n:>10} {ptype:>14}")

print()

# Summary: is the pattern consistent with information-filter?
div_rhos = [r for _, r, n, pt in task_rhos if pt == "divergent" and not math.isnan(r)]
ali_rhos = [r for _, r, n, pt in task_rhos if pt == "aligned" and not math.isnan(r)]
mix_rhos = [r for _, r, n, pt in task_rhos if pt == "mixed" and not math.isnan(r)]

print("SUMMARY:")
if div_rhos:
    print(f"  Proxy-divergent tasks:  mean rho = {mean(div_rhos):+.3f}  (expect negative)")
if ali_rhos:
    print(f"  Proxy-aligned tasks:    mean rho = {mean(ali_rhos):+.3f}  (expect ~zero)")
if mix_rhos:
    print(f"  Other tasks:            mean rho = {mean(mix_rhos):+.3f}")

print()
print("Interpretation: If proxy-divergent rho is strongly negative and")
print("proxy-aligned rho is near zero, this supports the information-filter")
print("hypothesis: more dev-evaluator interaction hurts on tasks where")
print("the dev proxy diverges from the hidden objective.")
