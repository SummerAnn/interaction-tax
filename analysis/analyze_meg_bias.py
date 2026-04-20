"""
Bootstrap-based MEG denominator bias correction.

Problem: MEG = E[Q_protocol] - max(E[Q_SR], E[Q_BoN], E[Q_VGS]).
The denominator takes the max of three sample means, each estimated from
n seeds.  E[max(X1, X2, X3)] >= max(E[X1], E[X2], E[X3]) only when the
estimators have variance — the max operator applied to noisy estimates is
biased upward relative to max of the true means.  This pushes MEG
conservatively negative by ~0.005/task.

Fix: resample each control's seed-level Q scores B=10,000 times, compute
the expected bootstrap denominator, and subtract the raw (plug-in)
denominator to estimate the bias.  Then correct MEG by subtracting
the bias from the denominator.

Run:  python3 analyze_meg_bias.py
Output: results/analysis/meg_bias_correction.tsv  (and console summary)
"""

import json, glob, os, math, random
from collections import defaultdict

random.seed(42)

RESULTS_DIR  = "results/full-v2"
OUT_DIR      = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

B = 10_000  # bootstrap resamples

FROZEN_SEEDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
MIN_SEEDS    = 5

# ── Anchors (same as analyze_bench.py) ────────────────────────────────────────
ANCHORS = {
    "bench-maxcut-g200":        dict(sb=3139,    sr=3517,      dir="max", sentinel=None),
    "bench-circle-packing-n20": dict(sb=0.5039,  sr=2.0,       dir="max", sentinel=None),
    "bench-difference-bases":   dict(sb=14.0,    sr=1.0,       dir="min", sentinel=0),
    "bench-flat-poly-deg50":    dict(sb=2.0489,  sr=1.0,       dir="min", sentinel=None),
    "bench-tsp-100":            dict(sb=50835,   sr=7517,      dir="min", sentinel=0),
    "bench-lj-n41":             dict(sb=-136.0,  sr=-198.0609, dir="min", sentinel=1e30),
    "bench-erdos-overlap":      dict(sb=0.2939,  sr=0.2321,    dir="min", sentinel=None),
    "bench-molecule-qed":       dict(sb=0.60,    sr=1.0,       dir="max", sentinel=None),
    "bench-graph-coloring-g80": dict(sb=16,      sr=10,        dir="min", sentinel=None),
    "bench-tsp-50":             dict(sb=25000,   sr=5400,      dir="min", sentinel=0),
}

TASK_LABELS = {
    "bench-maxcut-g200":        "MaxCut",
    "bench-circle-packing-n20": "CircPack",
    "bench-difference-bases":   "DiffBases",
    "bench-flat-poly-deg50":    "FlatPoly",
    "bench-tsp-100":            "TSP-100",
    "bench-lj-n41":             "LJ-n41",
    "bench-erdos-overlap":      "Erdos",
    "bench-molecule-qed":       "MolQED",
    "bench-graph-coloring-g80": "GColor",
    "bench-tsp-50":             "TSP-50",
}

DENOM_PROTOCOLS = {"self-refine", "best-of-n", "vgs"}

CORE_PROTOCOLS = {
    "single-shot", "self-refine", "best-of-n", "vgs",
    "homo-chain", "cross-chain", "magicore", "debate", "moa", "hpe",
}

EXCLUDE_FROM_AGG = set()

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

# ── Load all results ──────────────────────────────────────────────────────────
# cell_hidden[task][proto] = list of per-seed Q scores (invalid -> 0.0)
cell_hidden = defaultdict(lambda: defaultdict(list))

files = glob.glob(f"{RESULTS_DIR}/bench-*_*.json")
for f in files:
    name  = os.path.basename(f).replace(".json", "")
    rest  = name[len("bench-"):]
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
            task  = tid
            proto = body[len(tid) + 1:]
            break
    if task is None:
        continue

    d  = json.load(open(f))
    a  = ANCHORS[task]

    hidden_s = d.get("hiddenScore")
    hq       = compute_q(hidden_s, a)
    cell_hidden[task][proto].append(0.0 if hq is None else hq)

# ── Mean Q per (task, proto) ──────────────────────────────────────────────────
tasks  = list(ANCHORS.keys())
protos = sorted({p for t in cell_hidden for p in cell_hidden[t]})

q_hidden = {t: {p: mean(cell_hidden[t][p]) for p in protos} for t in tasks}

# ── Raw (plug-in) denominator ────────────────────────────────────────────────
denom_raw = {}
for t in tasks:
    vals = [q_hidden[t].get(p, 0.0) for p in DENOM_PROTOCOLS if p in q_hidden[t]]
    denom_raw[t] = max(vals) if vals else 0.0

# ── Bootstrap bias estimation ────────────────────────────────────────────────
# For each task:
#   1. Let Q_c[p] = list of per-seed Q scores for control protocol p.
#   2. For b = 1..B:
#        For each control p, resample Q_c[p] with replacement -> compute mean.
#        boot_denom_b = max over the three resampled means.
#   3. bias = E[boot_denom] - raw_denom
#      (positive bias = denominator is inflated = MEG is too negative)

bias = {}
boot_denom_mean = {}

print("=" * 80)
print("MEG DENOMINATOR BIAS CORRECTION (bootstrap, B={:,})".format(B))
print("=" * 80)
print()

print(f"{'Task':<12} {'raw_denom':>10} {'E[boot_d]':>10} {'bias':>10} {'n_SR':>5} {'n_BoN':>5} {'n_VGS':>5}")
print("-" * 68)

for t in tasks:
    # Gather per-seed Q vectors for each control
    control_seeds = {}
    for p in DENOM_PROTOCOLS:
        control_seeds[p] = cell_hidden[t].get(p, [])

    # Only estimate bias if we have data for at least one control
    has_data = any(len(v) > 0 for v in control_seeds.values())
    if not has_data:
        bias[t] = 0.0
        boot_denom_mean[t] = denom_raw[t]
        continue

    boot_denoms = []
    for _ in range(B):
        boot_means = []
        for p in DENOM_PROTOCOLS:
            seeds = control_seeds[p]
            if not seeds:
                boot_means.append(0.0)
            else:
                resamp = [random.choice(seeds) for _ in seeds]
                boot_means.append(mean(resamp))
        boot_denoms.append(max(boot_means))

    boot_denom_mean[t] = mean(boot_denoms)
    bias[t] = boot_denom_mean[t] - denom_raw[t]

    ns = {p: len(control_seeds[p]) for p in DENOM_PROTOCOLS}
    print(f"{TASK_LABELS[t]:<12} {denom_raw[t]:>10.4f} {boot_denom_mean[t]:>10.4f} {bias[t]:>+10.4f} "
          f"{ns['self-refine']:>5} {ns['best-of-n']:>5} {ns['vgs']:>5}")

total_bias = mean([bias[t] for t in tasks])
print(f"\n  Mean bias across {len(tasks)} tasks: {total_bias:+.5f}")

# ── Corrected denominator ────────────────────────────────────────────────────
denom_corrected = {}
for t in tasks:
    denom_corrected[t] = denom_raw[t] - bias[t]  # subtract upward bias

# ── Compute raw and corrected MEG for all core protocols ─────────────────────
print()
print("=" * 80)
print("RAW vs BIAS-CORRECTED MEG — CORE 10 PROTOCOLS")
print("=" * 80)
print()

agg_tasks = [t for t in tasks if t not in EXCLUDE_FROM_AGG]

# Header
hdr_tasks = " ".join(f"{TASK_LABELS[t]:>9}" for t in tasks)
print(f"{'Protocol':<16} {'rawAggMEG':>10} {'corrAggMEG':>11} {'delta':>8}  per-task raw -> corrected")
print("-" * 80)

tsv_rows = []
for proto in sorted(CORE_PROTOCOLS):
    raw_megs  = []
    corr_megs = []
    per_task_detail = []
    for t in agg_tasks:
        qh = q_hidden[t].get(proto, float("nan"))
        if math.isnan(qh):
            per_task_detail.append(f"{TASK_LABELS[t]}:nan")
            continue
        raw_meg  = qh - denom_raw[t]
        corr_meg = qh - denom_corrected[t]
        raw_megs.append(raw_meg)
        corr_megs.append(corr_meg)

    raw_agg  = mean(raw_megs)  if raw_megs  else float("nan")
    corr_agg = mean(corr_megs) if corr_megs else float("nan")
    delta    = corr_agg - raw_agg if not (math.isnan(raw_agg) or math.isnan(corr_agg)) else float("nan")

    print(f"{proto:<16} {raw_agg:>+10.4f} {corr_agg:>+11.4f} {delta:>+8.4f}")
    tsv_rows.append((proto, raw_agg, corr_agg, delta))

# ── MoA-diverse CI analysis ─────────────────────────────────────────────────
# Bootstrap the full corrected AggMEG for "moa" to get a CI and check if
# the correction moves the CI meaningfully.

print()
print("=" * 80)
print("MOA-DIVERSE CONFIDENCE INTERVAL — RAW vs CORRECTED")
print("=" * 80)
print()

N_BOOT_CI = 10_000

for proto_label in ["moa"]:
    # Raw CI (same method as analyze_bench.py — resample protocol seeds, fixed denom)
    raw_boot = []
    corr_boot = []
    for _ in range(N_BOOT_CI):
        raw_task_megs  = []
        corr_task_megs = []
        for t in agg_tasks:
            raw_seeds = cell_hidden[t].get(proto_label, [])
            if not raw_seeds:
                continue
            resamp = [random.choice(raw_seeds) for _ in raw_seeds]
            raw_task_megs.append(mean(resamp)  - denom_raw[t])
            corr_task_megs.append(mean(resamp) - denom_corrected[t])
        if raw_task_megs:
            raw_boot.append(mean(raw_task_megs))
        if corr_task_megs:
            corr_boot.append(mean(corr_task_megs))

    raw_boot.sort()
    corr_boot.sort()

    def ci(samples, alpha=0.025):
        lo = samples[int(alpha * len(samples))]
        hi = samples[int((1 - alpha) * len(samples))]
        return lo, hi

    raw_lo,  raw_hi  = ci(raw_boot)
    corr_lo, corr_hi = ci(corr_boot)
    raw_med  = raw_boot[len(raw_boot) // 2]
    corr_med = corr_boot[len(corr_boot) // 2]

    print(f"  Protocol: {proto_label}")
    print(f"  Raw   AggMEG:  median={raw_med:+.4f}  95% CI=[{raw_lo:+.4f}, {raw_hi:+.4f}]")
    print(f"  Corr  AggMEG:  median={corr_med:+.4f}  95% CI=[{corr_lo:+.4f}, {corr_hi:+.4f}]")
    print(f"  CI shift:      lo {corr_lo - raw_lo:+.4f}, hi {corr_hi - raw_hi:+.4f}")

    # Does the CI bracket change qualitative conclusion?
    raw_covers_zero  = raw_lo  <= 0 <= raw_hi
    corr_covers_zero = corr_lo <= 0 <= corr_hi
    print(f"  Raw  CI covers zero:  {raw_covers_zero}")
    print(f"  Corr CI covers zero:  {corr_covers_zero}")
    if raw_covers_zero == corr_covers_zero:
        print(f"  ==> No qualitative change: CI {'covers' if corr_covers_zero else 'excludes'} zero in both cases.")
    else:
        if raw_covers_zero and not corr_covers_zero:
            print(f"  ==> MEANINGFUL CHANGE: correction moves CI to exclude zero (stronger claim).")
        else:
            print(f"  ==> MEANINGFUL CHANGE: correction moves CI to cover zero (weaker claim).")

# ── Also check all core protocols for sign changes ───────────────────────────
print()
print("=" * 80)
print("SIGN CHANGES AFTER CORRECTION (all core protocols)")
print("=" * 80)
print()
print(f"{'Protocol':<16} {'raw_AggMEG':>11} {'corr_AggMEG':>12} {'raw_sign':>9} {'corr_sign':>10} {'flipped?':>9}")
print("-" * 72)
for proto, raw_agg, corr_agg, delta in sorted(tsv_rows, key=lambda r: -r[2]):
    raw_sign  = "+" if raw_agg  > 0 else ("-" if raw_agg  < 0 else "0")
    corr_sign = "+" if corr_agg > 0 else ("-" if corr_agg < 0 else "0")
    flipped   = raw_sign != corr_sign
    print(f"{proto:<16} {raw_agg:>+11.4f} {corr_agg:>+12.4f} {raw_sign:>9} {corr_sign:>10} {'YES' if flipped else 'no':>9}")

# ── Write TSV output ─────────────────────────────────────────────────────────
tsv_path = f"{OUT_DIR}/meg_bias_correction.tsv"
with open(tsv_path, "w") as fh:
    # Section 1: per-task bias
    fh.write("# Per-task denominator bias estimates\n")
    fh.write("task\traw_denom\tE_boot_denom\tbias\tn_SR\tn_BoN\tn_VGS\tcorr_denom\n")
    for t in tasks:
        ns = {p: len(cell_hidden[t].get(p, [])) for p in DENOM_PROTOCOLS}
        fh.write(f"{TASK_LABELS[t]}\t{denom_raw[t]:.6f}\t{boot_denom_mean[t]:.6f}\t{bias[t]:+.6f}\t"
                 f"{ns['self-refine']}\t{ns['best-of-n']}\t{ns['vgs']}\t{denom_corrected[t]:.6f}\n")
    fh.write(f"MEAN\t{mean([denom_raw[t] for t in tasks]):.6f}\t"
             f"{mean([boot_denom_mean[t] for t in tasks]):.6f}\t"
             f"{total_bias:+.6f}\t\t\t\t"
             f"{mean([denom_corrected[t] for t in tasks]):.6f}\n")

    fh.write("\n")

    # Section 2: raw vs corrected MEG
    fh.write("# Raw vs bias-corrected MEG for core 10 protocols\n")
    fh.write("protocol\traw_AggMEG\tcorr_AggMEG\tdelta\tsign_flipped\n")
    for proto, raw_agg, corr_agg, delta in sorted(tsv_rows, key=lambda r: -r[2]):
        raw_sign  = "+" if raw_agg > 0 else "-"
        corr_sign = "+" if corr_agg > 0 else "-"
        flipped   = raw_sign != corr_sign
        fh.write(f"{proto}\t{raw_agg:+.6f}\t{corr_agg:+.6f}\t{delta:+.6f}\t{flipped}\n")

    fh.write("\n")

    # Section 3: MoA CI
    fh.write("# MoA-diverse CI comparison\n")
    fh.write("variant\tmedian\tCI_lo\tCI_hi\tcovers_zero\n")
    fh.write(f"raw\t{raw_med:+.6f}\t{raw_lo:+.6f}\t{raw_hi:+.6f}\t{raw_covers_zero}\n")
    fh.write(f"corrected\t{corr_med:+.6f}\t{corr_lo:+.6f}\t{corr_hi:+.6f}\t{corr_covers_zero}\n")

print()
print(f"Wrote {tsv_path}")
