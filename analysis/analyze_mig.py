"""
Compute Marginal Interaction Gain (MIG) for all multi-agent protocols.

MIG(p) = E[Q_protocol] - E[Q_no-interact]

Where no-interact is the best single-agent output from the same backbone set:
  - Homo-Claude protocols:   no-interact = single-shot (Claude)
  - Mixed-frontier protocols: no-interact = max(SS-Claude, SS-GPT4o, SS-Gemini) per task
  - Homo-DeepSeek protocols: no-interact = single-shot-deepseek (proxy: best-of-n-deepseek not available as SS)
  - Mixed-OSS protocols:     no-interact = best per-task across OSS single-agent runs

For MoA, the existing ablation gives a cleaner decomposition (synthesis vs best-of-proposals),
but this script extends MIG to all protocol families using the single-agent counterfactual.

Run: python3 analyze_mig.py
"""

import json, glob, os
from collections import defaultdict

RESULTS_DIR  = "results/full-v2"
OUT_DIR      = "results/analysis"
FROZEN_SEEDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Anchors (copied from analyze_bench.py) ────────────────────────────────────
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
    "bench-maxcut-g200": "MaxCut", "bench-circle-packing-n20": "CircPack",
    "bench-difference-bases": "DiffBases", "bench-flat-poly-deg50": "FlatPoly",
    "bench-tsp-100": "TSP-100", "bench-lj-n41": "LJ-n41",
    "bench-erdos-overlap": "Erdős", "bench-molecule-qed": "MolQED",
}

CORE_TASKS = list(ANCHORS.keys())[:8]  # 8 core tasks

def is_sentinel(s, a):
    if s is None: return True
    if a["sentinel"] is None: return False
    if a["sentinel"] == 0: return s == 0
    return s >= a["sentinel"] * 0.5

def compute_q(s, a):
    if is_sentinel(s, a): return None
    sb, sr = a["sb"], a["sr"]
    if a["dir"] == "max":
        q = (s - sb) / (sr - sb)
    else:
        q = (sb - s) / (sb - sr)
    return max(0.0, min(1.0, q))

# ── Load all results ──────────────────────────────────────────────────────────
cell = defaultdict(lambda: defaultdict(list))  # cell[task][proto] = [Q, ...]

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
    if task not in CORE_TASKS:
        continue

    d = json.load(open(f))
    hs = d.get("hiddenScore")
    q = compute_q(hs, ANCHORS[task])
    cell[task][proto].append(q if q is not None else 0.0)

def mean_q(task, proto):
    vals = cell[task].get(proto, [])
    if not vals:
        return None
    return sum(vals) / len(vals)

def n_seeds(task, proto):
    return len(cell[task].get(proto, []))

# ── Define MIG groups ─────────────────────────────────────────────────────────
# Each group: (protocol, no-interact baseline definition)
# no-interact = list of single-agent protocols; per task, take max of their means

MIG_GROUPS = {
    # Homo-Claude protocols: no-interact = single-shot Claude
    "MAgICoRe": ("magicore", ["single-shot"]),
    "Debate": ("debate", ["single-shot"]),
    "HPE": ("hpe", ["single-shot"]),
    "Homo-chain": ("homo-chain", ["single-shot"]),
    "MoA-same": ("moa-same-model", ["single-shot"]),

    # Mixed-frontier: no-interact = best of {SS-Claude, SS-GPT4o, SS-Gemini}
    "MAgICoRe-mixed": ("magicore-mixed", ["single-shot", "single-shot-gpt4o", "single-shot-gemini"]),
    "Debate-mixed": ("debate-mixed", ["single-shot", "single-shot-gpt4o", "single-shot-gemini"]),
    "HPE-mixed": ("hpe-mixed", ["single-shot", "single-shot-gpt4o", "single-shot-gemini"]),
    "MoA-diverse": ("moa", ["single-shot", "single-shot-gpt4o", "single-shot-gemini"]),

    # Cross-chain uses Claude+GPT4o+Gemini sequentially
    "Cross-chain": ("cross-chain", ["single-shot", "single-shot-gpt4o", "single-shot-gemini"]),
}

# ── Compute MIG ───────────────────────────────────────────────────────────────
print("=" * 90)
print("MARGINAL INTERACTION GAIN (MIG) — Extended to All Multi-Agent Protocols")
print("=" * 90)
print()

# Per-task MIG table
header = f"{'Protocol':<22} {'Config':<8}"
for t in CORE_TASKS:
    header += f" {TASK_LABELS[t]:>8}"
header += f" {'MeanMIG':>8} {'n_tasks':>7}"
print(header)
print("-" * len(header))

results = []

for label, (proto, baselines) in MIG_GROUPS.items():
    config = "mixed" if "mixed" in label or "diverse" in label.lower() or label == "Cross-chain" else "homo"
    row = f"{label:<22} {config:<8}"
    mig_vals = []

    for task in CORE_TASKS:
        proto_q = mean_q(task, proto)
        n = n_seeds(task, proto)

        # Compute no-interact baseline: max of available baseline means
        baseline_means = []
        for bp in baselines:
            bq = mean_q(task, bp)
            if bq is not None:
                baseline_means.append(bq)

        if proto_q is not None and baseline_means and n >= 3:
            no_interact = max(baseline_means)
            mig = proto_q - no_interact
            mig_vals.append(mig)
            row += f" {mig:>+8.3f}"
        else:
            row += f" {'---':>8}"

    if mig_vals:
        mean_mig = sum(mig_vals) / len(mig_vals)
        row += f" {mean_mig:>+8.3f} {len(mig_vals):>7}"
        results.append((label, config, mean_mig, mig_vals))
    else:
        row += f" {'---':>8} {'0':>7}"

    print(row)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("SUMMARY: Same-model vs Diverse-model MIG")
print("=" * 90)
print()

homo_migs = [(l, m, v) for l, c, m, v in results if c == "homo"]
mixed_migs = [(l, m, v) for l, c, m, v in results if c == "mixed"]

print("Same-model (homo) protocols:")
for label, mean_mig, vals in homo_migs:
    sign = "+" if mean_mig > 0 else ""
    print(f"  {label:<22} MIG = {sign}{mean_mig:.3f}  ({len(vals)} tasks)")

if homo_migs:
    all_homo = [m for _, m, _ in homo_migs]
    print(f"  {'Grand mean':<22} MIG = {sum(all_homo)/len(all_homo):+.3f}")

print()
print("Diverse-model (mixed) protocols:")
for label, mean_mig, vals in mixed_migs:
    sign = "+" if mean_mig > 0 else ""
    print(f"  {label:<22} MIG = {sign}{mean_mig:.3f}  ({len(vals)} tasks)")

if mixed_migs:
    all_mixed = [m for _, m, _ in mixed_migs]
    print(f"  {'Grand mean':<22} MIG = {sum(all_mixed)/len(all_mixed):+.3f}")

# ── Interaction-specific view: homo vs mixed for same protocol family ─────────
print()
print("=" * 90)
print("PER-FAMILY COMPARISON: Does diversity change MIG sign?")
print("=" * 90)
families = [
    ("MAgICoRe", "MAgICoRe-mixed"),
    ("Debate", "Debate-mixed"),
    ("HPE", "HPE-mixed"),
    ("MoA-same", "MoA-diverse"),
]

results_dict = {l: (c, m, v) for l, c, m, v in results}

print(f"\n{'Family':<15} {'Homo MIG':>10} {'Mixed MIG':>10} {'Δ(mixed-homo)':>14}")
print("-" * 52)
for homo_label, mixed_label in families:
    if homo_label in results_dict and mixed_label in results_dict:
        _, hm, _ = results_dict[homo_label]
        _, mm, _ = results_dict[mixed_label]
        delta = mm - hm
        print(f"{homo_label.replace('-same',''):<15} {hm:>+10.3f} {mm:>+10.3f} {delta:>+14.3f}")

# ── Write TSV ─────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "mig_extended.tsv")
with open(out_path, "w") as f:
    cols = ["protocol", "config"] + [TASK_LABELS[t] for t in CORE_TASKS] + ["MeanMIG", "n_tasks"]
    f.write("\t".join(cols) + "\n")
    for label, (proto, baselines) in MIG_GROUPS.items():
        config = "mixed" if "mixed" in label or "diverse" in label.lower() or label == "Cross-chain" else "homo"
        row_vals = [label, config]
        mig_vals = []
        for task in CORE_TASKS:
            proto_q = mean_q(task, proto)
            n = n_seeds(task, proto)
            baseline_means = []
            for bp in baselines:
                bq = mean_q(task, bp)
                if bq is not None:
                    baseline_means.append(bq)
            if proto_q is not None and baseline_means and n >= 3:
                no_interact = max(baseline_means)
                mig = proto_q - no_interact
                mig_vals.append(mig)
                row_vals.append(f"{mig:+.4f}")
            else:
                row_vals.append("")
        if mig_vals:
            row_vals.append(f"{sum(mig_vals)/len(mig_vals):+.4f}")
            row_vals.append(str(len(mig_vals)))
        else:
            row_vals.extend(["", "0"])
        f.write("\t".join(row_vals) + "\n")

print(f"\nTSV written to {out_path}")
