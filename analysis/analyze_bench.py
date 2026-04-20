"""
Agent4Science-Bench — Full Analysis (paper-safe v2)
Computes: Q scores, MEG table, dev/hidden rank Spearman, brain-diversity delta
All sentinel-aware (hidden=0 for diff-bases/tsp-100, hidden>=1e10 for lj-n41)

Paper-safety guarantees:
  - FROZEN_SEEDS: seeds {1..10} are loaded (1-5 original + 6-10 extended 2026-04-16)
  - MIN_SEEDS: cells with fewer than MIN_SEEDS seeds are flagged in output
  - EXCLUDE_FROM_AGG: Erdős excluded from AggMEG (stale after verifier fix 2026-04-09)

Run: python3 analyze_bench.py
Outputs: results/analysis/meg_full.tsv, rank_all.tsv, brain_diversity.tsv
"""

import json, glob, os, itertools, math
from collections import defaultdict

RESULTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "full-v2")
OUT_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "analysis")
FROZEN_SEEDS     = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}   # seeds 1-5 original + 6-10 extended run (2026-04-16)
# MoA arms use 10 seeds (paper design); all other protocols use seeds 1-5 only.
MOA_PROTOCOLS    = {"moa", "moa-same-model", "moa-nosynth"}
MAX_SEED_DEFAULT = 5   # non-MoA protocols capped at seed 5
MIN_SEEDS        = 5                   # flag any cell with fewer seeds in output
EXCLUDE_FROM_AGG = {"bench-graph-coloring-g80"}  # supplemental task; paper uses 9 core tasks (TSP-50 added 2026-04-18 after anchor calibration)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Anchors ────────────────────────────────────────────────────────────────────
ANCHORS = {
    "bench-maxcut-g200":        dict(sb=3139,    sr=3517,      dir="max", sentinel=None),
    "bench-circle-packing-n20": dict(sb=0.5039,  sr=2.0,       dir="max", sentinel=None),
    "bench-difference-bases":   dict(sb=14.0,    sr=1.0,       dir="min", sentinel=0),
    "bench-flat-poly-deg50":    dict(sb=2.0489,  sr=1.0,       dir="min", sentinel=None),
    "bench-tsp-100":            dict(sb=50835,   sr=7517,      dir="min", sentinel=0),
    "bench-lj-n41":             dict(sb=-136.0,  sr=-198.0609, dir="min", sentinel=1e30),
    "bench-erdos-overlap":      dict(sb=0.2939,  sr=0.2321,    dir="min", sentinel=None),
    "bench-molecule-qed":       dict(sb=0.60,    sr=1.0,       dir="max", sentinel=None),
    # ── Supplemental tasks (PROVISIONAL anchors, 2026-04-08) ──────────────────
    "bench-graph-coloring-g80": dict(sb=16,      sr=10,        dir="min", sentinel=None),  # inf → Q clips to 0
    "bench-tsp-50":             dict(sb=25400,    sr=5500,      dir="min", sentinel=0),
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
    "bench-graph-coloring-g80": "GColor",
    "bench-tsp-50":             "TSP-50",
}

# MEG denominator baselines: max(E[Q(SR)], E[Q(BoN)], E[Q(VGS)]) — filled after computing Q
DENOM_PROTOCOLS = {"self-refine", "best-of-n", "vgs"}

# DeepSeek-specific denominator: used instead of DENOM_PROTOCOLS for all DeepSeek
# and OSS protocols once the control sweep has completed.  Without these controls the
# OSS MEG conflates model capability with protocol structure.
DENOM_PROTOCOLS_DEEPSEEK = {"self-refine-deepseek", "best-of-n-deepseek", "vgs-deepseek"}
DEEPSEEK_DENOM_PROTOS    = {"magicore-deepseek", "debate-deepseek", "hpe-deepseek",
                            "self-refine-deepseek", "best-of-n-deepseek", "vgs-deepseek",
                            "magicore-mixed-oss", "debate-mixed-oss", "hpe-mixed-oss"}

# GPT-4o-specific denominator: GPT-4o protocols get MEG against GPT-4o controls
DENOM_PROTOCOLS_GPT4O = {"self-refine-gpt4o", "best-of-n-gpt4o", "vgs-gpt4o"}
GPT4O_PROTOS = {"single-shot-gpt4o", "self-refine-gpt4o", "best-of-n-gpt4o", "vgs-gpt4o",
                "homo-chain-gpt4o", "magicore-gpt4o", "debate-gpt4o", "hpe-gpt4o"}

# Gemini-specific denominator: Gemini protocols get MEG against Gemini controls
DENOM_PROTOCOLS_GEMINI = {"self-refine-gemini", "best-of-n-gemini", "vgs-gemini"}
GEMINI_PROTOS = {"single-shot-gemini", "self-refine-gemini", "best-of-n-gemini", "vgs-gemini",
                 "homo-chain-gemini", "magicore-gemini", "debate-gemini", "hpe-gemini"}

# Core 10 protocols for dev/hidden rank analysis (paper claims use this subset)
CORE_PROTOCOLS = {
    "single-shot", "self-refine", "best-of-n", "vgs",
    "homo-chain", "cross-chain", "magicore", "debate", "moa", "hpe",
}

def is_sentinel(s, a):
    if s is None: return True
    if a["sentinel"] is None: return False
    if a["sentinel"] == 0:    return s == 0
    return s >= a["sentinel"] * 0.5   # catches 1e30

def compute_q(s, a):
    if is_sentinel(s, a): return None   # invalid — exclude from mean
    sb, sr = a["sb"], a["sr"]
    if a["dir"] == "max":
        q = (s - sb) / (sr - sb)
    else:
        q = (sb - s) / (sb - sr)
    return max(0.0, min(1.0, q))

# ── Load all results ───────────────────────────────────────────────────────────
# cell[task][proto] = list of Q scores — invalid runs mapped to 0.0 (penalise failures)
# Only seeds in FROZEN_SEEDS are loaded; seed=42 extras are skipped
cell_dev    = defaultdict(lambda: defaultdict(list))   # cell_dev[task][proto] = [q, ...]
cell_hidden = defaultdict(lambda: defaultdict(list))
raw_cell_dev    = defaultdict(lambda: defaultdict(list))   # raw scores (not Q) for dev/hidden rank analysis
raw_cell_hidden = defaultdict(lambda: defaultdict(list))
skipped_seeds = []   # log files skipped due to out-of-manifest seed

files = glob.glob(f"{RESULTS_DIR}/bench-*_*.json")
for f in files:
    name  = os.path.basename(f).replace(".json","")
    # parse task + proto robustly
    rest  = name[len("bench-"):]                       # e.g. maxcut-g200_vgs_s3
    # find last _sN suffix
    parts = rest.rsplit("_s", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        continue

    seed = int(parts[1])
    if seed not in FROZEN_SEEDS:
        skipped_seeds.append((name, seed))
        continue   # ← enforce frozen manifest: skip seed=42 extras etc.

    body = "bench-" + parts[0]    # bench-maxcut-g200_vgs
    # split into task + proto: task is the key that matches ANCHORS
    task = proto = None
    for tid in ANCHORS:
        if body.startswith(tid + "_"):
            task  = tid
            proto = body[len(tid)+1:]
            break
    if task is None:
        continue

    # Enforce per-protocol seed cap: MoA arms use 10, all others capped at 5
    if proto not in MOA_PROTOCOLS and seed > MAX_SEED_DEFAULT:
        continue

    d  = json.load(open(f))
    a  = ANCHORS[task]

    dev_s    = (d.get("bestArtifact") or {}).get("devScore")
    hidden_s = d.get("hiddenScore")

    dq = compute_q(dev_s,    a)
    hq = compute_q(hidden_s, a)

    # Use 0 for invalid in mean (not None) so protocol with 1/5 invalids is penalised
    cell_dev[task][proto].append(0.0 if dq    is None else dq)
    cell_hidden[task][proto].append(0.0 if hq is None else hq)
    # Raw scores for dev/hidden rank analysis (sentinels excluded)
    if dev_s is not None and not is_sentinel(dev_s, a):
        raw_cell_dev[task][proto].append(dev_s)
    if hidden_s is not None and not is_sentinel(hidden_s, a):
        raw_cell_hidden[task][proto].append(hidden_s)

if skipped_seeds:
    print(f"[manifest] Skipped {len(skipped_seeds)} out-of-manifest files (seeds not in {sorted(FROZEN_SEEDS)}):")
    for name, s in skipped_seeds:
        print(f"  seed={s}  {name}")

# ── Mean Q per (task, proto) ───────────────────────────────────────────────────
def mean(lst): return sum(lst)/len(lst) if lst else float("nan")

tasks    = list(ANCHORS.keys())
protos   = sorted({p for t in cell_hidden for p in cell_hidden[t]})

q_dev    = {t: {p: mean(cell_dev[t][p])    for p in protos} for t in tasks}
q_hidden = {t: {p: mean(cell_hidden[t][p]) for p in protos} for t in tasks}

# ── MEG denominator per task ───────────────────────────────────────────────────
denom = {}
for t in tasks:
    vals = [q_hidden[t].get(p, 0.0) for p in DENOM_PROTOCOLS if p in q_hidden[t]]
    denom[t] = max(vals) if vals else 0.0

# DeepSeek-specific denominator — auto-detected; falls back to Claude denom if not present.
_ds_controls_present = any(
    p in q_hidden.get(t, {})
    for t in tasks
    for p in DENOM_PROTOCOLS_DEEPSEEK
)
denom_deepseek = {}
for t in tasks:
    vals = [q_hidden[t][p] for p in DENOM_PROTOCOLS_DEEPSEEK if p in q_hidden[t] and not math.isnan(q_hidden[t][p])]
    denom_deepseek[t] = max(vals) if vals else denom[t]   # fall back to Claude denom

if _ds_controls_present:
    print(f"[denom] DeepSeek controls detected — using DeepSeek-specific MEG denominator for: {sorted(DEEPSEEK_DENOM_PROTOS)}")
else:
    print(f"[denom] DeepSeek controls NOT yet present — {sorted(DEEPSEEK_DENOM_PROTOS)} still use Claude denominator (run self-refine/best-of-n/vgs-deepseek to fix)")

# GPT-4o-specific denominator
_gpt4o_controls_present = any(
    p in q_hidden.get(t, {})
    for t in tasks
    for p in DENOM_PROTOCOLS_GPT4O
)
denom_gpt4o = {}
for t in tasks:
    vals = [q_hidden[t][p] for p in DENOM_PROTOCOLS_GPT4O if p in q_hidden[t] and not math.isnan(q_hidden[t][p])]
    denom_gpt4o[t] = max(vals) if vals else denom[t]

if _gpt4o_controls_present:
    print(f"[denom] GPT-4o controls detected — using GPT-4o-specific MEG denominator for: {sorted(GPT4O_PROTOS)}")
else:
    print(f"[denom] GPT-4o controls NOT yet present — GPT-4o protocols still use Claude denominator")

# Gemini-specific denominator
_gemini_controls_present = any(
    p in q_hidden.get(t, {})
    for t in tasks
    for p in DENOM_PROTOCOLS_GEMINI
)
denom_gemini = {}
for t in tasks:
    vals = [q_hidden[t][p] for p in DENOM_PROTOCOLS_GEMINI if p in q_hidden[t] and not math.isnan(q_hidden[t][p])]
    denom_gemini[t] = max(vals) if vals else denom[t]

if _gemini_controls_present:
    print(f"[denom] Gemini controls detected — using Gemini-specific MEG denominator for: {sorted(GEMINI_PROTOS)}")
else:
    print(f"[denom] Gemini controls NOT yet present — Gemini protocols still use Claude denominator")

def get_denom(task, proto):
    """Return the correct MEG denominator for (task, proto).

    Each backbone family gets its own denominator once controls are present.
    Falls back to the Claude denominator during incomplete sweeps.
    """
    if proto in DEEPSEEK_DENOM_PROTOS and _ds_controls_present:
        return denom_deepseek[task]
    if proto in GPT4O_PROTOS and _gpt4o_controls_present:
        return denom_gpt4o[task]
    if proto in GEMINI_PROTOS and _gemini_controls_present:
        return denom_gemini[task]
    return denom[task]

# ── Bootstrap CI for MEG ───────────────────────────────────────────────────────
import random
random.seed(42)
N_BOOT = 2000

def bootstrap_meg_ci(task, proto, n_boot=N_BOOT):
    raw = cell_hidden[task].get(proto, [])
    if not raw: return float("nan"), float("nan")
    b_dn = get_denom(task, proto)
    samples = []
    for _ in range(n_boot):
        resamp = [random.choice(raw) for _ in raw]
        samples.append(mean(resamp) - b_dn)
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot)]
    return lo, hi

# ── Write MEG table ────────────────────────────────────────────────────────────
task_order = list(ANCHORS.keys())
meg_rows = []
incomplete_cells = []   # (proto, task, n) for cells with n < MIN_SEEDS

for proto in protos:
    per_task = {}
    for t in task_order:
        n = len(cell_hidden[t].get(proto, []))
        if n > 0 and n < MIN_SEEDS:
            incomplete_cells.append((proto, t, n))
        qh  = q_hidden[t].get(proto, float("nan"))
        meg = qh - get_denom(t, proto) if not math.isnan(qh) else float("nan")
        per_task[t] = meg
    # AggMEG: exclude tasks in EXCLUDE_FROM_AGG (stale/re-run needed)
    agg_tasks = [t for t in task_order if t not in EXCLUDE_FROM_AGG]
    valid = [per_task[t] for t in agg_tasks if not math.isnan(per_task.get(t, float("nan")))]
    agg   = mean(valid) if valid else float("nan")
    meg_rows.append((proto, per_task, agg))

if incomplete_cells:
    print(f"\n[incomplete cells] {len(incomplete_cells)} cells have fewer than {MIN_SEEDS} seeds:")
    for proto, t, n in incomplete_cells:
        flag = " ⚠️  EXCLUDED_FROM_AGG" if t in EXCLUDE_FROM_AGG else ""
        print(f"  {proto:<30} {TASK_LABELS[t]:<12} n={n}{flag}")

meg_rows.sort(key=lambda r: -(r[2] if not math.isnan(r[2]) else -99))

with open(f"{OUT_DIR}/meg_full.tsv", "w") as fh:
    n_seed_cols = [f"n_{TASK_LABELS[t]}" for t in task_order]
    header = ["protocol"] + [TASK_LABELS[t] for t in task_order] + ["AggMEG", "CI_lo", "CI_hi"] + n_seed_cols + ["agg_excludes"]
    fh.write("\t".join(header) + "\n")
    for proto, per_task, agg in meg_rows:
        # aggregate CI — only over tasks not in EXCLUDE_FROM_AGG
        agg_tasks = [t for t in task_order if t not in EXCLUDE_FROM_AGG]
        all_boot = []
        for _ in range(N_BOOT):
            task_megs = []
            for t in agg_tasks:
                raw = cell_hidden[t].get(proto, [])
                if not raw: continue
                task_megs.append(mean([random.choice(raw) for _ in raw]) - get_denom(t, proto))
            all_boot.append(mean(task_megs) if task_megs else float("nan"))
        all_boot = [x for x in all_boot if not math.isnan(x)]
        all_boot.sort()
        ci_lo = all_boot[int(0.025*len(all_boot))] if all_boot else float("nan")
        ci_hi = all_boot[int(0.975*len(all_boot))] if all_boot else float("nan")
        # n_seeds per task for this protocol
        ns = [str(len(cell_hidden[t].get(proto, []))) for t in task_order]
        # flag cells in TSV: mark incomplete cells with *
        meg_cells = []
        for t in task_order:
            v = per_task[t]
            n = len(cell_hidden[t].get(proto, []))
            flag = "*" if (n > 0 and n < MIN_SEEDS) else ""
            meg_cells.append(f"{v:+.4f}{flag}" if not math.isnan(v) else "nan")
        excludes_str = ",".join(TASK_LABELS[t] for t in sorted(EXCLUDE_FROM_AGG)) if EXCLUDE_FROM_AGG else "none"
        row = [proto] + meg_cells + [f"{agg:+.4f}", f"{ci_lo:+.4f}", f"{ci_hi:+.4f}"] + ns + [excludes_str]
        fh.write("\t".join(row) + "\n")

print("Wrote", f"{OUT_DIR}/meg_full.tsv")

# ── Leave-one-task-out (jackknife) robustness ─────────────────────────────────
# For each protocol, drop one task at a time from the 6 non-excluded tasks and
# recompute AggMEG over the remaining 5.  Key question: does MoA stay positive
# when DiffBases is dropped?
agg_tasks_full = [t for t in task_order if t not in EXCLUDE_FROM_AGG]

jk_rows = []   # (proto, {dropped_task: agg_meg_without_it}, min, max, sign_stable)
for proto in protos:
    dropped = {}
    for drop_t in agg_tasks_full:
        remain = [t for t in agg_tasks_full if t != drop_t]
        vals = []
        for t in remain:
            qh = q_hidden[t].get(proto, float("nan"))
            if not math.isnan(qh):
                vals.append(qh - get_denom(t, proto))
        dropped[drop_t] = mean(vals) if vals else float("nan")
    valid_vals = [v for v in dropped.values() if not math.isnan(v)]
    jk_min = min(valid_vals) if valid_vals else float("nan")
    jk_max = max(valid_vals) if valid_vals else float("nan")
    sign_stable = all(v >= 0 for v in valid_vals) or all(v <= 0 for v in valid_vals) if valid_vals else False
    jk_rows.append((proto, dropped, jk_min, jk_max, sign_stable))

# Sort by original AggMEG (same order as meg_rows)
proto_agg = {proto: agg for proto, _, agg in meg_rows}
jk_rows.sort(key=lambda r: -(proto_agg.get(r[0], -99)))

with open(f"{OUT_DIR}/jackknife_meg.tsv", "w") as fh:
    header = ["protocol"] + [f"drop_{TASK_LABELS[t]}" for t in agg_tasks_full] + ["jk_min", "jk_max", "sign_stable", "orig_AggMEG"]
    fh.write("\t".join(header) + "\n")
    for proto, dropped, jk_min, jk_max, sign_stable in jk_rows:
        cells = [f"{dropped[t]:+.4f}" if not math.isnan(dropped[t]) else "nan" for t in agg_tasks_full]
        orig = proto_agg.get(proto, float("nan"))
        row = [proto] + cells + [f"{jk_min:+.4f}", f"{jk_max:+.4f}", str(sign_stable), f"{orig:+.4f}"]
        fh.write("\t".join(row) + "\n")

print("Wrote", f"{OUT_DIR}/jackknife_meg.tsv")

# Print jackknife summary
print()
print(f"=== LEAVE-ONE-TASK-OUT JACKKNIFE (6 tasks, drop 1 at a time) ===")
print(f"{'Protocol':<26} " + " ".join(f"{'−'+TASK_LABELS[t][:5]:>8}" for t in agg_tasks_full) + f"  {'jk_min':>8} {'jk_max':>8}  sign_stable")
print("-" * 105)
for proto, dropped, jk_min, jk_max, sign_stable in jk_rows:
    vals = []
    for t in agg_tasks_full:
        v = dropped[t]
        vals.append(f"{v:+.4f}" if not math.isnan(v) else "   nan")
    marker = "✓" if sign_stable else "✗ FLIPS"
    print(f"{proto:<26} " + "  ".join(vals) + f"  {jk_min:>+8.4f} {jk_max:>+8.4f}  {marker}")

print()

# ── Dev/hidden rank analysis: Spearman ρ dev-rank vs hidden-rank (Q-based, sentinel-aware) ──
def spearman(xs, ys):
    n = len(xs)
    if n < 3: return float("nan"), float("nan")
    def rank(lst):
        s = sorted(range(n), key=lambda i: lst[i])
        r = [0]*n
        for rank_val, idx in enumerate(s):
            r[idx] = rank_val + 1
        return r
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    rho = 1 - 6*d2/(n*(n*n-1))
    # approx p-value via t-distribution
    if abs(rho) >= 1.0: return rho, 0.0
    t_stat = rho * math.sqrt((n-2)/(1-rho**2))
    # two-tailed p ~ 2*(1 - CDF(|t|, df=n-2)); rough approx
    import math as m
    p = 2.0 / (1 + m.exp(0.717*abs(t_stat) - 0.416))   # logistic approximation
    return rho, p

def compute_rank_correlation(task_order, protos_filter, cell_dev, cell_hidden):
    """Compute dev/hidden Spearman rho for each task on RAW scores (not Q).

    Raw scores are the correct basis for within-task rank correlation because
    Q normalization clips floor tasks (all Q=0) and destroys rank information.
    The question "does dev ranking match hidden ranking?" is a within-task rank
    question that should be answered on the actual evaluator outputs.
    """
    rows = []
    for t in task_order:
        a = ANCHORS[t]
        # Compute mean raw dev and hidden scores per protocol
        raw_dev    = {}
        raw_hidden = {}
        for p in protos_filter:
            dvals = cell_dev[t].get(p, [])
            hvals = cell_hidden[t].get(p, [])
            if not dvals or not hvals:
                continue
            # Use raw scores from cells (before Q normalization).
            # cell_dev/cell_hidden store Q values, so we need the raw values.
            # Re-read from results files would be expensive, so instead use
            # the raw-score means from a separate accumulator.
            raw_dev[p]    = raw_dev_scores[t].get(p, float("nan"))
            raw_hidden[p] = raw_hidden_scores[t].get(p, float("nan"))

        common = [p for p in raw_dev if p in raw_hidden
                  and not math.isnan(raw_dev[p]) and not math.isnan(raw_hidden[p])]
        if len(common) < 3:
            continue

        # Sort direction: for all tasks, higher raw score is "better" for maximize,
        # lower is "better" for minimize.  We rank so rank 1 = best.
        rev = (a["dir"] == "max")
        dev_sorted    = sorted(common, key=lambda p: raw_dev[p],    reverse=rev)
        hidden_sorted = sorted(common, key=lambda p: raw_hidden[p], reverse=rev)

        dev_rank    = {p: i+1 for i, p in enumerate(dev_sorted)}
        hidden_rank = {p: i+1 for i, p in enumerate(hidden_sorted)}

        xs = [dev_rank[p]    for p in common]
        ys = [hidden_rank[p] for p in common]
        rho, pval = spearman(xs, ys)

        top1_dev    = dev_sorted[0]
        top1_hidden = hidden_sorted[0]
        inverted    = top1_dev != top1_hidden
        top3_dev    = set(dev_sorted[:3])
        top3_hidden = set(hidden_sorted[:3])
        top3_olap   = len(top3_dev & top3_hidden)

        rows.append((t, len(common), rho, pval, top1_dev, top1_hidden, inverted, top3_olap))
    return rows

def write_rank_correlation(path, rows):
    with open(path, "w") as fh:
        fh.write("task\tn_protocols\trho\tp_approx\ttop1_dev\ttop1_hidden\ttop1_inverted\ttop3_overlap\n")
        for t, n, rho, pval, t1d, t1h, inv, olap in rows:
            fh.write(f"{t}\t{n}\t{rho:.4f}\t{pval:.4f}\t{t1d}\t{t1h}\t{inv}\t{olap}/3\n")

# ── Mean raw scores per (task, proto) for dev/hidden rank analysis ─────────────
raw_dev_scores    = {t: {p: mean(raw_cell_dev[t][p])    for p in protos} for t in tasks}
raw_hidden_scores = {t: {p: mean(raw_cell_hidden[t][p]) for p in protos} for t in tasks}

# Compute both: all-protocol and core-only
rank_all  = compute_rank_correlation(task_order, set(protos), cell_dev, cell_hidden)
rank_core = compute_rank_correlation(task_order, CORE_PROTOCOLS, cell_dev, cell_hidden)

write_rank_correlation(f"{OUT_DIR}/rank_all.tsv", rank_all)
write_rank_correlation(f"{OUT_DIR}/rank_core.tsv", rank_core)

print("Wrote", f"{OUT_DIR}/rank_all.tsv")
print("Wrote", f"{OUT_DIR}/rank_core.tsv")

# ── Brain-diversity delta ──────────────────────────────────────────────────────
# Full 2×2: (frontier vs OSS) × (homo vs mixed)
# homo_oss comes from *-deepseek protocols (DeepSeek V3 × 3 roles)
families = [
    ("magicore", "magicore-mixed", "magicore-mixed-oss", "magicore-deepseek"),
    ("debate",   "debate-mixed",   "debate-mixed-oss",   "debate-deepseek"),
    ("hpe",      "hpe-mixed",      "hpe-mixed-oss",      "hpe-deepseek"),
]

def agg_meg(proto):
    megs = []
    for t in task_order:
        if t in EXCLUDE_FROM_AGG:
            continue
        qh = q_hidden[t].get(proto, float("nan"))
        if not math.isnan(qh):
            megs.append(qh - get_denom(t, proto))
    return mean(megs) if megs else float("nan")

def per_task_meg(proto):
    return {TASK_LABELS[t]: f"{q_hidden[t].get(proto, float('nan')) - get_denom(t, proto):+.3f}" for t in task_order}

with open(f"{OUT_DIR}/brain_diversity.tsv", "w") as fh:
    # Columns:
    #   homo_MEG      = frontier-homo (Claude×3)
    #   mixed_MEG     = frontier-mixed (Claude+GPT4o+Gemini)
    #   homo_oss_MEG  = OSS-homo (DeepSeek×3)  ← new
    #   mixed_oss_MEG = OSS-mixed (DeepSeek+Llama+Qwen)
    #   delta_mix     = mixed_MEG - homo_MEG       (diversity gain within frontier)
    #   delta_cap     = homo_oss_MEG - homo_MEG    (capability gap, no diversity)
    #   delta_mix_oss = mixed_oss_MEG - homo_oss_MEG (diversity gain within OSS)
    fh.write("family\thomo_MEG\tmixed_MEG\thomo_oss_MEG\tmixed_oss_MEG\tdelta_mix\tdelta_cap\tdelta_mix_oss\tn_homo\tn_mix\tn_homo_oss\tn_mix_oss\n")
    for homo, mix, oss, homo_oss in families:
        h_meg   = agg_meg(homo)
        m_meg   = agg_meg(mix)
        o_meg   = agg_meg(oss)
        ho_meg  = agg_meg(homo_oss)
        dm      = m_meg  - h_meg  if not (math.isnan(m_meg)  or math.isnan(h_meg))  else float("nan")
        dc      = ho_meg - h_meg  if not (math.isnan(ho_meg) or math.isnan(h_meg))  else float("nan")
        dmo     = o_meg  - ho_meg if not (math.isnan(o_meg)  or math.isnan(ho_meg)) else float("nan")
        n_h     = len(cell_hidden["bench-erdos-overlap"].get(homo,     []))
        n_m     = len(cell_hidden["bench-erdos-overlap"].get(mix,      []))
        n_ho    = len(cell_hidden["bench-erdos-overlap"].get(homo_oss, []))
        n_o     = len(cell_hidden["bench-erdos-overlap"].get(oss,      []))
        fh.write(f"{homo}\t{h_meg:+.4f}\t{m_meg:+.4f}\t{ho_meg:+.4f}\t{o_meg:+.4f}\t{dm:+.4f}\t{dc:+.4f}\t{dmo:+.4f}\t{n_h}\t{n_m}\t{n_ho}\t{n_o}\n")

print("Wrote", f"{OUT_DIR}/brain_diversity.tsv")

# ── MEG denominator sensitivity (Q1) ─────────────────────────────────────────
# Report MEG under three denominator variants:
#   1. max(SR, BoN, VGS) — current default
#   2. VGS-only
#   3. BoN-only
# If MoA's sign flips under a different denominator, disclose in paper.
denom_variants = {
    "max(SR,BoN,VGS)": DENOM_PROTOCOLS,      # current
    "VGS-only":         {"vgs"},
    "BoN-only":         {"best-of-n"},
    "SR-only":          {"self-refine"},
}

sens_rows = []  # (proto, {variant_name: agg_meg})
for proto in protos:
    row = {}
    for vname, vprotos in denom_variants.items():
        d_var = {}
        for t in tasks:
            vals = [q_hidden[t].get(p, 0.0) for p in vprotos if p in q_hidden[t]]
            d_var[t] = max(vals) if vals else 0.0
        agg_tasks = [t for t in task_order if t not in EXCLUDE_FROM_AGG]
        megs = []
        for t in agg_tasks:
            qh = q_hidden[t].get(proto, float("nan"))
            if not math.isnan(qh):
                megs.append(qh - d_var[t])
        row[vname] = mean(megs) if megs else float("nan")
    sens_rows.append((proto, row))

sens_rows.sort(key=lambda r: -(r[1].get("max(SR,BoN,VGS)", -99)))

variant_names = list(denom_variants.keys())
with open(f"{OUT_DIR}/meg_denom_sensitivity.tsv", "w") as fh:
    fh.write("\t".join(["protocol"] + variant_names + ["sign_stable_across_denoms"]) + "\n")
    for proto, row in sens_rows:
        vals = [row[v] for v in variant_names]
        valid = [v for v in vals if not math.isnan(v)]
        stable = all(v >= 0 for v in valid) or all(v <= 0 for v in valid) if valid else False
        cells = [f"{row[v]:+.4f}" if not math.isnan(row[v]) else "nan" for v in variant_names]
        fh.write("\t".join([proto] + cells + [str(stable)]) + "\n")

print("Wrote", f"{OUT_DIR}/meg_denom_sensitivity.tsv")

# Print denominator sensitivity summary
print()
print("=== MEG DENOMINATOR SENSITIVITY (Q1 — does MoA sign survive?) ===")
print(f"{'Protocol':<26} " + "  ".join(f"{v:>16}" for v in variant_names) + "  stable?")
print("-" * 105)
for proto, row in sens_rows:
    vals = [row[v] for v in variant_names]
    valid = [v for v in vals if not math.isnan(v)]
    stable = all(v >= 0 for v in valid) or all(v <= 0 for v in valid) if valid else False
    cells = [f"{row[v]:>+16.4f}" if not math.isnan(row[v]) else "             nan" for v in variant_names]
    marker = "✓" if stable else "✗ FLIPS"
    print(f"{proto:<26} " + "  ".join(cells) + f"  {marker}")

print()

# ── Print summary ──────────────────────────────────────────────────────────────
print()
excl_note = ", ".join(TASK_LABELS[t] for t in EXCLUDE_FROM_AGG)
print(f"=== MEG FULL TABLE ({len(protos)} protocols, {len(tasks)} tasks | AggMEG excludes: {excl_note}) ===")
hdr_labels = " ".join(f"{TASK_LABELS[t]:>7}" for t in task_order)
print(f"{'Protocol':<26} {hdr_labels}  {'AggMEG':>8}  (* = n<{MIN_SEEDS})")
print("-"*(26 + 8*len(tasks) + 12))
for proto, per_task, agg in meg_rows:
    vals = []
    for t in task_order:
        v = per_task[t]
        n = len(cell_hidden[t].get(proto, []))
        flag = "*" if (n > 0 and n < MIN_SEEDS) else " "
        vals.append(f"{v:+.3f}{flag}" if not math.isnan(v) else "  nan ")
    row_str = " ".join(vals)
    print(f"{proto:<26} {row_str}  {agg:>+8.3f}")

print()
print("=== DEV/HIDDEN RANK SPEARMAN — ALL PROTOCOLS ===")
for line in open(f"{OUT_DIR}/rank_all.tsv"):
    print(line.rstrip())

print()
print("=== DEV/HIDDEN RANK SPEARMAN — CORE 10 PROTOCOLS (paper claims) ===")
for line in open(f"{OUT_DIR}/rank_core.tsv"):
    print(line.rstrip())

print()
print("=== BRAIN DIVERSITY DELTA (note n_erdos for validity) ===")
for line in open(f"{OUT_DIR}/brain_diversity.tsv"):
    print(line.rstrip())

# ── BACKBONE COMPARISON — does MEG≤0 generalize across model families? ────────
# Map each backbone-specific protocol to its "canonical" name (e.g. magicore-gpt4o → magicore)
BACKBONE_SUFFIXES = {"gpt4o": GPT4O_PROTOS, "gemini": GEMINI_PROTOS}
BACKBONE_LABELS   = {"claude": "Claude", "gpt4o": "GPT-4o", "gemini": "Gemini"}

# Canonical protocol names (strip backbone suffix)
def canonical(proto):
    for suffix in ["-gpt4o", "-gemini", "-deepseek"]:
        if proto.endswith(suffix):
            return proto[:-len(suffix)]
    return proto

# Protocols that exist across all three backbones (controls + multi-agent)
BACKBONE_CANONICAL = ["self-refine", "best-of-n", "vgs", "homo-chain", "magicore", "debate", "hpe"]

def backbone_agg_meg(proto_id):
    """Compute AggMEG for a specific protocol ID using its backbone-appropriate denominator."""
    megs = []
    for t in task_order:
        if t in EXCLUDE_FROM_AGG:
            continue
        qh = q_hidden[t].get(proto_id, float("nan"))
        if not math.isnan(qh):
            megs.append(qh - get_denom(t, proto_id))
    return mean(megs) if megs else float("nan")

def backbone_per_task_meg(proto_id):
    """Return dict of {task_label: MEG} for a protocol."""
    result = {}
    for t in task_order:
        qh = q_hidden[t].get(proto_id, float("nan"))
        if not math.isnan(qh):
            result[TASK_LABELS[t]] = qh - get_denom(t, proto_id)
        else:
            result[TASK_LABELS[t]] = float("nan")
    return result

backbone_comparison = []  # (canonical_name, {backbone: agg_meg}, {backbone: {task: meg}})
for canon in BACKBONE_CANONICAL:
    agg = {}
    per_task = {}
    for bb, bb_label in [("claude", "Claude"), ("gpt4o", "GPT-4o"), ("gemini", "Gemini")]:
        pid = canon if bb == "claude" else f"{canon}-{bb}"
        agg[bb_label] = backbone_agg_meg(pid)
        per_task[bb_label] = backbone_per_task_meg(pid)
    backbone_comparison.append((canon, agg, per_task))

# Write backbone comparison TSV
with open(f"{OUT_DIR}/backbone_comparison.tsv", "w") as fh:
    header = ["protocol", "Claude_AggMEG", "GPT4o_AggMEG", "Gemini_AggMEG", "all_negative", "rank_agreement"]
    for t in task_order:
        header.extend([f"Claude_{TASK_LABELS[t]}", f"GPT4o_{TASK_LABELS[t]}", f"Gemini_{TASK_LABELS[t]}"])
    fh.write("\t".join(header) + "\n")
    for canon, agg, per_task in backbone_comparison:
        vals = [agg.get(b, float("nan")) for b in ["Claude", "GPT-4o", "Gemini"]]
        valid = [v for v in vals if not math.isnan(v)]
        all_neg = all(v <= 0 for v in valid) if len(valid) >= 2 else False
        # Rank agreement: do all backbones agree on sign?
        signs = [("+" if v > 0 else "-" if v < 0 else "0") for v in valid]
        agree = len(set(signs)) <= 1 if signs else False
        row = [canon]
        row.extend([f"{v:+.4f}" if not math.isnan(v) else "nan" for v in vals])
        row.append(str(all_neg))
        row.append(str(agree))
        for t in task_order:
            for bb in ["Claude", "GPT-4o", "Gemini"]:
                v = per_task.get(bb, {}).get(TASK_LABELS[t], float("nan"))
                row.append(f"{v:+.4f}" if not math.isnan(v) else "nan")
        fh.write("\t".join(row) + "\n")

print("Wrote", f"{OUT_DIR}/backbone_comparison.tsv")

# Print backbone comparison
print()
print("=== BACKBONE COMPARISON — Does MEG≤0 generalize across model families? ===")
print(f"{'Protocol':<16} {'Claude':>10} {'GPT-4o':>10} {'Gemini':>10}  all≤0?  sign_agree?")
print("-" * 72)
for canon, agg, _ in backbone_comparison:
    vals = [agg.get(b, float("nan")) for b in ["Claude", "GPT-4o", "Gemini"]]
    valid = [v for v in vals if not math.isnan(v)]
    all_neg = all(v <= 0 for v in valid) if len(valid) >= 2 else False
    signs = [("+" if v > 0 else "-" if v < 0 else "0") for v in valid]
    agree = len(set(signs)) <= 1 if signs else False
    cells = [f"{v:>+10.4f}" if not math.isnan(v) else "       nan" for v in vals]
    neg_mark = "✓" if all_neg else ("✗" if len(valid) >= 2 else "?")
    agr_mark = "✓" if agree else ("✗" if len(valid) >= 2 else "?")
    print(f"{canon:<16} " + " ".join(cells) + f"  {neg_mark:>6}  {agr_mark:>10}")

# Backbone-invariance: Spearman rank correlation across backbones per task
print()
print("=== BACKBONE RANK INVARIANCE (per-task Spearman) ===")
for t in task_order:
    tl = TASK_LABELS[t]
    # Collect MEG for each backbone on this task
    bb_megs = {}
    for bb in ["Claude", "GPT-4o", "Gemini"]:
        megs = {}
        for canon in BACKBONE_CANONICAL:
            pid = canon if bb == "Claude" else f"{canon}-{bb.lower().replace('-','')}"
            # Fix: GPT-4o → gpt4o, Gemini → gemini
            if bb == "GPT-4o":
                pid = canon if bb == "Claude" else f"{canon}-gpt4o"
            elif bb == "Gemini":
                pid = canon if bb == "Claude" else f"{canon}-gemini"
            qh = q_hidden[t].get(pid, float("nan"))
            if not math.isnan(qh):
                megs[canon] = qh - get_denom(t, pid)
        bb_megs[bb] = megs

    # Pairwise Spearman between backbones
    for bb1, bb2 in [("Claude", "GPT-4o"), ("Claude", "Gemini"), ("GPT-4o", "Gemini")]:
        common = [c for c in BACKBONE_CANONICAL if c in bb_megs[bb1] and c in bb_megs[bb2]]
        if len(common) >= 3:
            xs = [bb_megs[bb1][c] for c in common]
            ys = [bb_megs[bb2][c] for c in common]
            rho, pval = spearman(xs, ys)
            print(f"  {tl:<10} {bb1:>8} vs {bb2:<8}: rho={rho:+.3f} (n={len(common)})")

print()
