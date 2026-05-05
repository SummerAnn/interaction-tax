#!/usr/bin/env python3
"""
MATCHED 2x2 Diversity x Synthesis ablation analysis.

Uses best-of-3 (N=3, same model) instead of best-of-n (N=8) to eliminate
the team-size confound. All four cells now use N=3 proposals.

                    Selection         Synthesis
  Same backbone:    best-of-3 (NEW)   moa-same-model
  Diverse backbone: moa-nosynth       moa

Factors coded as binary indicators:
  diversity  = 1 if diverse (moa, moa-nosynth), 0 if same (moa-same-model, best-of-3)
  synthesis  = 1 if synthesis (moa, moa-same-model), 0 if selection (moa-nosynth, best-of-3)

Model:  Q ~ diversity + synthesis + diversity:synthesis + task_dummies
        (task dummies absorb per-task intercept variation, analogous to random intercept)

Bootstrap (10,000 resamples, resampling at observation level) used for CIs.
"""

import json
import os
import glob
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = "/Users/summerann/Desktop/neurips-experiments/results/full-v2"

TASKS = {
    # taskId: (direction, s_bad, s_ref)
    "bench-difference-bases": ("minimize", 14.0, 1.0),
    "bench-erdos-overlap":    ("minimize", 0.2939, 0.2321),
    "bench-molecule-qed":     ("maximize", 0.60, 1.0),
}

PROTOCOLS = {
    # protocolId: (diversity, synthesis)
    "moa":            (1, 1),  # diverse + synthesis
    "moa-same-model": (0, 1),  # same   + synthesis
    "moa-nosynth":    (1, 0),  # diverse + selection
    "best-of-3":      (0, 0),  # same   + selection (MATCHED N=3)
}

SEED_RANGES = {
    "moa":            range(1, 11),
    "moa-same-model": range(1, 11),
    "moa-nosynth":    range(1, 11),
    "best-of-3":      range(1, 11),  # 10 seeds, matching MoA arms
}

N_BOOTSTRAP = 10_000
RNG_SEED = 42


# ── Q-normalization ───────────────────────────────────────────────────────

def compute_q(score, direction, s_bad, s_ref):
    """Normalize raw score to Q in [0, 1]."""
    if direction == "minimize":
        q = (s_bad - score) / (s_bad - s_ref)
    else:  # maximize
        q = (score - s_bad) / (s_ref - s_bad)
    return max(0.0, min(1.0, q))


# ── Load data ─────────────────────────────────────────────────────────────

rows = []  # each row: (Q, diversity, synthesis, task_index)
task_names = list(TASKS.keys())
missing = []

for task_idx, (task_id, (direction, s_bad, s_ref)) in enumerate(TASKS.items()):
    for proto_id, (div_flag, synth_flag) in PROTOCOLS.items():
        for seed in SEED_RANGES[proto_id]:
            fname = f"{task_id}_{proto_id}_s{seed}.json"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                missing.append(fname)
                continue
            with open(fpath) as f:
                data = json.load(f)
            score = data["hiddenScore"]
            q = compute_q(score, direction, s_bad, s_ref)
            rows.append((q, div_flag, synth_flag, task_idx))

print(f"Loaded {len(rows)} observations across {len(TASKS)} tasks, "
      f"{len(PROTOCOLS)} protocols")
if missing:
    print(f"  WARNING: {len(missing)} files missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
print()

# ── Build design matrix ──────────────────────────────────────────────────

data_arr = np.array(rows, dtype=float)
Q = data_arr[:, 0]
diversity = data_arr[:, 1]
synthesis = data_arr[:, 2]
interaction = diversity * synthesis
task_idx = data_arr[:, 3].astype(int)

n_tasks = len(task_names)
# Task dummies (drop first task as reference -> intercept absorbs it)
task_dummies = np.zeros((len(Q), n_tasks - 1))
for t in range(1, n_tasks):
    task_dummies[:, t - 1] = (task_idx == t).astype(float)

# X = [intercept, diversity, synthesis, interaction, task_dummy_1, task_dummy_2]
X = np.column_stack([
    np.ones(len(Q)),
    diversity,
    synthesis,
    interaction,
    task_dummies,
])

col_names = ["intercept", "diversity", "synthesis", "diversity:synthesis"] + \
            [f"task_{task_names[t]}" for t in range(1, n_tasks)]


# ── OLS fit ───────────────────────────────────────────────────────────────

def ols_fit(X, y):
    """Ordinary least squares: beta = (X'X)^-1 X'y"""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

beta_hat = ols_fit(X, Q)

# Residuals and R^2
y_hat = X @ beta_hat
resid = Q - y_hat
ss_res = np.sum(resid ** 2)
ss_tot = np.sum((Q - np.mean(Q)) ** 2)
r_squared = 1.0 - ss_res / ss_tot

print("=" * 65)
print("OLS POINT ESTIMATES  (Q ~ diversity * synthesis + task dummies)")
print("=" * 65)
for name, b in zip(col_names, beta_hat):
    print(f"  {name:30s}  {b:+.4f}")
print(f"\n  R^2 = {r_squared:.4f}")
print(f"  N   = {len(Q)}")
print()


# ── Bootstrap CIs ────────────────────────────────────────────────────────

rng = np.random.RandomState(RNG_SEED)
n = len(Q)
boot_betas = np.zeros((N_BOOTSTRAP, X.shape[1]))
boot_div_gt_synth = 0  # count: diversity coeff > synthesis coeff

for b in range(N_BOOTSTRAP):
    idx = rng.randint(0, n, size=n)
    boot_betas[b] = ols_fit(X[idx], Q[idx])
    if boot_betas[b][1] > boot_betas[b][2]:  # diversity > synthesis
        boot_div_gt_synth += 1

ci_lo = np.percentile(boot_betas, 2.5, axis=0)
ci_hi = np.percentile(boot_betas, 97.5, axis=0)

print("=" * 65)
print("BOOTSTRAP 95% CIs  (10,000 resamples, percentile method)")
print("=" * 65)
for i, name in enumerate(col_names):
    sig = "*" if (ci_lo[i] > 0 or ci_hi[i] < 0) else ""
    print(f"  {name:30s}  {beta_hat[i]:+.4f}  "
          f"[{ci_lo[i]:+.4f}, {ci_hi[i]:+.4f}]  {sig}")
print()
print("  (* = 95% CI excludes zero)")
print()


# ── Diversity > Synthesis test ────────────────────────────────────────────

p_div_gt_synth = boot_div_gt_synth / N_BOOTSTRAP
print("=" * 65)
print("COMPARISON: diversity vs. synthesis main effects")
print("=" * 65)
print(f"  diversity coeff  = {beta_hat[1]:+.4f}")
print(f"  synthesis coeff  = {beta_hat[2]:+.4f}")
print(f"  difference       = {beta_hat[1] - beta_hat[2]:+.4f}")

# Bootstrap CI on the difference
diff_boots = boot_betas[:, 1] - boot_betas[:, 2]
diff_lo = np.percentile(diff_boots, 2.5)
diff_hi = np.percentile(diff_boots, 97.5)
print(f"  95% CI of diff   = [{diff_lo:+.4f}, {diff_hi:+.4f}]")
print(f"  P(diversity > synthesis) = {p_div_gt_synth:.4f}  "
      f"({'significant at 0.05' if p_div_gt_synth > 0.975 or p_div_gt_synth < 0.025 else 'not significant at 0.05'})")
print()


# ── Cell means summary ───────────────────────────────────────────────────

print("=" * 65)
print("CELL MEANS  (Q, pooled across tasks)")
print("=" * 65)
labels = {
    (0, 0): "same + selection   (best-of-3)",
    (0, 1): "same + synthesis   (moa-same-model)",
    (1, 0): "diverse + selection (moa-nosynth)",
    (1, 1): "diverse + synthesis (moa)",
}
for (d, s), label in labels.items():
    mask = (diversity == d) & (synthesis == s)
    vals = Q[mask]
    print(f"  {label:42s}  mean={np.mean(vals):.4f}  "
          f"sd={np.std(vals, ddof=1):.4f}  n={len(vals)}")
print()


# ── Per-task cell means ──────────────────────────────────────────────────

print("=" * 65)
print("PER-TASK CELL MEANS")
print("=" * 65)
for t, tname in enumerate(task_names):
    tmask = (task_idx == t)
    print(f"\n  {tname}:")
    for (d, s), label in labels.items():
        mask = tmask & (diversity == d) & (synthesis == s)
        vals = Q[mask]
        if len(vals) > 0:
            print(f"    {label:42s}  mean={np.mean(vals):.4f}  "
                  f"sd={np.std(vals, ddof=1):.4f}  n={len(vals)}")
print()


# ── Interaction interpretation ────────────────────────────────────────────

print("=" * 65)
print("INTERPRETATION")
print("=" * 65)
div_eff = beta_hat[1]
synth_eff = beta_hat[2]
inter_eff = beta_hat[3]

print(f"  Main effect of diversity:  {div_eff:+.4f}")
if ci_lo[1] > 0:
    print(f"    -> Diverse backbones significantly INCREASE Q")
elif ci_hi[1] < 0:
    print(f"    -> Diverse backbones significantly DECREASE Q")
else:
    print(f"    -> Effect of diverse backbones is NOT significant")

print(f"  Main effect of synthesis:  {synth_eff:+.4f}")
if ci_lo[2] > 0:
    print(f"    -> Synthesis significantly INCREASES Q")
elif ci_hi[2] < 0:
    print(f"    -> Synthesis significantly DECREASES Q")
else:
    print(f"    -> Effect of synthesis is NOT significant")

print(f"  Interaction:               {inter_eff:+.4f}")
if ci_lo[3] > 0 or ci_hi[3] < 0:
    print(f"    -> Significant interaction: the two factors are NOT additive")
else:
    print(f"    -> No significant interaction: effects are approximately additive")

if p_div_gt_synth > 0.975:
    print(f"\n  Diversity coefficient is significantly LARGER than synthesis coefficient")
elif p_div_gt_synth < 0.025:
    print(f"\n  Synthesis coefficient is significantly LARGER than diversity coefficient")
else:
    print(f"\n  No significant difference between diversity and synthesis coefficients")
