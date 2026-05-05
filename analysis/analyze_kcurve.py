#!/usr/bin/env python3
"""
analyze_kcurve.py — Plot dev vs hidden score as a function of K (eval call cap)

Reads results/kcurve-maxcut/bench-maxcut-g200_vgs-k*_s*.json
Produces a table and optional matplotlib figure.

Usage:
  python3 analyze_kcurve.py
  python3 analyze_kcurve.py --plot  # save figure to figures/kcurve_maxcut.pdf
"""

import json, glob, sys, os
from collections import defaultdict
import statistics

RESULTS_DIR = "results/kcurve-maxcut"
PLOT = "--plot" in sys.argv

# ── Load results ─────────────────────────────────────────────────────────────

# Search in per-K subdirectories (results/kcurve-maxcut/k*/bench-*.json)
# Protocol ID is just 'vgs'; K is encoded in the directory name
files = glob.glob(os.path.join(RESULTS_DIR, "k*", "bench-maxcut-g200_vgs_s*.json"))
if not files:
    print(f"No result files found in {RESULTS_DIR}/")
    print("Run the K-curve experiment first: npx tsx src/runner/run-kcurve.ts")
    sys.exit(1)

# Parse K from directory name (e.g., ".../k10/bench-..." → 10)
data = defaultdict(lambda: {"dev": [], "hidden": [], "eval_calls": []})

for f in sorted(files):
    with open(f) as fh:
        r = json.load(fh)
    # Extract K from parent directory name (k1, k3, k5, ...)
    parent_dir = os.path.basename(os.path.dirname(f))
    k_str = parent_dir.replace("k", "")
    try:
        k = int(k_str)
    except ValueError:
        continue

    dev = r.get("bestArtifact", {}).get("devScore")
    hidden = r.get("hiddenScore")
    n_calls = len(r.get("budgetTrace", {}).get("evalCalls", []))

    if dev is not None:
        data[k]["dev"].append(dev)
    if hidden is not None:
        data[k]["hidden"].append(hidden)
    data[k]["eval_calls"].append(n_calls)

# ── Print table ──────────────────────────────────────────────────────────────

print(f"{'K':>4}  {'n':>3}  {'Dev Mean':>10}  {'Dev SD':>8}  {'Hidden Mean':>12}  {'Hidden SD':>10}  {'Calls Used':>11}")
print("-" * 72)

k_vals = sorted(data.keys())
table = []
for k in k_vals:
    d = data[k]
    n_dev = len(d["dev"])
    n_hid = len(d["hidden"])
    dev_mean = statistics.mean(d["dev"]) if d["dev"] else float("nan")
    dev_sd = statistics.stdev(d["dev"]) if len(d["dev"]) > 1 else 0.0
    hid_mean = statistics.mean(d["hidden"]) if d["hidden"] else float("nan")
    hid_sd = statistics.stdev(d["hidden"]) if len(d["hidden"]) > 1 else 0.0
    calls_mean = statistics.mean(d["eval_calls"]) if d["eval_calls"] else 0

    hid_str = f"{hid_mean:12.1f}" if d["hidden"] else "     pending"
    hid_sd_str = f"{hid_sd:10.1f}" if len(d["hidden"]) > 1 else "       —"

    print(f"{k:4d}  {n_dev:3d}  {dev_mean:10.1f}  {dev_sd:8.1f}  {hid_str}  {hid_sd_str}  {calls_mean:11.1f}")
    table.append((k, dev_mean, dev_sd, hid_mean, hid_sd, n_dev, n_hid))

# ── Proxy overfit test ───────────────────────────────────────────────────────

if len(k_vals) >= 3:
    k_lo, k_hi = k_vals[0], k_vals[-1]
    dev_lo = statistics.mean(data[k_lo]["dev"]) if data[k_lo]["dev"] else None
    dev_hi = statistics.mean(data[k_hi]["dev"]) if data[k_hi]["dev"] else None
    hid_lo = statistics.mean(data[k_lo]["hidden"]) if data[k_lo]["hidden"] else None
    hid_hi = statistics.mean(data[k_hi]["hidden"]) if data[k_hi]["hidden"] else None

    print(f"\nProxy overfit diagnostic:")
    if dev_lo and dev_hi:
        print(f"  Dev  K={k_lo} → K={k_hi}: {dev_lo:.1f} → {dev_hi:.1f}  (Δ={dev_hi-dev_lo:+.1f})")
    if hid_lo and hid_hi:
        print(f"  Hidden K={k_lo} → K={k_hi}: {hid_lo:.1f} → {hid_hi:.1f}  (Δ={hid_hi-hid_lo:+.1f})")
        if dev_lo and dev_hi and dev_hi > dev_lo and hid_hi <= hid_lo:
            print("  → Dev rises, hidden flat/drops: PROXY OVERFIT detected")
        elif dev_lo and dev_hi and dev_hi > dev_lo and hid_hi > hid_lo:
            print("  → Dev and hidden both rise: no proxy overfit (genuine improvement)")
        else:
            print("  → No clear proxy overfit pattern")
    elif not hid_lo or not hid_hi:
        print("  → Hidden scores still pending from platform")

# ── Optional plot ────────────────────────────────────────────────────────────

if PLOT:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.set_xlabel("K (evaluator call cap)")
        ax1.set_ylabel("Dev Score (80%-edge cut)", color="tab:blue")

        dev_means = [statistics.mean(data[k]["dev"]) for k in k_vals if data[k]["dev"]]
        dev_sds = [statistics.stdev(data[k]["dev"]) if len(data[k]["dev"]) > 1 else 0 for k in k_vals if data[k]["dev"]]
        k_dev = [k for k in k_vals if data[k]["dev"]]
        ax1.errorbar(k_dev, dev_means, yerr=dev_sds, marker="o", color="tab:blue", label="Dev (public)")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Hidden on secondary axis
        k_hid = [k for k in k_vals if data[k]["hidden"]]
        if k_hid:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Hidden Score (full-graph cut)", color="tab:red")
            hid_means = [statistics.mean(data[k]["hidden"]) for k in k_hid]
            hid_sds = [statistics.stdev(data[k]["hidden"]) if len(data[k]["hidden"]) > 1 else 0 for k in k_hid]
            ax2.errorbar(k_hid, hid_means, yerr=hid_sds, marker="s", color="tab:red", label="Hidden")
            ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.suptitle("MaxCut K-Curve: Dev vs Hidden Score")
        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/kcurve_maxcut.pdf", bbox_inches="tight")
        print(f"\nFigure saved to figures/kcurve_maxcut.pdf")
    except ImportError:
        print("\nmatplotlib not installed — skipping plot")
