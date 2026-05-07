#!/usr/bin/env python3
"""
Dev/Hidden Rank Analysis: dev rank vs hidden rank per task.

For each task, ranks all protocols by mean dev score and mean hidden score,
computes Spearman ρ, and flags rank inversions (especially top-1).

Output:
  results/dev_hidden_rank/spearman_per_task.tsv   — ρ, p-value, top-1 inversion flag
  results/dev_hidden_rank/rank_table.tsv          — full rank matrix per task
  results/dev_hidden_rank/inversion_summary.txt   — human-readable report
"""

import json
import os
import math
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results/full-v2")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../../results/dev_hidden_rank")

# Scoring direction per task (maximize = higher raw score is better)
SCORING_DIRECTION = {
    "bench-maxcut-g200":        "maximize",
    "bench-circle-packing-n20": "maximize",
    "bench-difference-bases":   "minimize",
    "bench-flat-poly-deg50":    "minimize",
    "bench-tsp-100":            "minimize",
    "bench-lj-n41":             "minimize",
    "bench-erdos-overlap":      "minimize",
    "bench-molecule-qed":       "maximize",
}

# Core tasks included in the current rank-analysis pass.
CORE_TASKS = list(SCORING_DIRECTION.keys())
CORE_PROTOCOLS = [
    "single-shot", "best-of-n", "self-refine", "vgs", "tot",
    "homo-chain", "cross-chain", "magicore", "debate", "moa", "hpe",
]

SENTINEL_DEV  = {  # scores to treat as invalid (floor exhibits etc)
    "bench-lj-n41": 1e20,     # positive energy sentinel → invalid
}
SENTINEL_HID = {
    "bench-lj-n41": 1e20,
}


def spearman_r(x, y):
    """Compute Spearman ρ and approximate p-value without scipy."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    def rank(arr):
        sorted_idx = sorted(range(n), key=lambda i: arr[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and arr[sorted_idx[j]] == arr[sorted_idx[i]]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                r[sorted_idx[k]] = avg
            i = j
        return r

    rx, ry = rank(x), rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if sx == 0 or sy == 0:
        return float("nan"), float("nan")
    rho = num / (sx * sy)

    # t-statistic approximation
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt(n - 2) / math.sqrt(1 - rho ** 2)
    # Two-tailed p-value via normal approximation for n >= 10
    z = abs(t) / math.sqrt(1 + t*t/max(n-2, 1)) * math.sqrt(n - 2)
    p = 2 * (1 - _normal_cdf(abs(t) * math.sqrt((n-2) / (n - 1 + t*t / (n-2)))))
    return rho, p


def _normal_cdf(x):
    """Approximation of standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load_results():
    """Load all result JSON files. Returns list of dicts."""
    records = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue

        task_id    = d.get("taskId")
        protocol   = d.get("protocolId")
        seed       = d.get("seed")
        dev_score  = (d.get("bestArtifact") or {}).get("devScore")
        hid_score  = d.get("hiddenScore")

        if task_id is None or protocol is None or dev_score is None:
            continue

        records.append({
            "task":     task_id,
            "protocol": protocol,
            "seed":     seed,
            "dev":      float(dev_score),
            "hidden":   float(hid_score) if hid_score is not None else None,
        })
    return records


def mean_scores(records, task, protocol):
    """Mean dev and hidden scores for (task, protocol), excluding sentinels."""
    dev_sentinel = SENTINEL_DEV.get(task, 1e30)
    hid_sentinel = SENTINEL_HID.get(task, 1e30)
    rows = [r for r in records if r["task"] == task and r["protocol"] == protocol]
    devs = [r["dev"] for r in rows if abs(r["dev"]) < dev_sentinel]
    hids = [r["hidden"] for r in rows if r["hidden"] is not None and abs(r["hidden"]) < hid_sentinel]
    if not devs or not hids:
        return None, None
    return sum(devs) / len(devs), sum(hids) / len(hids)


def rank_protocols(scores_dict, direction):
    """Rank protocols; returns dict protocol→rank (1=best)."""
    items = [(p, s) for p, s in scores_dict.items() if s is not None]
    reverse = (direction == "maximize")
    items.sort(key=lambda x: x[1], reverse=reverse)
    return {p: i + 1 for i, (p, _) in enumerate(items)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = load_results()
    print(f"Loaded {len(records)} result records.")

    # Restrict to the primary comparison set for paper-facing rank analysis.
    all_protocols = [p for p in CORE_PROTOCOLS if any(
        r["task"] in CORE_TASKS and r["protocol"] == p for r in records
    )]

    # ── Per-task Spearman analysis ────────────────────────────────────────────
    spearman_rows = []
    rank_table_rows = []
    inversion_lines = []

    for task in CORE_TASKS:
        direction = SCORING_DIRECTION[task]

        dev_means, hid_means = {}, {}
        for proto in all_protocols:
            d, h = mean_scores(records, task, proto)
            if d is not None:
                dev_means[proto] = d
            if h is not None:
                hid_means[proto] = h

        # Only protocols with both dev and hidden
        common = [p for p in all_protocols if p in dev_means and p in hid_means]
        if len(common) < 3:
            print(f"  {task}: only {len(common)} protocols with both scores, skipping")
            continue

        dev_ranks = rank_protocols({p: dev_means[p] for p in common}, direction)
        hid_ranks = rank_protocols({p: hid_means[p] for p in common}, direction)

        dev_rank_list = [dev_ranks[p] for p in common]
        hid_rank_list = [hid_ranks[p] for p in common]

        rho, pval = spearman_r(dev_rank_list, hid_rank_list)

        # Top-1 inversion
        dev_top1 = min(common, key=lambda p: dev_ranks[p])
        hid_top1 = min(common, key=lambda p: hid_ranks[p])
        top1_inverted = dev_top1 != hid_top1

        # Top-3 inversions (how many of dev top-3 are NOT in hid top-3)
        dev_top3 = {p for p in common if dev_ranks[p] <= 3}
        hid_top3 = {p for p in common if hid_ranks[p] <= 3}
        top3_overlap = len(dev_top3 & hid_top3)

        spearman_rows.append({
            "task": task,
            "n_protocols": len(common),
            "rho": rho,
            "p_value": pval,
            "top1_dev": dev_top1,
            "top1_hidden": hid_top1,
            "top1_inverted": top1_inverted,
            "top3_overlap": top3_overlap,
        })

        # Full rank table for this task
        for proto in sorted(common):
            rank_table_rows.append({
                "task": task,
                "protocol": proto,
                "dev_mean": dev_means[proto],
                "hidden_mean": hid_means[proto],
                "dev_rank": dev_ranks[proto],
                "hidden_rank": hid_ranks[proto],
                "rank_delta": hid_ranks[proto] - dev_ranks[proto],
            })

        # Human-readable inversion report
        inv_str = "INVERTED" if top1_inverted else "stable"
        inversion_lines.append(
            f"\n{'='*60}\n{task}  (direction={direction}, n={len(common)}, ρ={rho:.3f}, p={pval:.3f})"
        )
        inversion_lines.append(f"  Top-1 dev:    {dev_top1}  (score={dev_means[dev_top1]:.4f})")
        inversion_lines.append(f"  Top-1 hidden: {hid_top1}  (score={hid_means[hid_top1]:.4f})  [{inv_str}]")
        inversion_lines.append(f"  Top-3 overlap: {top3_overlap}/3")
        inversion_lines.append("  Full ranking (dev rank → hidden rank):")
        sorted_by_dev = sorted(common, key=lambda p: dev_ranks[p])
        for proto in sorted_by_dev:
            delta = hid_ranks[proto] - dev_ranks[proto]
            arrow = f"{'▲' if delta < 0 else '▼' if delta > 0 else '='}{abs(delta)}" if delta != 0 else "  ="
            inversion_lines.append(
                f"    #{dev_ranks[proto]:2d} {proto:<22s}  dev={dev_means[proto]:10.4f}  hid={hid_means[proto]:10.4f}  → #{hid_ranks[proto]:2d} ({arrow})"
            )

    # ── Write outputs ─────────────────────────────────────────────────────────

    # 1. Spearman summary TSV
    spearman_path = os.path.join(OUT_DIR, "spearman_per_task.tsv")
    with open(spearman_path, "w") as f:
        f.write("task\tn_protocols\trho\tp_value\ttop1_dev\ttop1_hidden\ttop1_inverted\ttop3_overlap\n")
        for row in spearman_rows:
            f.write(
                f"{row['task']}\t{row['n_protocols']}\t{row['rho']:.4f}\t{row['p_value']:.4f}\t"
                f"{row['top1_dev']}\t{row['top1_hidden']}\t{row['top1_inverted']}\t{row['top3_overlap']}\n"
            )
    print(f"Written: {spearman_path}")

    # 2. Full rank table TSV
    rank_path = os.path.join(OUT_DIR, "rank_table.tsv")
    with open(rank_path, "w") as f:
        f.write("task\tprotocol\tdev_mean\thidden_mean\tdev_rank\thidden_rank\trank_delta\n")
        for row in rank_table_rows:
            f.write(
                f"{row['task']}\t{row['protocol']}\t{row['dev_mean']:.6f}\t{row['hidden_mean']:.6f}\t"
                f"{row['dev_rank']}\t{row['hidden_rank']}\t{row['rank_delta']}\n"
            )
    print(f"Written: {rank_path}")

    # 3. Human-readable report
    report_path = os.path.join(OUT_DIR, "inversion_summary.txt")
    with open(report_path, "w") as f:
        f.write("Dev/Hidden Rank Analysis — Dev vs Hidden Rank Inversions\n")
        f.write("Anonymous benchmark release (NeurIPS 2026)\n")
        f.write("=" * 60 + "\n\n")
        f.write("SUMMARY\n")
        f.write(f"{'Task':<35s}  {'ρ':>6}  {'p':>6}  {'Top-1 inversion?':>20}  {'Top-3 overlap':>13}\n")
        f.write("-" * 90 + "\n")
        for row in spearman_rows:
            inv_flag = "*** INVERTED ***" if row["top1_inverted"] else "stable"
            f.write(
                f"{row['task']:<35s}  {row['rho']:>6.3f}  {row['p_value']:>6.3f}  {inv_flag:>20}  {row['top3_overlap']:>5}/3\n"
            )
        f.write("\n")
        f.write("\n".join(inversion_lines))
    print(f"Written: {report_path}")

    # ── Print summary to stdout ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("DEV/HIDDEN RANK ANALYSIS SUMMARY")
    print(f"{'Task':<35s}  {'ρ':>6}  {'p':>6}  {'Top-1 OK?':>12}  {'Top-3':>6}")
    print("-" * 70)
    n_inverted = 0
    for row in spearman_rows:
        inv = "INVERTED" if row["top1_inverted"] else "ok"
        if row["top1_inverted"]:
            n_inverted += 1
        print(f"{row['task']:<35s}  {row['rho']:>6.3f}  {row['p_value']:>6.3f}  {inv:>12}  {row['top3_overlap']:>3}/3")
    print("-" * 70)
    print(f"Tasks with top-1 inversion: {n_inverted}/{len(spearman_rows)}")
    mean_rho = sum(r["rho"] for r in spearman_rows if not math.isnan(r["rho"])) / max(len(spearman_rows), 1)
    print(f"Mean Spearman ρ across tasks: {mean_rho:.3f}")


if __name__ == "__main__":
    main()
