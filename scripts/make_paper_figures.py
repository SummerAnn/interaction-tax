#!/usr/bin/env python3
"""Generate Figures 1–4 for the NeurIPS 2026 paper (9-task analysis)."""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RANK_TABLE = ROOT / "results" / "dev_hidden_rank" / "rank_table.tsv"
MEG_TSV = ROOT / "results" / "analysis" / "meg_full.tsv"
OUTDIR = ROOT / "figures"

MAXCUT_TASK = "bench-maxcut-g200"
CORE_PROTOCOLS = [
    "single-shot",
    "best-of-n",
    "self-refine",
    "vgs",
    "homo-chain",
    "cross-chain",
    "magicore",
    "debate",
    "moa",
    "hpe",
]
TASKS = ["MaxCut", "CircPack", "DiffBases", "FlatPoly", "TSP-100", "LJ-n41", "Erdős", "MolQED", "TSP-50"]

# Bar chart ordering: sorted by AggMEG descending, with moa-same-model inserted
BAR_PROTOCOLS = [
    "moa",
    "self-refine",
    "best-of-n",
    "cross-chain",
    "homo-chain",
    "magicore",
    "moa-same-model",
    "vgs",
    "debate",
    "single-shot",
    "hpe",
]

STORY_BAR_PROTOCOLS = [
    "moa",
    "best-of-n",
    "vgs",
    "moa-same-model",
    "debate",
    "hpe",
]


LABELS = {
    "single-shot": "SS",
    "best-of-n": "BoN",
    "self-refine": "SR",
    "vgs": "VGS",
    "homo-chain": "HC",
    "cross-chain": "CC",
    "magicore": "MAG",
    "debate": "DEB",
    "moa": "MoA",
    "moa-same-model": "MoA-same",
    "hpe": "HPE",
}

FULL_LABELS = {
    "moa": "Mixture-of-Agents (diverse)",
    "best-of-n": "Best-of-N",
    "vgs": "Verifier-guided search",
    "moa-same-model": "MoA (same model)",
    "debate": "Debate",
    "hpe": "Planner-Executor",
    "self-refine": "Self-Refine",
    "cross-chain": "Cross-chain",
    "homo-chain": "Homo-chain",
    "magicore": "MAgICoRe",
    "single-shot": "Single-shot",
}

FAMILIES = {
    "single-shot": "single",
    "best-of-n": "single",
    "self-refine": "single",
    "vgs": "single",
    "homo-chain": "chain",
    "cross-chain": "chain",
    "magicore": "role",
    "debate": "role",
    "moa": "ensemble",
    "moa-same-model": "ensemble-same",
    "hpe": "decomp",
}

COLORS = {
    "single": "#4e79a7",
    "chain": "#59a14f",
    "role": "#f28e2b",
    "ensemble": "#e15759",
    "ensemble-same": "#ff9d9a",
    "decomp": "#b07aa1",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_float(raw: str) -> float:
    cleaned = raw.replace("*", "").strip()
    if cleaned == "nan" or cleaned == "":
        return float("nan")
    return float(cleaned)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTDIR / f"{stem}.pdf"
    png_path = OUTDIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


# ── Figure 1: MaxCut dev/hidden rank scatter (core 10 protocols) ──────

def plot_maxcut_rank_inversion() -> None:
    all_rows = [r for r in read_tsv(RANK_TABLE) if r["task"] == MAXCUT_TASK]
    # Filter to core 10 protocols only
    core_set = set(CORE_PROTOCOLS)
    rows = [r for r in all_rows if r["protocol"] in core_set]

    # Recompute ranks within core-10 set
    rows.sort(key=lambda r: float(r["dev_mean"]), reverse=True)
    for i, r in enumerate(rows):
        r["dev_rank_core"] = i + 1
    rows.sort(key=lambda r: float(r["hidden_mean"]), reverse=True)
    for i, r in enumerate(rows):
        r["hidden_rank_core"] = i + 1

    n = len(rows)
    d_sq = sum((r["dev_rank_core"] - r["hidden_rank_core"]) ** 2 for r in rows)
    rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0.5, n + 0.5], [0.5, n + 0.5], linestyle="--", linewidth=1.1, color="#c7c7c7", zorder=0)

    for row in rows:
        protocol = row["protocol"]
        dr = row["dev_rank_core"]
        hr = row["hidden_rank_core"]
        family = FAMILIES.get(protocol, "single")
        color = COLORS[family]

        size = 72
        edge = "white"
        lw = 1.0
        if protocol in {"vgs", "cross-chain"}:
            size = 130
            edge = "black"
            lw = 1.4

        ax.scatter(dr, hr, s=size, color=color, edgecolor=edge, linewidth=lw, zorder=3)

        # Offset labels to avoid overlap
        offsets = {
            "vgs": (0.25, -0.15),
            "cross-chain": (0.25, 0.20),
            "best-of-n": (0.25, -0.15),
            "magicore": (0.25, 0.20),
            "self-refine": (0.25, -0.15),
            "debate": (0.25, 0.20),
            "moa": (0.25, -0.15),
            "homo-chain": (0.25, 0.0),
            "hpe": (0.25, 0.0),
            "single-shot": (0.25, 0.0),
        }
        dx, dy = offsets.get(protocol, (0.25, 0.0))
        ax.text(
            dr + dx, hr + dy,
            LABELS.get(protocol, protocol),
            fontsize=9, ha="left", va="center",
        )

    ax.invert_yaxis()
    ax.set_xlim(0.3, n + 0.7)
    ax.set_ylim(n + 0.7, 0.3)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_xlabel("Development-evaluator rank (1 = best)")
    ax.set_ylabel("Hidden-evaluator rank (1 = best)")
    ax.set_title(
        "MaxCut: visible feedback changes the ranking",
        fontsize=13, pad=12,
    )
    ax.text(
        0.02, -0.12,
        f"Core-protocol raw-score Spearman \u03c1 = {rho:.3f}; top-1 protocol flips from VGS to cross-chain.",
        transform=ax.transAxes, fontsize=9.5, color="#444444",
    )

    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=color, markersize=8)
        for name, color in [
            ("single-agent", COLORS["single"]),
            ("chain", COLORS["chain"]),
            ("role-structured", COLORS["role"]),
            ("ensemble", COLORS["ensemble"]),
            ("decomposition", COLORS["decomp"]),
        ]
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8.5, frameon=False, ncol=2)
    ax.grid(alpha=0.15, linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, "maxcut_rank_inversion")
    plt.close(fig)


# ── Figure 2: Aggregate MEG bar chart with bootstrap CIs ─────────────

def plot_agg_meg_bar() -> None:
    rows = read_tsv(MEG_TSV)
    row_map = {r["protocol"]: r for r in rows}
    protocols = [p for p in BAR_PROTOCOLS if p in row_map]

    vals = [parse_float(row_map[p]["AggMEG"]) for p in protocols]
    ci_lo = [parse_float(row_map[p]["CI_lo"]) for p in protocols]
    ci_hi = [parse_float(row_map[p]["CI_hi"]) for p in protocols]
    err_lo = [v - lo for v, lo in zip(vals, ci_lo)]
    err_hi = [hi - v for v, hi in zip(vals, ci_hi)]

    colors = []
    for p, v in zip(protocols, vals):
        family = FAMILIES.get(p, "single")
        if v > 0:
            colors.append("#2e8b57")
        else:
            colors.append(COLORS.get(family, "#999999"))

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    y = np.arange(len(protocols))
    ax.barh(
        y, vals, color=colors, edgecolor="none",
        xerr=[err_lo, err_hi], error_kw=dict(ecolor="#555555", capsize=3, linewidth=1.2),
    )
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([FULL_LABELS.get(p, p) for p in protocols], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate hidden-score MEG (9 tasks; MoA arms n=10, others n=5)")
    ax.set_title(
        "Diverse MoA is consistent with null; same-model MoA is clearly negative",
        fontsize=12.5, pad=10,
    )
    ax.grid(axis="x", alpha=0.15, linewidth=0.7)

    for yi, v in zip(y, vals):
        offset = 0.006
        if v >= 0:
            ax.text(v + offset, yi, f"{v:+.03f}", va="center", ha="left", fontsize=8.5)
        else:
            ax.text(v - offset, yi, f"{v:+.03f}", va="center", ha="right", fontsize=8.5)

    ax.text(
        0.0, -0.11,
        "Error bars: bootstrap 95% CI (2000 resamples). "
        "BoN, SR, and VGS are the single-agent controls in the MEG denominator.",
        transform=ax.transAxes, fontsize=9, color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, "agg_meg_core_bar")
    plt.close(fig)


# ── Figure 3: MEG heatmap (9 tasks x 10 core protocols) ──────────────

def plot_meg_heatmap() -> None:
    rows = read_tsv(MEG_TSV)
    row_map = {r["protocol"]: r for r in rows}
    values = []
    for protocol in CORE_PROTOCOLS:
        row = row_map[protocol]
        values.append([parse_float(row[task]) for task in TASKS])
    mat = np.array(values)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            text_color = "black" if abs(val) < 0.15 else "white"
            ax.text(j, i, f"{val:+.02f}", ha="center", va="center", fontsize=7.5, color=text_color)

    ax.set_xticks(np.arange(len(TASKS)))
    ax.set_xticklabels(TASKS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(CORE_PROTOCOLS)))
    ax.set_yticklabels([LABELS[p] for p in CORE_PROTOCOLS])
    ax.set_title("Hidden-score MEG by task and protocol (9 tasks)", fontsize=13, pad=10)
    ax.set_xlabel("Task")
    ax.set_ylabel("Protocol")

    cbar = fig.colorbar(im, ax=ax, shrink=0.93)
    cbar.set_label("MEG")

    ax.text(
        0.0, -0.18,
        "Positive MEG means beating max(Self-Refine, Best-of-N, VGS) on the hidden evaluator. "
        "All 9 tasks included (Erd\u0151s and MolQED complete).",
        transform=ax.transAxes, fontsize=8.5, color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, "meg_heatmap_primary")
    plt.close(fig)


# ── Figure 4: Story bar (5 key protocols + MoA-same-model) ───────────

def plot_story_bar() -> None:
    rows = read_tsv(MEG_TSV)
    row_map = {r["protocol"]: r for r in rows}
    protocols = [p for p in STORY_BAR_PROTOCOLS if p in row_map]

    vals = [parse_float(row_map[p]["AggMEG"]) for p in protocols]
    ci_lo = [parse_float(row_map[p]["CI_lo"]) for p in protocols]
    ci_hi = [parse_float(row_map[p]["CI_hi"]) for p in protocols]
    err_lo = [v - lo for v, lo in zip(vals, ci_lo)]
    err_hi = [hi - v for v, hi in zip(vals, ci_hi)]

    colors = []
    for p, v in zip(protocols, vals):
        if p == "moa":
            colors.append("#2e8b57")
        elif p == "moa-same-model":
            colors.append("#ff9d9a")
        elif p in {"best-of-n", "vgs"}:
            colors.append(COLORS["single"])
        else:
            colors.append(COLORS[FAMILIES.get(p, "role")])

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(protocols))
    ax.barh(
        y, vals, color=colors, edgecolor="none",
        xerr=[err_lo, err_hi], error_kw=dict(ecolor="#555555", capsize=3, linewidth=1.2),
    )
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([FULL_LABELS.get(p, p) for p in protocols], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate hidden-score MEG (9 tasks; MoA arms n=10, others n=5)")
    ax.set_title(
        "Diverse MoA is consistent with null; same-model MoA is negative; HPE inverts",
        fontsize=12, pad=10,
    )
    ax.grid(axis="x", alpha=0.15, linewidth=0.7)

    for yi, v in zip(y, vals):
        offset = 0.006
        if v >= 0:
            ax.text(v + offset, yi, f"{v:+.03f}", va="center", ha="left", fontsize=9)
        else:
            ax.text(v - offset, yi, f"{v:+.03f}", va="center", ha="right", fontsize=9)

    ax.text(
        0.0, -0.15,
        "The diverse\u2013vs.\u2013same-model MoA split is the clearest evidence that model diversity matters. "
        "HPE scores best on dev but worst on hidden.",
        transform=ax.transAxes, fontsize=9, color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, "agg_meg_story_bar")
    plt.close(fig)


# ── Slopegraph variant (core 10 only) ────────────────────────────────

def plot_maxcut_rank_slopegraph() -> None:
    all_rows = [r for r in read_tsv(RANK_TABLE) if r["task"] == MAXCUT_TASK]
    core_set = set(CORE_PROTOCOLS)
    rows = [r for r in all_rows if r["protocol"] in core_set]

    # Recompute ranks within core-10 set
    rows.sort(key=lambda r: float(r["dev_mean"]), reverse=True)
    for i, r in enumerate(rows):
        r["dev_rank_core"] = i + 1
    rows.sort(key=lambda r: float(r["hidden_mean"]), reverse=True)
    for i, r in enumerate(rows):
        r["hidden_rank_core"] = i + 1
    rows.sort(key=lambda r: r["dev_rank_core"])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    x_dev, x_hid = 0.0, 1.0

    for row in rows:
        protocol = row["protocol"]
        dr = row["dev_rank_core"]
        hr = row["hidden_rank_core"]
        family = FAMILIES.get(protocol, "single")
        color = COLORS[family]
        alpha = 0.6
        lw = 1.8
        z = 2

        if protocol in {"vgs", "cross-chain"}:
            color = "#111111"
            alpha = 1.0
            lw = 2.8
            z = 4

        ax.plot([x_dev, x_hid], [dr, hr], color=color, alpha=alpha, linewidth=lw, zorder=z)

        left_label = LABELS.get(protocol, protocol)
        right_label = LABELS.get(protocol, protocol)
        if protocol == "vgs":
            left_label = "VGS (dev #1)"
        if protocol == "cross-chain":
            right_label = "CC (hidden #1)"

        ax.text(x_dev - 0.05, dr, left_label, ha="right", va="center", fontsize=9)
        ax.text(x_hid + 0.05, hr, right_label, ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([x_dev, x_hid])
    ax.set_xticklabels(["Development rank", "Hidden rank"])
    ax.set_yticks(range(1, n + 1))
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("MaxCut protocol ranking changes under hidden evaluation", fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.15, linewidth=0.7)

    d_sq = sum((r["dev_rank_core"] - r["hidden_rank_core"]) ** 2 for r in rows)
    rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))

    ax.text(
        0.0, -0.10,
        f"VGS is best on the visible score; cross-chain is best on the hidden score. "
        f"\u03c1 = {rho:.3f}, top-1 inverted.",
        transform=ax.transAxes, fontsize=9.5, color="#444444",
    )

    fig.tight_layout()
    save_figure(fig, "maxcut_rank_slopegraph")
    plt.close(fig)


def main() -> None:
    plot_maxcut_rank_inversion()
    plot_maxcut_rank_slopegraph()
    plot_meg_heatmap()
    plot_agg_meg_bar()
    plot_story_bar()


if __name__ == "__main__":
    main()
