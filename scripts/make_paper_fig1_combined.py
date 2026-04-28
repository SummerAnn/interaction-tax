#!/usr/bin/env python3
"""
Generate combined Figure 1 for icml2026_combined.tex.
Two panels:
  (a) AggMEG lollipop — "no protocol reliably beats baseline"
  (b) MIG quadrant scatter — "interaction erases diversity"
Output: figures/paper_fig1_combined.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size":         8.5,
    "axes.labelsize":    8.5,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   8.5,
    "figure.dpi":        300,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

RED    = "#c0392b"
GREEN  = "#27ae60"
CORAL  = "#e07b6a"   # muted red for multi-agent protocols
STEEL  = "#6c8ebf"   # blue-gray for single-agent controls
GRAY   = "#7f8c8d"
LGRAY  = "#bdc3c7"
BLACK  = "#2c3e50"
CREAM  = "#fdfaf6"

fig, (ax_l, ax_r) = plt.subplots(
    1, 2, figsize=(6.9, 3.6),
    gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.38}
)

# ============================================================================
# LEFT PANEL — AggMEG lollipop (best at top, worst at bottom)
# ============================================================================
# Data from Table 3 (tab:aggmeg), sorted worst→best so best appears at top
data = [
    ("HPE",          -0.225, -0.247, -0.201, "multi"),
    ("Single-Shot",  -0.107, -0.118, -0.094, "ctrl"),
    ("Debate",       -0.088, -0.142, -0.031, "multi"),
    ("VGS",          -0.072, -0.091, -0.048, "ctrl"),
    ("MAgICoRe",     -0.063, -0.098, -0.028, "multi"),
    ("Homo-Chain",   -0.056, -0.080, -0.030, "multi"),
    ("Cross-Chain",  -0.053, -0.089, -0.016, "multi"),
    ("Best-of-$N$",  -0.051, -0.084, -0.013, "ctrl"),
    ("Self-Refine",  -0.038, -0.071, -0.005, "ctrl"),
    ("MoA",          -0.022, -0.059, +0.015, "moa"),
]

names  = [d[0] for d in data]
megs   = [d[1] for d in data]
ci_lo  = [d[2] for d in data]
ci_hi  = [d[3] for d in data]
types  = [d[4] for d in data]

color_map = {"moa": GREEN, "ctrl": STEEL, "multi": CORAL}
colors = [color_map[t] for t in types]

y = np.arange(len(data))

for i, (m, lo, hi, c, name) in enumerate(zip(megs, ci_lo, ci_hi, colors, names)):
    # stem line from 0 to value
    ax_l.plot([0, m], [i, i], color=c, linewidth=1.0, alpha=0.5, zorder=2)
    # CI whisker
    ax_l.plot([lo, hi], [i, i], color=c, linewidth=2.2, alpha=0.7,
              solid_capstyle="round", zorder=3)
    # dot
    ax_l.scatter(m, i, color=c, s=52, zorder=5, edgecolors="white",
                 linewidths=0.6)

# MoA: annotate CI crosses zero
ax_l.annotate("CI includes 0",
    xy=(0.015, 9), xytext=(0.045, 8.3),
    fontsize=6.5, color=GREEN,
    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.8, mutation_scale=6),
    ha="left", va="top")

# zero line
ax_l.axvline(0, color=BLACK, linewidth=0.9, linestyle="--", alpha=0.5, zorder=1)

ax_l.set_yticks(y)
ax_l.set_yticklabels(names, fontsize=8)
for tick, c in zip(ax_l.get_yticklabels(), colors):
    tick.set_color(c)
    if c == GREEN:
        tick.set_fontweight("bold")

ax_l.set_xlabel("Aggregate MEG  (95% CI, 9 tasks)", labelpad=4)
ax_l.set_xlim(-0.30, 0.09)
ax_l.set_ylim(-0.7, len(data) - 0.3)
ax_l.spines["left"].set_visible(False)
ax_l.spines["top"].set_visible(False)
ax_l.spines["right"].set_visible(False)
ax_l.tick_params(axis="y", length=0, pad=4)
ax_l.xaxis.grid(True, color="#e8e8e8", linewidth=0.5, zorder=0)
ax_l.set_axisbelow(True)

# legend
leg_handles = [
    mpatches.Patch(color=GREEN,  label="MoA (diverse)"),
    mpatches.Patch(color=CORAL,  label="Multi-agent w/ interaction"),
    mpatches.Patch(color=STEEL,  label="Single-agent controls"),
]
ax_l.legend(handles=leg_handles, fontsize=6.2, loc="lower right",
            frameon=True, framealpha=0.92, edgecolor="#dddddd",
            handlelength=0.8, labelspacing=0.3, borderpad=0.5)

ax_l.set_title("(a) No protocol reliably beats single-agent baseline",
               fontsize=8, loc="left", pad=6, fontstyle="italic")

# ============================================================================
# RIGHT PANEL — MIG quadrant scatter
# ============================================================================
# x = same-model MIG, y = diverse-model MIG
# From mig_extended.tsv (averaged across 8 tasks)
protocols_mig = [
    ("Chain",      0.051, -0.024),   # Homo=same, Cross=diverse
    ("MAgICoRe",   0.044, -0.035),
    ("Debate",     0.012, -0.078),
    ("MoA",        0.012,  0.016),
    ("HPE",       -0.163, -0.145),
]

xvals = [p[1] for p in protocols_mig]
yvals = [p[2] for p in protocols_mig]
plabels = [p[0] for p in protocols_mig]

pt_colors = [RED, RED, RED, GREEN, GRAY]

lim = 0.22

# ── quadrant shading ────────────────────────────────────────────────────────
# bottom-right: same+, diverse- = DIVERSITY TAX ZONE
ax_r.fill_between([0, lim], [-lim, -lim], [0, 0],
                  color=RED, alpha=0.07, zorder=0)
# top-right: both + = interaction helps
ax_r.fill_between([0, lim], [0, 0], [lim, lim],
                  color=GREEN, alpha=0.05, zorder=0)
# bottom-left: both - = interaction always hurts
ax_r.fill_between([-lim, 0], [-lim, -lim], [0, 0],
                  color=GRAY, alpha=0.07, zorder=0)

# quadrant labels — pushed to corners to avoid protocol labels
ax_r.text(0.105, -0.195, "diversity tax zone",
          fontsize=5.8, color=RED, alpha=0.70,
          ha="center", va="bottom", style="italic", fontweight="bold")
ax_r.text(0.105, 0.195, "interaction\nhelps both",
          fontsize=5.6, color=GREEN, alpha=0.60,
          ha="center", va="top", style="italic")
ax_r.text(-0.105, -0.145, "interaction\nalways hurts",
          fontsize=5.6, color=GRAY, alpha=0.60,
          ha="center", va="center", style="italic")

# ── reference lines ─────────────────────────────────────────────────────────
# y = x diagonal (no change with diversity)
ax_r.plot([-lim, lim], [-lim, lim], color=LGRAY,
          linewidth=0.9, linestyle=":", zorder=1, label="y = x  (no diversity effect)")
# zero lines
ax_r.axhline(0, color=BLACK, linewidth=0.7, linestyle="--", alpha=0.45, zorder=1)
ax_r.axvline(0, color=BLACK, linewidth=0.7, linestyle="--", alpha=0.45, zorder=1)

# ── protocol points ──────────────────────────────────────────────────────────
# label positions: (dx, dy, ha, va)
label_offsets = {
    "Chain":     ( 0.016,  0.006, "left",  "bottom"),
    "MAgICoRe":  ( 0.016, -0.006, "left",  "top"),
    "Debate":    (-0.016,  0.000, "right", "center"),
    "MoA":       ( 0.016,  0.006, "left",  "bottom"),
    "HPE":       ( 0.016,  0.000, "left",  "center"),
}

for (name, xs, ys), c in zip(protocols_mig, pt_colors):
    ax_r.scatter(xs, ys, color=c, s=72, zorder=5,
                 edgecolors="white", linewidths=0.8)
    dx, dy, ha, va = label_offsets[name]
    ax_r.text(xs + dx, ys + dy, name,
              fontsize=7.5, color=c, ha=ha, va=va,
              fontweight="bold" if name == "MoA" else "normal")

# ── axes ─────────────────────────────────────────────────────────────────────
ax_r.set_xlabel("MIG — same-model agents", labelpad=4)
ax_r.set_ylabel("MIG — diverse-model agents", labelpad=4)
ax_r.set_xlim(-lim, lim)
ax_r.set_ylim(-lim, lim)
ax_r.set_aspect("equal")
ax_r.spines["top"].set_visible(False)
ax_r.spines["right"].set_visible(False)
ax_r.xaxis.grid(True, color="#e8e8e8", linewidth=0.4, zorder=0)
ax_r.yaxis.grid(True, color="#e8e8e8", linewidth=0.4, zorder=0)
ax_r.set_axisbelow(True)

ax_r.text(-0.195, 0.205, "y = x", fontsize=6, color=LGRAY,
          style="italic", rotation=45, ha="center")

ax_r.set_title("(b) Interaction erases diversity",
               fontsize=8, loc="left", pad=6, fontstyle="italic")

# ============================================================================
fig.savefig("figures/paper_fig1_combined.pdf", bbox_inches="tight", dpi=300)
fig.savefig("figures/paper_fig1_combined.png", bbox_inches="tight", dpi=300)
print("saved figures/paper_fig1_combined.{pdf,png}")
plt.close(fig)
