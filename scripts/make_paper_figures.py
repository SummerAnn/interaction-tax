#!/usr/bin/env python3
"""
Generate clean publication figures for icml2026_combined.tex.
Outputs PDF + PNG to figures/paper_*.

Figure 1: MIG flip -- connected dot plot (single column)
Figure 2: Two-panel -- 2x2 factorial (left) + multi-synth robustness (right)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size":        9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

RED    = "#c0392b"
GREEN  = "#27ae60"
BLUE   = "#2980b9"
GRAY   = "#7f8c8d"
LGRAY  = "#bdc3c7"
BLACK  = "#2c3e50"


# ============================================================================
# FIGURE 1 — MIG flip
# ============================================================================
# Data from Table 5 (tab:mig). Order: flipped (top) → MoA (exception) → HPE
families   = ["Debate", "MAgICoRe", "Chain", "MoA", "HPE"]
mig_same   = [ 0.012,   0.044,      0.051,   0.012, -0.163]
mig_div    = [-0.078,  -0.035,     -0.024,   0.016, -0.145]

fig1, ax = plt.subplots(figsize=(3.4, 3.1))

y = np.arange(len(families))

for i, (s, d, name) in enumerate(zip(mig_same, mig_div, families)):
    if name == "MoA":
        col = GREEN
    elif name == "HPE":
        col = GRAY
    else:
        col = RED  # flipped

    # connecting line
    ax.plot([s, d], [i, i], color=col, linewidth=1.8, alpha=0.85, zorder=3,
            solid_capstyle="round")
    # same-model dot (open circle)
    ax.scatter(s, i, color="white", edgecolors=GRAY, s=50, zorder=5,
               linewidths=1.2)
    # diverse-model dot (filled diamond)
    ax.scatter(d, i, color=col, marker="D", s=44, zorder=6)

    # delta label — always at a consistent right margin
    delta = d - s
    sign = "+" if delta >= 0 else ""
    label_x = 0.105  # fixed right margin for all labels
    ax.text(label_x, i, f"{sign}{delta:.3f}",
            va="center", ha="left", fontsize=7.8,
            color=col, fontweight="bold" if name == "MoA" else "normal")

# zero line
ax.axvline(0, color=BLACK, linewidth=0.8, linestyle="--", alpha=0.55, zorder=2)

# light shading
ax.axvspan(-0.28, 0, alpha=0.035, color=RED,   zorder=0)
ax.axvspan(0, 0.10, alpha=0.035, color=GREEN, zorder=0)

# "Δ =" header above labels
ax.text(0.105, len(families) - 0.05, "Δ",
        va="bottom", ha="left", fontsize=7.5, color=GRAY, style="italic")

ax.set_yticks(y)
ax.set_yticklabels(families, fontsize=9.5)
ax.set_xlabel("Marginal Interaction Gain (MIG)", labelpad=5)
ax.set_xlim(-0.28, 0.16)
ax.set_ylim(-0.7, len(families) - 0.1)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=5)
ax.xaxis.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)

# compact inline legend using text annotations
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor=GRAY, markeredgewidth=1.2, markersize=6,
           label="Same-model agents"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=RED,
           markersize=6, label="Diverse-model — flips negative (Chain/MAgICoRe/Debate)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=GREEN,
           markersize=6, label="Diverse-model — MoA (stays positive)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=GRAY,
           markersize=6, label="HPE (already negative both configs)"),
]
ax.legend(handles=legend_handles, fontsize=6.2,
          loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=2, frameon=True, framealpha=0.95, edgecolor="#dddddd",
          handlelength=0.8, labelspacing=0.4, borderpad=0.6,
          columnspacing=0.8)

fig1.tight_layout(pad=0.6)
fig1.subplots_adjust(bottom=0.28)

for ext in ("pdf", "png"):
    path = f"figures/paper_fig1_mig_flip.{ext}"
    fig1.savefig(path, bbox_inches="tight", dpi=300)
    print(f"saved {path}")

plt.close(fig1)


# ============================================================================
# FIGURE 2 — two-panel: 2x2 factorial (left) + multi-synth robustness (right)
# ============================================================================
fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.8, 2.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

# ── Left panel: 2x2 factorial coefficients ──────────────────────────────────
# Data from Table 8 (tab:2x2) and text
terms   = ["Diversity\n× Synthesis\n(interaction)", "Synthesis\nstep", "Backbone\ndiversity"]
ests    = [0.046,  -0.010,  0.188]
ci_lo   = [0.046 - (-0.130), 0.010 + 0.134, 0.188 - 0.064]   # half-widths: est - lo_bound
ci_hi   = [0.222 - 0.046,    0.115 - (-0.010), 0.312 - 0.188]

colors_l = [GRAY, GRAY, GREEN]
markers  = ["o", "o", "D"]

y_l = np.arange(len(terms))

for i, (e, lo, hi, c, m) in enumerate(zip(ests, ci_lo, ci_hi, colors_l, markers)):
    ax_l.errorbar(e, i, xerr=[[lo], [hi]], fmt="none",
                  color=c, capsize=4, linewidth=1.5, zorder=3)
    ax_l.scatter(e, i, color=c, marker=m, s=50, zorder=4)

    sig = "*" if c == GREEN else "n.s."
    label = f"{e:+.3f} {sig}"
    offset = hi + 0.025 if e >= 0 else -(lo + 0.025)
    ha = "left" if e >= 0 else "right"
    ax_l.text(e + (hi + 0.018 if e >= 0 else -(lo + 0.018)),
              i, label, va="center", ha=ha,
              fontsize=7.5, color=c,
              fontweight="bold" if c == GREEN else "normal")

ax_l.axvline(0, color=BLACK, linewidth=0.8, linestyle="--", alpha=0.6)
ax_l.axvspan(0, 0.55, alpha=0.04, color=GREEN)
ax_l.set_yticks(y_l)
ax_l.set_yticklabels(terms, fontsize=8)
ax_l.set_xlabel("OLS coefficient (95% CI)", labelpad=5)
ax_l.set_xlim(-0.35, 0.60)
ax_l.set_ylim(-0.7, len(terms) - 0.3)
ax_l.spines["left"].set_visible(False)
ax_l.tick_params(axis="y", length=0, pad=4)
ax_l.xaxis.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
ax_l.set_title("(a) 2×2 factorial ($N{=}120$)", fontsize=8.5, pad=6, loc="left")

# ── Right panel: multi-synthesizer robustness ────────────────────────────────
# Data from paper text (Discussion section)
synths     = ["Gemini\n2.5 Flash", "GPT-4o", "Claude\nSonnet 4"]
div_coefs  = [0.170, 0.176, 0.234]
div_lo     = [0.170 - 0.044, 0.176 - 0.049, 0.234 - 0.106]
div_hi     = [0.298 - 0.170, 0.303 - 0.176, 0.355 - 0.234]

y_r = np.arange(len(synths))

for i, (e, lo, hi) in enumerate(zip(div_coefs, div_lo, div_hi)):
    ax_r.errorbar(e, i, xerr=[[lo], [hi]], fmt="none",
                  color=GREEN, capsize=4, linewidth=1.5, zorder=3)
    ax_r.scatter(e, i, color=GREEN, marker="D", s=50, zorder=4)
    ax_r.text(e + hi + 0.015, i, f"+{e:.3f}*",
              va="center", ha="left", fontsize=7.5, color=GREEN, fontweight="bold")

# shaded region where all three CIs sit
ax_r.axvspan(0.044, 0.355, alpha=0.08, color=GREEN, zorder=0)
ax_r.axvline(0, color=BLACK, linewidth=0.8, linestyle="--", alpha=0.6)
ax_r.set_yticks(y_r)
ax_r.set_yticklabels(synths, fontsize=8)
ax_r.set_xlabel("Diversity coefficient (95% CI)", labelpad=5)
ax_r.set_xlim(-0.15, 0.55)
ax_r.set_ylim(-0.7, len(synths) - 0.3)
ax_r.spines["left"].set_visible(False)
ax_r.tick_params(axis="y", length=0, pad=4)
ax_r.xaxis.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
ax_r.set_title("(b) Diversity effect by synthesizer", fontsize=8.5, pad=6, loc="left")

fig2.tight_layout(pad=0.8, w_pad=2.0)

for ext in ("pdf", "png"):
    path = f"figures/paper_fig2_mechanism.{ext}"
    fig2.savefig(path, bbox_inches="tight", dpi=300)
    print(f"saved {path}")

plt.close(fig2)

print("\nDone. Use in LaTeX:")
print("  \\includegraphics[width=\\columnwidth]{figures/paper_fig1_mig_flip}")
print("  \\includegraphics[width=\\textwidth]{figures/paper_fig2_mechanism}")
