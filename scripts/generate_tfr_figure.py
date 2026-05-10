"""
generate_tfr_figure.py
----------------------
TEA Benchmark — Publication Figure Generator

Produces two publication-quality figures formatted for an IJCAI workshop paper:

  Figure 1 — Grouped bar chart of Token Fertility Ratios (TFR) across six
              languages and three tokenizers.

  Figure 2 — Bar chart of Effective Context Window (ECW) per language under
              GPT-4o tokenization, with the nominal 128k window as a reference.

Outputs
-------
    figures/tfr_by_language_tokenizer.pdf   — vector PDF for submission
    figures/tfr_by_language_tokenizer.png   — raster preview at 300 DPI
    figures/ecw_by_language.pdf             — vector PDF for submission
    figures/ecw_by_language.png             — raster preview at 300 DPI

Usage
-----
    python scripts/generate_tfr_figure.py

Requires: pandas, matplotlib
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
import pandas as pd


# ── Resolve paths relative to repo root ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

OUT_PDF = FIGURES_DIR / "tfr_by_language_tokenizer.pdf"
OUT_PNG = FIGURES_DIR / "tfr_by_language_tokenizer.png"


# ── Data ────────────────────────────────────────────────────────────────────
TFR_DATA = {
    "Language":   ["English", "Arabic", "Hindi", "Bengali", "Tamil", "Yoruba"],
    "GPT-4o":     [1.00, 1.41, 1.70, 1.80, 2.07, 2.38],
    "Qwen2.5-7B": [1.00, 1.66, 4.72, 5.29, 6.41, 3.13],
    "Mistral-7B": [1.00, 3.79, 5.07, 5.24, 6.43, 3.30],
}
df = pd.DataFrame(TFR_DATA)

TOKENIZERS  = ["GPT-4o", "Qwen2.5-7B", "Mistral-7B"]
LANGUAGES   = df["Language"].tolist()

# ── Visual design ───────────────────────────────────────────────────────────
# Colorblind-safe palette (Wong, 2011 — Nature Methods)
COLORS = {
    "GPT-4o":     "#0072B2",   # blue
    "Qwen2.5-7B": "#E69F00",   # amber
    "Mistral-7B": "#009E73",   # green
}

# Typography — use LaTeX-style fonts if available, fall back gracefully
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "pdf.fonttype":      42,   # embed fonts as TrueType in PDF
    "ps.fonttype":       42,
})

# ── Layout ──────────────────────────────────────────────────────────────────
# 7 in wide × 4 in tall — fills a figure* across both columns in a two-column
# IJCAI template (text width ≈ 6.75 in); small margin keeps it clean.
FIG_W, FIG_H = 7.0, 4.0

n_langs = len(LANGUAGES)
n_toks  = len(TOKENIZERS)
BAR_W   = 0.22     # width of each individual bar
GROUP_W = BAR_W * n_toks
offsets = np.array([i * BAR_W for i in range(n_toks)]) - GROUP_W / 2 + BAR_W / 2

x = np.arange(n_langs)

# ── Draw figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

for i, tok in enumerate(TOKENIZERS):
    values = df[tok].tolist()
    bars = ax.bar(
        x + offsets[i],
        values,
        width=BAR_W,
        color=COLORS[tok],
        label=tok,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    # Value labels above each bar
    for bar, val in zip(bars, values):
        # Skip English (all 1.00) to reduce clutter
        if val == 1.00:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#333333",
            rotation=90,
        )

# English baseline
ax.axhline(
    y=1.0,
    color="#CC3311",
    linewidth=1.2,
    linestyle="--",
    zorder=2,
    label="English baseline (TFR = 1.0)",
)

# ── Axes formatting ─────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(LANGUAGES, fontsize=8.5)
ax.set_xlim(-0.55, n_langs - 0.45)

ax.set_ylabel("Token Fertility Ratio (relative to English)", labelpad=6)
ax.set_ylim(0, 7.4)
ax.set_yticks(range(0, 8))
ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#cccccc", zorder=0)
ax.set_axisbelow(True)

# Light shading for non-English groups to aid reading
for i in range(1, n_langs):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)
ax.tick_params(axis="both", length=3, width=0.6)

# ── Legend ───────────────────────────────────────────────────────────────────
legend = ax.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
    handlelength=1.4,
    handleheight=0.9,
    borderpad=0.6,
    labelspacing=0.35,
)
legend.get_frame().set_linewidth(0.5)

# ── Caption-style title (sits above the figure as a short descriptor) ────────
ax.set_title(
    "Token Fertility Ratio by Language and Tokenizer  (TEA Benchmark, $n = 70$)",
    pad=7,
    fontsize=9.5,
    fontweight="normal",
)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.8)
fig.savefig(str(OUT_PDF), format="pdf", bbox_inches="tight")
fig.savefig(str(OUT_PNG), format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")


# ── Figure 2: Effective Context Window ──────────────────────────────────────

ECW_DATA = {
    "Language": ["English", "Arabic", "Hindi", "Bengali", "Tamil",  "Yoruba"],
    "ECW":      [128000,    90702,    75365,   70937,     61880,    53856],
}
NOMINAL     = 128_000
OUT_ECW_PDF = FIGURES_DIR / "ecw_by_language.pdf"
OUT_ECW_PNG = FIGURES_DIR / "ecw_by_language.png"

ecw_df = pd.DataFrame(ECW_DATA)
ecw_colors = [
    "#0072B2" if lang == "English" else "#E69F00"
    for lang in ecw_df["Language"]
]

fig2, ax2 = plt.subplots(figsize=(FIG_W, 3.8))

bars2 = ax2.bar(
    ecw_df["Language"],
    ecw_df["ECW"],
    color=ecw_colors,
    edgecolor="white",
    linewidth=0.4,
    zorder=3,
)

# Nominal window reference line
ax2.axhline(
    y=NOMINAL,
    color="#CC3311",
    linewidth=1.2,
    linestyle="--",
    zorder=2,
    label=f"Nominal window ({NOMINAL:,} tokens)",
)

# Value labels — token count and % of nominal
for bar, (_, row) in zip(bars2, ecw_df.iterrows()):
    pct = 100 * row["ECW"] / NOMINAL
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1800,
        f"{int(row['ECW']):,}\n({pct:.0f}%)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333333",
        linespacing=1.3,
    )

ax2.set_ylabel("Effective Context Window (tokens)", labelpad=6)
ax2.set_ylim(0, 150_000)
ax2.set_yticks(range(0, 150_001, 20_000))
ax2.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
)
ax2.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#cccccc", zorder=0)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_linewidth(0.6)
ax2.spines["bottom"].set_linewidth(0.6)
ax2.tick_params(axis="both", length=3, width=0.6)
ax2.set_title(
    "Effective Context Window Under GPT-4o Tokenization  (TEA Benchmark, nominal = 128k)",
    pad=7,
    fontsize=9.5,
    fontweight="normal",
)
ax2.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
    handlelength=1.4,
    borderpad=0.6,
).get_frame().set_linewidth(0.5)

fig2.tight_layout(pad=0.8)
fig2.savefig(str(OUT_ECW_PDF), format="pdf", bbox_inches="tight")
fig2.savefig(str(OUT_ECW_PNG), format="png", dpi=300, bbox_inches="tight")
plt.close(fig2)

print(f"Saved: {OUT_ECW_PDF}")
print(f"Saved: {OUT_ECW_PNG}")

from log_run import append_run_entry
append_run_entry(
    script="generate_tfr_figure.py",
    stats={
        "figures": 4,
        "tfr_pdf": str(OUT_PDF.name),
        "ecw_pdf": str(OUT_ECW_PDF.name),
    },
)
