"""
05_flores_open_weight.py
------------------------
TEA Benchmark — Extension Task 1: Open-Weight Base-Model Tokenizers on FLORES-200

Purpose
-------
Extends the FLORES-200 same-text comparison (03_flores_same_text.py) from the
three OpenAI encodings to every tokenizer in the shared registry
(04_tokenizer_registry.py): current open-weight base-model tokenizers plus the
OpenAI encodings, all pinned.

The FLORES loading, token counting and summary statistics are imported from
03_flores_same_text.py unchanged, so every number here is computed exactly as
in that script. This script adds the share of sentences above 2.0x and the
figure, and records which tokenizers ran in the tokenizer manifest.

For each language x tokenizer it reports:
    corpus ratio = sum(tokens_L) / sum(tokens_EN), with a 95% bootstrap CI
    mean / median / SD of per-sentence ratios (the TEA measure)
    share of sentences above 1.5x and above 2.0x
    tokens per word (language and English) and the per-word ratio
    mean English tokens per sentence (how efficient each tokenizer is on English)

Usage
-----
    python scripts/05_flores_open_weight.py
    python scripts/05_flores_open_weight.py --labels o200k_base qwen3_8b
    python scripts/05_flores_open_weight.py --self-test

Outputs
-------
    results/extension/tokenizer_manifest.csv
    results/extension/flores_open_weight/summary.csv
    results/extension/flores_open_weight/per_sentence_counts.csv
    results/extension/flores_open_weight/review_table.md
    figures/extension/flores_open_weight_ratio.png (+ .pdf)
"""

import argparse
import importlib.util
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve repository root (same convention as 02_tokenize.py) ─────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


def load_script_module(name: str, filename: str):
    """Import a numbered script (e.g. 03_flores_same_text.py) as a module."""
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


flores = load_script_module("flores_same_text", "03_flores_same_text.py")
registry = load_script_module("tokenizer_registry", "04_tokenizer_registry.py")

LANGUAGES = list(flores.FLORES_CODES)          # English first, then TEA's five
THRESHOLDS = (1.5, 2.0)
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 42

OUT_DIR = Path("results") / "extension" / "flores_open_weight"
FIG_DIR = Path("figures") / "extension"
FIG_STEM = "flores_open_weight_ratio"

SUMMARY_COLUMNS = [
    "tokenizer", "language", "n_sentences",
    "corpus_ratio", "corpus_ratio_ci_low", "corpus_ratio_ci_high",
    "mean_item_ratio", "median_item_ratio", "std_item_ratio",
    "share_above_1.5x", "share_above_2.0x",
    "tokens_per_word_lang", "tokens_per_word_en", "per_word_ratio",
    "mean_tokens_per_sentence_lang", "mean_tokens_per_sentence_en",
]


# ── Statistics (built on 03_flores_same_text.summarize) ─────────────────────

def build_summary(counts: pd.DataFrame, languages: list[str],
                  n_resamples: int = BOOTSTRAP_RESAMPLES,
                  seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Run 03's summarize() once per threshold and merge the threshold shares.
    All other columns (and the seeded bootstrap CI) are identical between the
    two calls, so they are taken from the first.
    """
    base = flores.summarize(counts, languages, THRESHOLDS[0], n_resamples, seed)
    for thr in THRESHOLDS[1:]:
        extra = flores.summarize(counts, languages, thr, n_resamples, seed)
        col = f"share_above_{thr}x"
        base = base.merge(extra[["tokenizer", "language", col]],
                          on=["tokenizer", "language"], how="left")
    return base[SUMMARY_COLUMNS]


# ── Output ───────────────────────────────────────────────────────────────────

def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted by tokenizer (registry order) then language (TEA order)."""
    t = df.copy()
    t["_tok"] = t["tokenizer"].map({l: i for i, l in enumerate(registry.ALL_LABELS)})
    t["_lang"] = t["language"].map({l: i for i, l in enumerate(LANGUAGES)})
    return t.sort_values(["_tok", "_lang"]).drop(columns=["_tok", "_lang"])


def write_review_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Tokenizer | Language | Corpus ratio (95% CI) | Mean / median / SD per-sentence "
        "| > 1.5× | > 2.0× | Per-word ratio | Mean EN tokens / sentence |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in sort_rows(summary).iterrows():
        lines.append(
            f"| {r['tokenizer']} | {r['language']} "
            f"| {r['corpus_ratio']:.2f}× ({r['corpus_ratio_ci_low']:.2f}–{r['corpus_ratio_ci_high']:.2f}) "
            f"| {r['mean_item_ratio']:.2f}× / {r['median_item_ratio']:.2f}× / {r['std_item_ratio']:.2f} "
            f"| {100 * r['share_above_1.5x']:.0f}% | {100 * r['share_above_2.0x']:.0f}% "
            f"| {r['per_word_ratio']:.2f}× | {r['mean_tokens_per_sentence_en']:.1f} |"
        )
    lines += [
        "",
        f"FLORES-200 dev set, {int(summary['n_sentences'].max())} parallel sentences. "
        "Corpus ratio = total tokens in the language / total English tokens, with a 95% "
        f"percentile bootstrap CI ({BOOTSTRAP_RESAMPLES} resamples of sentence pairs, seed "
        f"{RANDOM_SEED}). Per-sentence ratios are the TEA measure. Per-word ratio = tokens "
        "per whitespace word relative to English. Tokenizer revisions are listed in "
        "`results/extension/tokenizer_manifest.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(summary: pd.DataFrame, stem: Path) -> list[Path]:
    """
    One panel per language; tokenizers on the y-axis; corpus ratio with 95% CI
    error bars; dashed reference line at 1.5x. Colour and marker encode the
    tokenizer family (OpenAI tiktoken vs open-weight Hugging Face).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401  (used via matplotlib.ticker below)

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    family_style = {
        "tiktoken": {"color": "#2a78d6", "marker": "o", "label": "OpenAI (tiktoken)"},
        "hf":       {"color": "#eb6834", "marker": "s", "label": "Open-weight (Hugging Face)"},
    }

    langs = [l for l in LANGUAGES if l != "English" and l in set(summary["language"])]
    toks = [l for l in registry.ALL_LABELS if l in set(summary["tokenizer"])]
    y = {t: i for i, t in enumerate(reversed(toks))}

    xmax = float(summary["corpus_ratio_ci_high"].max())
    fig, axes = plt.subplots(1, len(langs), figsize=(2.3 * len(langs) + 1.4, 0.32 * len(toks) + 1.6),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, lang in zip(axes, langs):
        sub = summary[summary["language"] == lang]
        for _, r in sub.iterrows():
            st = family_style[registry.SPECS[r["tokenizer"]].kind]
            yy = y[r["tokenizer"]]
            ax.errorbar(r["corpus_ratio"], yy,
                        xerr=[[r["corpus_ratio"] - r["corpus_ratio_ci_low"]],
                              [r["corpus_ratio_ci_high"] - r["corpus_ratio"]]],
                        fmt=st["marker"], ms=5, color=st["color"], ecolor=st["color"],
                        elinewidth=1.5, capsize=2.5, mec="#fcfcfb", mew=0.8, zorder=3)
        ax.axvline(1.5, color=ink2, lw=1, ls="--", zorder=1)
        ax.set_title(lang, fontsize=10, color=ink)
        # Log scale: ratios are multiplicative, and r50k_base reaches ~15x on
        # Tamil, which would otherwise compress the 1-3x range.
        ax.set_xscale("log")
        ax.set_xlim(0.9, max(4.0, xmax * 1.15))
        ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 15, 20) if t <= max(4.0, xmax * 1.15)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis="x", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=8)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels(list(y.keys()), fontsize=8, color=ink)
    fig.supxlabel("Corpus ratio vs English (tokens$_L$ / tokens$_{EN}$, log scale); "
                  "error bars = 95% bootstrap CI (often narrower than the marker); "
                  "dashed line = 1.5×", fontsize=9, color=ink2)

    used = {registry.SPECS[t].kind for t in toks}
    handles = [plt.Line2D([], [], ls="", marker=family_style[k]["marker"],
                          color=family_style[k]["color"], label=family_style[k]["label"])
               for k in ("tiktoken", "hf") if k in used]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("FLORES-200 dev: same-text token ratio by language and tokenizer",
                 fontsize=11, color=ink, y=1.07)
    fig.tight_layout()

    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def print_bn_hi(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("FLORES-200: BENGALI AND HINDI BY TOKENIZER")
    print("=" * 96)
    print(f"{'Tokenizer':<21}{'Language':<9}{'Corpus':>8}{'95% CI':>14}{'Mean':>7}"
          f"{'>1.5x':>7}{'>2.0x':>7}{'Per-word':>9}{'EN tok/sent':>13}")
    print("-" * 96)
    t = sort_rows(summary[summary["language"].isin(["Bengali", "Hindi"])])
    for _, r in t.iterrows():
        ci = f"{r['corpus_ratio_ci_low']:.2f}–{r['corpus_ratio_ci_high']:.2f}"
        print(f"{r['tokenizer']:<21}{r['language']:<9}{r['corpus_ratio']:>7.2f}x{ci:>14}"
              f"{r['mean_item_ratio']:>6.2f}x{100 * r['share_above_1.5x']:>6.0f}%"
              f"{100 * r['share_above_2.0x']:>6.0f}%{r['per_word_ratio']:>8.2f}x"
              f"{r['mean_tokens_per_sentence_en']:>13.1f}")
    print("=" * 96)


# ── Self-test (no network, no tokenizers) ────────────────────────────────────

def self_test():
    """
    Synthetic corpus with a character-count tokenizer, so every value can be
    checked by hand:
        English tokens  4, 4, 8, 4    (sum 20)
        Bengali tokens  6, 4, 12, 12  (sum 34)
        per-sentence ratios 1.5, 1.0, 1.5, 3.0
    """
    print("Running self-test on synthetic data ...")
    df = pd.DataFrame({
        "sentence_id": [1, 2, 3, 4],
        "English": ["aaaa", "aaaa", "aaaaaaaa", "aaaa"],
        "Bengali": ["bbbbbb", "bbbb", "bbbbbbbbbbbb", "bbbbbbbbbbbb"],
    })
    counts = flores.count_all(df, ["English", "Bengali"], {"chars": len})
    s = build_summary(counts, ["English", "Bengali"])
    r = s.iloc[0]
    assert list(s.columns) == SUMMARY_COLUMNS
    assert math.isclose(r["corpus_ratio"], 34 / 20), r["corpus_ratio"]           # 1.70
    assert math.isclose(r["mean_item_ratio"], 7.0 / 4), r["mean_item_ratio"]     # 1.75
    assert math.isclose(r["median_item_ratio"], 1.5), r["median_item_ratio"]
    # Sample SD of (1.5, 1.0, 1.5, 3.0): squared deviations sum to 2.25, / 3 = 0.75
    assert math.isclose(r["std_item_ratio"], math.sqrt(0.75)), r["std_item_ratio"]
    assert math.isclose(r["share_above_1.5x"], 0.25), r["share_above_1.5x"]      # only 3.0
    assert math.isclose(r["share_above_2.0x"], 0.25), r["share_above_2.0x"]      # only 3.0
    assert math.isclose(r["mean_tokens_per_sentence_en"], 5.0), r["mean_tokens_per_sentence_en"]
    # One word per sentence in both languages, so per-word ratio == corpus ratio
    assert math.isclose(r["per_word_ratio"], 1.7), r["per_word_ratio"]
    assert r["corpus_ratio_ci_low"] <= r["corpus_ratio"] <= r["corpus_ratio_ci_high"]
    # Seeded bootstrap is reproducible
    s2 = build_summary(counts, ["English", "Bengali"])
    assert s2.iloc[0]["corpus_ratio_ci_low"] == r["corpus_ratio_ci_low"]
    print("Self-test passed: corpus ratio 1.70, mean 1.75, median 1.50, SD 0.866, "
          "shares above 1.5x and 2.0x 25%, mean EN tokens 5.0, per-word ratio 1.70, "
          "and a reproducible CI that contains the point estimate.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flores-dir", default="data/flores200_dataset",
                        help="Extracted FLORES-200 folder (contains dev/)")
    parser.add_argument("--labels", nargs="+", default=registry.ALL_LABELS,
                        choices=registry.ALL_LABELS,
                        help="Tokenizers to run (default: every registered tokenizer)")
    parser.add_argument("--self-test", action="store_true",
                        help="Check the statistics on synthetic data and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return None

    print("\nLoading FLORES-200 dev split ...")
    df = flores.load_flores(Path(args.flores_dir), LANGUAGES)

    print("\nLoading tokenizers ...")
    counters = registry.load_counters(args.labels)
    manifest = registry.write_manifest(registry.MANIFEST_PATH)
    registry.print_manifest(manifest)
    if not counters:
        sys.exit("\nNo tokenizer could be loaded; see the manifest for reasons.\n")

    print("\nCounting tokens ...")
    counts = flores.count_all(df, LANGUAGES, counters)

    print("\nComputing summary statistics ...")
    summary = sort_rows(build_summary(counts, LANGUAGES))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT_DIR / "per_sentence_counts.csv", index=False, encoding="utf-8")
    summary.round(4).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8")
    write_review_table(summary, OUT_DIR / "review_table.md")
    fig_paths = make_figure(summary, FIG_DIR / FIG_STEM)

    print_bn_hi(summary)
    print(f"\nOutput files written:")
    for p in [registry.MANIFEST_PATH, OUT_DIR / "summary.csv",
              OUT_DIR / "per_sentence_counts.csv", OUT_DIR / "review_table.md", *fig_paths]:
        print(f"  {p}")

    return {
        "sentences":           len(df),
        "languages":           len(LANGUAGES),
        "tokenizers_ran":      len(counters),
        "tokenizers_skipped":  len(args.labels) - len(counters),
    }


if __name__ == "__main__":
    registry.configure_console()
    stats = main()
    if stats is not None:
        from log_run import append_run_entry
        append_run_entry(
            script="05_flores_open_weight.py",
            note="FLORES-200 dev same-text ratios on the extension tokenizer registry.",
            stats=stats,
        )
