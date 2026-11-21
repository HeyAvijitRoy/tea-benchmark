"""
07_domain_comparison.py
-----------------------
TEA Benchmark — Extension Task 3: Technical vs General Text, Matched Aggregation

Purpose
-------
Compares TEA's technical tutoring corpus with general-domain FLORES-200 text
using the SAME aggregation on both sides: the mean of per-item (TEA) or
per-sentence (FLORES) ratios of tokens in the language to tokens in English.
The FLORES corpus-level ratio is included for reference only.

    TEA side    per-item TFR computed exactly as 02_tokenize.py does
                (imported via 06_hindi_script_share.compute_item_tfr)
                subsets: all items for every language; additionally
                clean-only for Bengali (bn_quality == "clean", Step 1) and
                Hindi (hi_quality == "clean", Task 2)
    FLORES side results/extension/flores_open_weight/summary.csv (Task 1)

    domain gap  = TEA mean − FLORES mean, with its sign and a 95% bootstrap CI
                  (TEA items and FLORES sentences resampled independently,
                  2,000 resamples, seed 42)

Verification (run before any output is written; the script stops on failure)
    1. o200k_base TEA means equal the published TEA values to two decimals:
       Bengali 1.56, Hindi 1.72, Arabic 1.44, Tamil 2.09, Yoruba 2.37.
    2. o200k_base FLORES means equal results/flores_same_text/summary.csv.

Usage
-----
    python scripts/07_domain_comparison.py
    python scripts/07_domain_comparison.py --self-test

Outputs
-------
    results/extension/domain_comparison/domain_comparison.csv
    results/extension/domain_comparison/review_table.md
    figures/extension/domain_gap_bengali_hindi.png (+ .pdf)
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
    """Import a numbered script (e.g. 06_hindi_script_share.py) as a module."""
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hindi = load_script_module("hindi_script_share", "06_hindi_script_share.py")
registry = hindi.registry

LANGUAGES = ["Bengali", "Hindi", "Arabic", "Tamil", "Yoruba"]
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 42

CORPUS_PATH = Path("data") / "tea_corpus_flagged.csv"
HINDI_FLAGS = Path("data") / "extension" / "tea_hindi_flagged.csv"
FLORES_OPEN = Path("results") / "extension" / "flores_open_weight" / "summary.csv"
FLORES_PER_SENTENCE = Path("results") / "extension" / "flores_open_weight" / "per_sentence_counts.csv"
FLORES_PUBLISHED = Path("results") / "flores_same_text" / "summary.csv"
OUT_DIR = Path("results") / "extension" / "domain_comparison"
FIG_STEM = Path("figures") / "extension" / "domain_gap_bengali_hindi"

# Published TEA mean TFR, tiktoken_o200k (= o200k_base), README / acm_table.csv.
PUBLISHED_TEA_O200K = {"Bengali": 1.56, "Hindi": 1.72, "Arabic": 1.44,
                       "Tamil": 2.09, "Yoruba": 2.37}
FLORES_TOLERANCE = 1e-4   # both summary files are rounded to 4 decimals


# ── Per-sentence FLORES ratios (needed for the gap CI) ───────────────────────

def flores_sentence_ratios(path: Path = FLORES_PER_SENTENCE) -> pd.DataFrame:
    """Per-sentence ratio tokens_L / tokens_EN from Task 1's per-sentence counts."""
    c = pd.read_csv(path, encoding="utf-8")
    en = c[c["language"] == "English"][["sentence_id", "tokenizer", "tokens"]]
    lg = c[c["language"] != "English"]
    m = lg.merge(en, on=["sentence_id", "tokenizer"], suffixes=("", "_en"))
    m = m[m["tokens_en"] > 0]
    m["ratio"] = m["tokens"] / m["tokens_en"]
    return m[["sentence_id", "tokenizer", "language", "ratio"]]


def bootstrap_gap_ci(tea: np.ndarray, flores: np.ndarray,
                     n_resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = RANDOM_SEED) -> tuple[float, float]:
    """95% percentile CI for mean(tea) − mean(flores), resampling each side independently."""
    rng = np.random.default_rng(seed)
    t = tea[rng.integers(0, len(tea), size=(n_resamples, len(tea)))].mean(axis=1)
    f = flores[rng.integers(0, len(flores), size=(n_resamples, len(flores)))].mean(axis=1)
    d = t - f
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def gap_sign(gap: float, ci_low: float, ci_high: float) -> str:
    direction = "TEA higher" if gap > 0 else ("FLORES higher" if gap < 0 else "equal")
    if ci_low > 0 or ci_high < 0:
        return direction
    return f"{direction} (CI includes 0)"


# ── Comparison table ─────────────────────────────────────────────────────────

def build_comparison(items: pd.DataFrame, flores_summary: pd.DataFrame,
                     flores_ratios: pd.DataFrame) -> pd.DataFrame:
    """
    items: per-item TEA TFRs with columns tokenizer, language, tfr, bn_quality, hi_quality.
    One row per tokenizer x language x TEA subset.
    """
    rows = []
    for tok in items["tokenizer"].unique():
        for lang in LANGUAGES:
            fs = flores_summary[(flores_summary["tokenizer"] == tok)
                                & (flores_summary["language"] == lang)]
            if fs.empty:
                continue
            fs = fs.iloc[0]
            fr = flores_ratios[(flores_ratios["tokenizer"] == tok)
                               & (flores_ratios["language"] == lang)]["ratio"].to_numpy(float)
            base = items[(items["tokenizer"] == tok) & (items["language"] == lang)]
            subsets = [("all", base)]
            if lang == "Bengali":
                subsets.append(("clean_only", base[base["bn_quality"] == "clean"]))
            if lang == "Hindi":
                subsets.append(("clean_only", base[base["hi_quality"] == "clean"]))
            for subset, sdf in subsets:
                vals = sdf["tfr"].dropna().to_numpy(float)
                tea_mean = float(vals.mean())
                gap = tea_mean - float(fs["mean_item_ratio"])
                lo, hi = bootstrap_gap_ci(vals, fr) if len(fr) else (float("nan"),) * 2
                rows.append({
                    "tokenizer": tok, "language": lang, "tea_subset": subset,
                    "tea_n_items": len(vals),
                    "tea_mean_tfr": tea_mean,
                    "tea_median_tfr": float(np.median(vals)),
                    "flores_n_sentences": int(fs["n_sentences"]),
                    "flores_mean_ratio": float(fs["mean_item_ratio"]),
                    "flores_median_ratio": float(fs["median_item_ratio"]),
                    "flores_corpus_ratio": float(fs["corpus_ratio"]),
                    "domain_gap": gap,
                    "domain_gap_ci_low": lo,
                    "domain_gap_ci_high": hi,
                    "gap_sign": gap_sign(gap, lo, hi),
                })
    return pd.DataFrame(rows)


# ── Verification ─────────────────────────────────────────────────────────────

def verify(comparison: pd.DataFrame, published_flores: pd.DataFrame,
           published_tea: dict = PUBLISHED_TEA_O200K) -> tuple[bool, list[str]]:
    lines, ok = [], True
    o = comparison[(comparison["tokenizer"] == "o200k_base") & (comparison["tea_subset"] == "all")]
    if o.empty:
        return False, ["o200k_base did not run; cannot verify against published TEA values"]

    lines.append("Check 1: TEA o200k_base means vs published TEA values (two decimals)")
    for lang, expected in published_tea.items():
        r = o[o["language"] == lang]
        ours = float(r["tea_mean_tfr"].iloc[0]) if not r.empty else float("nan")
        match = (not math.isnan(ours)) and round(ours, 2) == expected
        ok &= match
        lines.append(f"  {lang:<8} ours {ours:.4f} -> {ours:.2f}  published {expected:.2f}  "
                     f"{'MATCH' if match else 'MISMATCH'}")

    lines.append("Check 2: FLORES o200k_base means vs results/flores_same_text/summary.csv")
    pub = published_flores[published_flores["tokenizer"] == "o200k_base"].set_index("language")
    for lang in LANGUAGES:
        r = o[o["language"] == lang]
        ours = float(r["flores_mean_ratio"].iloc[0]) if not r.empty else float("nan")
        theirs = float(pub.loc[lang, "mean_item_ratio"]) if lang in pub.index else float("nan")
        match = abs(ours - theirs) <= FLORES_TOLERANCE
        ok &= match
        lines.append(f"  {lang:<8} ours {ours:.4f}  committed {theirs:.4f}  "
                     f"{'MATCH' if match else 'MISMATCH'}")
    return ok, lines


# ── Output ───────────────────────────────────────────────────────────────────

def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    t["_tok"] = t["tokenizer"].map({l: i for i, l in enumerate(registry.ALL_LABELS)})
    t["_lang"] = t["language"].map({l: i for i, l in enumerate(LANGUAGES)})
    t["_sub"] = t["tea_subset"].map({"all": 0, "clean_only": 1})
    return t.sort_values(["_lang", "_tok", "_sub"]).drop(columns=["_tok", "_lang", "_sub"])


def write_review_table(comp: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Language | Tokenizer | TEA subset (n) | TEA mean | FLORES mean | Gap (95% CI) | Sign "
        "| FLORES corpus ratio |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for _, r in sort_rows(comp).iterrows():
        lines.append(
            f"| {r['language']} | {r['tokenizer']} | {r['tea_subset']} ({r['tea_n_items']}) "
            f"| {r['tea_mean_tfr']:.2f}× | {r['flores_mean_ratio']:.2f}× "
            f"| {r['domain_gap']:+.2f} ({r['domain_gap_ci_low']:+.2f} to {r['domain_gap_ci_high']:+.2f}) "
            f"| {r['gap_sign']} | {r['flores_corpus_ratio']:.2f}× |")
    lines += [
        "",
        "Both sides use the mean of per-item (TEA, 120 items) or per-sentence (FLORES-200 dev, "
        "997 sentences) ratios of tokens in the language to tokens in English. Gap = TEA mean − "
        "FLORES mean; its 95% CI resamples TEA items and FLORES sentences independently (2,000 "
        "resamples, seed 42). Clean-only rows exclude mixed and English-retained items "
        "(Bengali: Step 1 labels; Hindi: `data/extension/tea_hindi_flagged.csv`). The FLORES "
        "corpus ratio is given for reference.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(comp: pd.DataFrame, stem: Path) -> list[Path]:
    """
    Paired points per tokenizer for Bengali and Hindi: FLORES mean, TEA all-items
    mean and TEA clean-only mean, joined by a thin line. Log x-axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = [  # (key, label, colour, marker) — validated categorical slots 1–3
        ("flores", "FLORES-200 (general)", "#2a78d6", "o"),
        ("all", "TEA, all items (technical)", "#eb6834", "s"),
        ("clean_only", "TEA, clean-only items", "#1baf7a", "D"),
    ]
    toks = [l for l in registry.ALL_LABELS if l in set(comp["tokenizer"])]
    y = {t: i for i, t in enumerate(reversed(toks))}
    langs = ["Bengali", "Hindi"]
    sub = comp[comp["language"].isin(langs)]
    vmax = float(max(sub["tea_mean_tfr"].max(), sub["flores_mean_ratio"].max()))
    ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 15) if t <= vmax * 1.2]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.5 * len(toks) + 1.8), sharey=True, sharex=True)
    for ax, lang in zip(axes, langs):
        for tok in toks:
            rows = sub[(sub["language"] == lang) & (sub["tokenizer"] == tok)].set_index("tea_subset")
            if rows.empty:
                continue
            vals = {"flores": rows["flores_mean_ratio"].iloc[0],
                    "all": rows.loc["all", "tea_mean_tfr"],
                    "clean_only": rows.loc["clean_only", "tea_mean_tfr"]
                    if "clean_only" in rows.index else None}
            # Small vertical offsets keep near-equal values (e.g. Hindi all vs
            # clean-only) visible; each TEA point is joined to its FLORES point.
            offset = {"flores": 0.0, "all": 0.22, "clean_only": -0.22}
            for key in ("all", "clean_only"):
                if vals[key] is not None:
                    ax.plot([vals["flores"], vals[key]], [y[tok], y[tok] + offset[key]],
                            color="#c9c8c3", lw=1.2, zorder=1)
            for key, _, colour, marker in series:
                if vals[key] is not None:
                    ax.plot(vals[key], y[tok] + offset[key], marker=marker, ls="", ms=6,
                            color=colour, mec="#fcfcfb", mew=1.0, zorder=3)
        ax.set_title(lang, fontsize=10, color=ink)
        ax.set_xscale("log")
        ax.set_xlim(0.95, vmax * 1.2)
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
    fig.supxlabel("Mean per-item ratio vs English (log scale)", fontsize=9, color=ink2)
    handles = [plt.Line2D([], [], ls="", marker=m, color=c, ms=6.5, label=l)
               for _, l, c, m in series]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Technical (TEA) vs general (FLORES-200) text: Bengali and Hindi",
                 fontsize=11, color=ink, y=1.06)
    fig.tight_layout()

    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def print_bn_hi(comp: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("DOMAIN COMPARISON (mean of per-item ratios): BENGALI AND HINDI")
    print("=" * 100)
    print(f"{'Language':<9}{'Tokenizer':<21}{'Subset':<12}{'n':>4}{'TEA':>8}{'FLORES':>8}"
          f"{'Gap':>8}{'95% CI':>18}  Sign")
    print("-" * 100)
    t = sort_rows(comp[comp["language"].isin(["Bengali", "Hindi"])])
    for _, r in t.iterrows():
        ci = f"{r['domain_gap_ci_low']:+.2f} to {r['domain_gap_ci_high']:+.2f}"
        print(f"{r['language']:<9}{r['tokenizer']:<21}{r['tea_subset']:<12}{r['tea_n_items']:>4}"
              f"{r['tea_mean_tfr']:>8.3f}{r['flores_mean_ratio']:>8.3f}{r['domain_gap']:>+8.3f}"
              f"{ci:>18}  {r['gap_sign']}")
    print("=" * 100)


# ── Self-test (no network, no tokenizers) ────────────────────────────────────

def self_test():
    """
    Hand-checked synthetic case for one tokenizer and two languages:
        TEA Bengali per-item TFR 2.0 (clean), 1.5 (clean), 1.0 (mixed)
            all mean 1.5, clean-only mean 1.75
        FLORES Bengali mean 1.25  -> gaps +0.25 (all), +0.50 (clean-only)
        TEA Hindi per-item TFR 1.0, 1.2 (both clean)  mean 1.1
        FLORES Hindi mean 1.6     -> gap −0.50, FLORES higher
    """
    print("Running self-test on synthetic data ...")
    items = pd.DataFrame({
        "tokenizer": ["o200k_base"] * 5,
        "language":  ["Bengali"] * 3 + ["Hindi"] * 2,
        "tfr":       [2.0, 1.5, 1.0, 1.0, 1.2],
        "bn_quality": ["clean", "clean", "mixed", "clean", "clean"],
        "hi_quality": ["clean"] * 5,
    })
    fsum = pd.DataFrame({
        "tokenizer": ["o200k_base"] * 2, "language": ["Bengali", "Hindi"],
        "n_sentences": [4, 4], "mean_item_ratio": [1.25, 1.6],
        "median_item_ratio": [1.25, 1.6], "corpus_ratio": [1.3, 1.55],
    })
    fr = pd.DataFrame({
        "sentence_id": [1, 2, 3, 4] * 2, "tokenizer": ["o200k_base"] * 8,
        "language": ["Bengali"] * 4 + ["Hindi"] * 4,
        "ratio": [1.0, 1.5, 1.0, 1.5, 1.6, 1.6, 1.6, 1.6],
    })
    comp = build_comparison(items, fsum, fr)
    get = lambda lang, sub: comp[(comp["language"] == lang) & (comp["tea_subset"] == sub)].iloc[0]
    assert len(comp) == 4                       # Bengali all + clean-only, Hindi all + clean-only
    assert math.isclose(get("Hindi", "clean_only")["tea_mean_tfr"], 1.1)   # both Hindi items clean
    assert math.isclose(get("Bengali", "all")["tea_mean_tfr"], 1.5)
    assert math.isclose(get("Bengali", "all")["domain_gap"], 0.25)
    assert math.isclose(get("Bengali", "clean_only")["tea_mean_tfr"], 1.75)
    assert math.isclose(get("Bengali", "clean_only")["domain_gap"], 0.5)
    assert get("Bengali", "clean_only")["tea_n_items"] == 2
    h = get("Hindi", "all")
    assert math.isclose(h["domain_gap"], -0.5)
    # Hindi FLORES ratios are constant, so the CI is TEA-side only: means of {1.0, 1.2}
    # resamples lie in [1.0, 1.2], giving a gap CI inside [-0.6, -0.4]: sign is robust.
    assert -0.6 - 1e-12 <= h["domain_gap_ci_low"] <= h["domain_gap_ci_high"] <= -0.4 + 1e-12
    assert h["gap_sign"] == "FLORES higher"
    b = get("Bengali", "all")
    assert b["domain_gap_ci_low"] <= 0.25 <= b["domain_gap_ci_high"]
    assert gap_sign(0.1, -0.05, 0.2) == "TEA higher (CI includes 0)"
    # Seeded bootstrap is reproducible
    again = build_comparison(items, fsum, fr)
    assert again["domain_gap_ci_low"].tolist() == comp["domain_gap_ci_low"].tolist()

    # Verification: passes when values match, fails (with MISMATCH lines) otherwise.
    global LANGUAGES
    saved = LANGUAGES
    LANGUAGES = ["Bengali", "Hindi"]
    try:
        pub_flores = pd.DataFrame({"tokenizer": ["o200k_base"] * 2,
                                   "language": ["Bengali", "Hindi"],
                                   "mean_item_ratio": [1.25, 1.6]})
        ok, _ = verify(comp, pub_flores, published_tea={"Bengali": 1.50, "Hindi": 1.10})
        assert ok
        ok_bad, lines = verify(comp, pub_flores, published_tea={"Bengali": 1.56, "Hindi": 1.10})
        assert not ok_bad and sum("MISMATCH" in l for l in lines) == 1
        pub_bad = pub_flores.assign(mean_item_ratio=[1.25, 1.7])
        ok_bad2, lines2 = verify(comp, pub_bad, published_tea={"Bengali": 1.50, "Hindi": 1.10})
        assert not ok_bad2 and sum("MISMATCH" in l for l in lines2) == 1
    finally:
        LANGUAGES = saved
    print("Self-test passed: matched means, clean-only subsets, gaps (+0.25, +0.50, −0.50), "
          "seeded gap CIs and sign flags, and both verification checks behave as expected.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", default=None, choices=registry.ALL_LABELS,
                        help="Tokenizers to run (default: those marked 'ran' in the manifest)")
    parser.add_argument("--self-test", action="store_true",
                        help="Check the logic on synthetic data and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return None

    for p in (FLORES_OPEN, FLORES_PER_SENTENCE, HINDI_FLAGS):
        if not p.is_file():
            sys.exit(f"\nMissing {p}. Run 05_flores_open_weight.py and "
                     "06_hindi_script_share.py first.\n")

    print(f"\nLoading corpus from {CORPUS_PATH} ...")
    df = pd.read_csv(CORPUS_PATH, encoding="utf-8")
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
    hi_flags = pd.read_csv(HINDI_FLAGS, encoding="utf-8")[["Key", "hi_quality"]]

    labels = args.labels or registry.ran_labels(registry.MANIFEST_PATH)
    print(f"\nLoading tokenizers that ran in Task 1: {', '.join(labels)}")
    counters = registry.load_counters(labels)

    print("\nComputing per-item TFR on the TEA corpus ...")
    items = hindi.compute_item_tfr(df, counters, LANGUAGES)
    items = (items.merge(df[["Key", "bn_quality"]], left_on="key", right_on="Key")
                  .drop(columns="Key")
                  .merge(hi_flags, left_on="key", right_on="Key").drop(columns="Key"))

    print("Joining with FLORES-200 results ...")
    flores_summary = pd.read_csv(FLORES_OPEN, encoding="utf-8")
    comparison = build_comparison(items, flores_summary, flores_sentence_ratios())

    print("\nVERIFICATION")
    ok, lines = verify(comparison, pd.read_csv(FLORES_PUBLISHED, encoding="utf-8"))
    for line in lines:
        print(f"  {line}")
    if not ok:
        sys.exit("\nVERIFICATION FAILED: see the discrepancies above. No outputs were written.\n")
    print("  Both checks passed.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = sort_rows(comparison)
    comparison.round(4).to_csv(OUT_DIR / "domain_comparison.csv", index=False, encoding="utf-8")
    write_review_table(comparison, OUT_DIR / "review_table.md")
    fig_paths = make_figure(comparison, FIG_STEM)

    print_bn_hi(comparison)
    print("\nOutput files written:")
    for p in [OUT_DIR / "domain_comparison.csv", OUT_DIR / "review_table.md", *fig_paths]:
        print(f"  {p}")

    return {"tokenizers": len(counters), "rows": len(comparison), "verification": "passed"}


if __name__ == "__main__":
    registry.configure_console()
    stats = main()
    if stats is not None:
        from log_run import append_run_entry
        append_run_entry(
            script="07_domain_comparison.py",
            note="TEA vs FLORES-200 domain comparison with matched aggregation.",
            stats=stats,
        )
