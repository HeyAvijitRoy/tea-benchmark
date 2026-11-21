"""
03_flores_same_text.py
----------------------
TEA Benchmark — Extension: Same-Text Ratios on FLORES-200 (general-domain text)

Purpose
-------
Measures the same-text token ratio (tokens in language L / tokens in English
for the same parallel sentences) on the FLORES-200 development set, so that
TEA's technical-domain results can be compared with a general-domain baseline.

It complements TEA's technical-domain corpus with a general-domain baseline.
It reports the measure in three ways, because the aggregation choice changes
the number a fairness threshold sees:

    1. corpus_ratio      = sum(tokens_L) / sum(tokens_EN)        (corpus-level ratio)
    2. mean_item_ratio   = mean over sentences of tokens_L / tokens_EN   (TEA measure)
    3. median_item_ratio = median of the same per-sentence ratios

It also reports:
    - share of sentences whose ratio exceeds a threshold (default 1.5x)
    - tokens-per-word ratio vs English (fertility relative to English)
    - a bootstrap 95% CI for the corpus ratio

Sanity check (optional)
-----------------------
If a reference CSV is supplied (--reference-csv, default
data/reference_ratios.local.csv when that file exists), the script compares its
corpus ratios with externally reported values for the same corpus and
tokenizers. Columns: language, tokenizer, reference. The default file name is
git-ignored, so reference values taken from unpublished sources stay local.

Data
----
FLORES-200 (NLLB Team, 2022), dev split, 997 sentences per language.
Download once and extract:

    https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz

The extracted folder contains dev/eng_Latn.dev, dev/ben_Beng.dev, etc.
Pass its path with --flores-dir (default: data/flores200_dataset).

Usage
-----
    # Default: OpenAI encodings only
    python scripts/03_flores_same_text.py --flores-dir data/flores200_dataset

    # Also include the two open-weight tokenizers TEA already pins
    python scripts/03_flores_same_text.py --flores-dir data/flores200_dataset --include-hf

    # Offline check of the statistics code (no tiktoken or FLORES needed)
    python scripts/03_flores_same_text.py --self-test

Outputs (results/flores_same_text/)
-----------------------------------
    per_sentence_counts.csv   one row per sentence x language x tokenizer
    summary.csv               all aggregate measures per language x tokenizer
    sanity_check.csv          corpus ratios vs reference values (only if supplied;
                              git-ignored)
    review_table.md           a compact table ready to paste into a review comment
"""

import argparse
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

# ── Configuration ────────────────────────────────────────────────────────────

# TEA languages mapped to FLORES-200 codes. Arabic uses Modern Standard Arabic
# (arb_Arab), which is the closest match to TEA's Arabic column.
FLORES_CODES = {
    "English": "eng_Latn",
    "Bengali": "ben_Beng",
    "Hindi":   "hin_Deva",
    "Arabic":  "arb_Arab",
    "Tamil":   "tam_Taml",
    "Yoruba":  "yor_Latn",
}

# OpenAI public encodings (r50k = GPT-3 era, cl100k = GPT-4,
# o200k = GPT-4o / GPT-5 era).
TIKTOKEN_ENCODINGS = ["r50k_base", "cl100k_base", "o200k_base"]

# Same pinned revisions as 02_tokenize.py, so results stay comparable to TEA.
HF_TOKENIZERS = {
    "qwen2.5_7b": (
        "Qwen/Qwen2.5-7B",
        "d149729398750b98c0af14eb82c78cfe92750796",
    ),
    "mistral_7b_v0.1": (
        "mistralai/Mistral-7B-v0.1",
        "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    ),
}

# Optional reference values for the sanity check. Kept out of the code (and
# out of git) so that figures from unpublished sources are never committed.
DEFAULT_REFERENCE_CSV = Path("data") / "reference_ratios.local.csv"

# Tolerance for the sanity check. Small differences can come from tiktoken
# version or whitespace handling; anything larger needs investigating.
SANITY_TOLERANCE = 0.03

DEFAULT_THRESHOLD = 1.5
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 42

OUT_DIR = Path("results") / "flores_same_text"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_flores(flores_dir: Path, languages: list[str]) -> pd.DataFrame:
    """
    Load the FLORES-200 dev split for the requested languages.
    Returns a DataFrame with one row per sentence and one column per language.
    Raises a clear error if a file is missing or the files are not aligned.
    """
    dev_dir = flores_dir / "dev"
    if not dev_dir.is_dir():
        sys.exit(
            f"\nFLORES dev folder not found at: {dev_dir}\n"
            "Download https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz,\n"
            "extract it, and pass the extracted folder with --flores-dir.\n"
        )

    columns = {}
    for lang in languages:
        code = FLORES_CODES[lang]
        path = dev_dir / f"{code}.dev"
        if not path.is_file():
            sys.exit(f"\nMissing FLORES file for {lang}: {path}\n")
        # Keep each line exactly as distributed; only strip the trailing newline.
        with open(path, encoding="utf-8") as f:
            columns[lang] = [line.rstrip("\n") for line in f]

    lengths = {lang: len(lines) for lang, lines in columns.items()}
    if len(set(lengths.values())) != 1:
        sys.exit(f"\nFLORES files are not aligned (line counts differ): {lengths}\n")

    df = pd.DataFrame(columns)
    df.insert(0, "sentence_id", range(1, len(df) + 1))
    print(f"  Loaded {len(df)} parallel sentences for {len(languages)} languages")
    return df


# ── Tokenizer loaders (same pattern as 02_tokenize.py) ──────────────────────

def load_tiktoken_counters() -> dict:
    import tiktoken
    counters = {}
    for name in TIKTOKEN_ENCODINGS:
        enc = tiktoken.get_encoding(name)
        # encode_ordinary ignores special-token parsing, which is the right
        # behaviour for plain parallel text.
        counters[name] = (lambda text, e=enc: len(e.encode_ordinary(text)))
        print(f"  [tiktoken {tiktoken.__version__}] Loaded '{name}'")
    return counters


def load_hf_counters() -> dict:
    from transformers import AutoTokenizer
    counters = {}
    for short_name, (model_id, revision) in HF_TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(
            model_id, revision=revision, use_fast=True, trust_remote_code=True
        )
        counters[short_name] = (
            lambda text, t=tok: len(t.encode(text, add_special_tokens=False))
        )
        print(f"  [HuggingFace] Loaded '{model_id}' at revision {revision}")
    return counters


# ── Counting ─────────────────────────────────────────────────────────────────

def count_all(df: pd.DataFrame, languages: list[str], counters: dict) -> pd.DataFrame:
    """
    Tokenize every sentence in every language with every tokenizer.
    Returns a long DataFrame: sentence_id, language, tokenizer, tokens, words.
    'words' is the whitespace word count, used for the tokens-per-word measure.
    """
    records = []
    for tok_name, count_fn in counters.items():
        print(f"  Tokenizing with {tok_name} ...")
        for lang in languages:
            for sid, text in zip(df["sentence_id"], df[lang]):
                records.append({
                    "sentence_id": sid,
                    "language":    lang,
                    "tokenizer":   tok_name,
                    "tokens":      count_fn(text),
                    "words":       len(text.split()),
                })
    return pd.DataFrame(records)


# ── Statistics ───────────────────────────────────────────────────────────────

def bootstrap_corpus_ratio_ci(lang_tokens: np.ndarray, en_tokens: np.ndarray,
                              n_resamples: int, seed: int) -> tuple[float, float]:
    """
    95% percentile bootstrap CI for sum(lang)/sum(en), resampling sentences
    (pairs are kept together so each resample is still parallel text).
    """
    rng = np.random.default_rng(seed)
    n = len(lang_tokens)
    idx = rng.integers(0, n, size=(n_resamples, n))
    ratios = lang_tokens[idx].sum(axis=1) / en_tokens[idx].sum(axis=1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def summarize(counts: pd.DataFrame, languages: list[str], threshold: float,
              n_resamples: int, seed: int) -> pd.DataFrame:
    """
    For each tokenizer x non-English language, compute all aggregate measures.
    """
    rows = []
    for tok_name, tok_df in counts.groupby("tokenizer", sort=False):
        en = (tok_df[tok_df["language"] == "English"]
              .set_index("sentence_id")[["tokens", "words"]])
        en_tokens_per_word = en["tokens"].sum() / en["words"].sum()

        for lang in languages:
            if lang == "English":
                continue
            lg = (tok_df[tok_df["language"] == lang]
                  .set_index("sentence_id")[["tokens", "words"]])

            # Align on sentence_id and drop any sentence where English has 0 tokens
            paired = en.join(lg, lsuffix="_en", rsuffix="_lang", how="inner")
            paired = paired[paired["tokens_en"] > 0]

            item_ratio = paired["tokens_lang"] / paired["tokens_en"]
            corpus_ratio = paired["tokens_lang"].sum() / paired["tokens_en"].sum()
            ci_low, ci_high = bootstrap_corpus_ratio_ci(
                paired["tokens_lang"].to_numpy(dtype=float),
                paired["tokens_en"].to_numpy(dtype=float),
                n_resamples, seed,
            )

            lang_tokens_per_word = paired["tokens_lang"].sum() / paired["words_lang"].sum()

            rows.append({
                "tokenizer":               tok_name,
                "language":                lang,
                "n_sentences":             len(paired),
                "corpus_ratio":            corpus_ratio,
                "corpus_ratio_ci_low":     ci_low,
                "corpus_ratio_ci_high":    ci_high,
                "mean_item_ratio":         item_ratio.mean(),
                "median_item_ratio":       item_ratio.median(),
                "std_item_ratio":          item_ratio.std(),
                f"share_above_{threshold}x": (item_ratio > threshold).mean(),
                "tokens_per_word_lang":    lang_tokens_per_word,
                "tokens_per_word_en":      en_tokens_per_word,
                "per_word_ratio":          lang_tokens_per_word / en_tokens_per_word,
                "mean_tokens_per_sentence_lang": paired["tokens_lang"].mean(),
                "mean_tokens_per_sentence_en":   paired["tokens_en"].mean(),
            })
    return pd.DataFrame(rows)


def load_reference(path: Path) -> dict:
    """Read optional reference ratios: columns language, tokenizer, reference."""
    if path is None or not path.is_file():
        return {}
    ref = pd.read_csv(path, encoding="utf-8")
    return {(r["language"], r["tokenizer"]): float(r["reference"])
            for _, r in ref.iterrows()}


def sanity_check(summary: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Compare our corpus ratios with externally reported reference values."""
    rows = []
    for (lang, tok), published in reference.items():
        match = summary[(summary["language"] == lang) & (summary["tokenizer"] == tok)]
        if match.empty:
            continue
        ours = float(match["corpus_ratio"].iloc[0])
        diff = ours - published
        rows.append({
            "language":        lang,
            "tokenizer":       tok,
            "reference":       published,
            "ours":            round(ours, 3),
            "difference":      round(diff, 3),
            "within_tolerance": abs(diff) <= SANITY_TOLERANCE,
        })
    return pd.DataFrame(rows)


# ── Output ───────────────────────────────────────────────────────────────────

def write_review_table(summary: pd.DataFrame, threshold: float, path: Path):
    """
    Compact Markdown table for the review comment: o200k_base first, as the
    newest public OpenAI encoding.
    """
    share_col = f"share_above_{threshold}x"
    order = {name: i for i, name in enumerate(
        ["o200k_base", "cl100k_base", "r50k_base", "qwen2.5_7b", "mistral_7b_v0.1"])}
    t = summary.copy()
    t["_order"] = t["tokenizer"].map(order).fillna(99)
    t = t.sort_values(["_order", "language"])

    lines = [
        "| Tokenizer | Language | Corpus ratio (95% CI) | Mean / median per-sentence ratio "
        f"| Sentences above {threshold}× | Per-word ratio |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['tokenizer']} | {r['language']} "
            f"| {r['corpus_ratio']:.2f}× ({r['corpus_ratio_ci_low']:.2f}–{r['corpus_ratio_ci_high']:.2f}) "
            f"| {r['mean_item_ratio']:.2f}× / {r['median_item_ratio']:.2f}× "
            f"| {100 * r[share_col]:.0f}% "
            f"| {r['per_word_ratio']:.2f}× |"
        )
    lines.append("")
    lines.append(
        f"FLORES-200 dev set, {int(t['n_sentences'].max())} parallel sentences. "
        "Corpus ratio = total tokens in the language / total English tokens "
        "(corpus-level ratio). Per-sentence ratios are the TEA measure. "
        "Per-word ratio = tokens per whitespace word relative to English "
        "(fertility relative to English)."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: pd.DataFrame, sanity: pd.DataFrame, threshold: float):
    share_col = f"share_above_{threshold}x"
    print("\n" + "=" * 92)
    print("FLORES-200 SAME-TEXT RATIOS VS ENGLISH")
    print("=" * 92)
    print(f"{'Tokenizer':<16}{'Language':<10}{'Corpus':>9}{'95% CI':>16}"
          f"{'Mean':>8}{'Median':>8}{f'>{threshold}x':>9}{'Per-word':>10}")
    print("-" * 92)
    for _, r in summary.iterrows():
        ci = f"{r['corpus_ratio_ci_low']:.2f}–{r['corpus_ratio_ci_high']:.2f}"
        print(f"{r['tokenizer']:<16}{r['language']:<10}{r['corpus_ratio']:>8.2f}x{ci:>16}"
              f"{r['mean_item_ratio']:>7.2f}x{r['median_item_ratio']:>7.2f}x"
              f"{100 * r[share_col]:>8.0f}%{r['per_word_ratio']:>9.2f}x")
    print("=" * 92)

    if not sanity.empty:
        print("\nSANITY CHECK AGAINST REFERENCE VALUES (corpus ratio)")
        print(sanity.to_string(index=False))
        if sanity["within_tolerance"].all():
            print("  -> All reference values reproduced within "
                  f"±{SANITY_TOLERANCE}. Other languages are measured the same way.")
        else:
            print("  -> WARNING: some reference values differ by more than "
                  f"±{SANITY_TOLERANCE}. Check the tiktoken version and the FLORES "
                  "files before using the Bengali figures in the review.")


# ── Self-test (no network, no tiktoken) ──────────────────────────────────────

def self_test():
    """
    Checks the statistics on a tiny synthetic corpus where the answers are
    known by hand. Uses a character-count 'tokenizer' so no packages are needed.
    """
    print("Running self-test on synthetic data ...")
    df = pd.DataFrame({
        "sentence_id": [1, 2, 3, 4],
        "English": ["aaaa", "aaaa", "aaaaaaaa", "aaaa"],   # 4, 4, 8, 4  -> sum 20
        "Bengali": ["bbbbbb", "bbbb", "bbbbbbbbbbbb", "bbbbbbbb"],  # 6, 4, 12, 8 -> sum 30
    })
    counters = {"chars": lambda text: len(text)}
    counts = count_all(df, ["English", "Bengali"], counters)
    s = summarize(counts, ["English", "Bengali"], threshold=1.5,
                  n_resamples=500, seed=RANDOM_SEED)
    r = s.iloc[0]

    # Per-sentence ratios: 1.5, 1.0, 1.5, 2.0
    assert math.isclose(r["corpus_ratio"], 30 / 20), r["corpus_ratio"]          # 1.50
    assert math.isclose(r["mean_item_ratio"], 6.0 / 4), r["mean_item_ratio"]    # 1.50
    assert math.isclose(r["median_item_ratio"], 1.5), r["median_item_ratio"]
    assert math.isclose(r["share_above_1.5x"], 0.25), r["share_above_1.5x"]     # only 2.0 > 1.5
    assert r["corpus_ratio_ci_low"] <= r["corpus_ratio"] <= r["corpus_ratio_ci_high"]
    # One word per sentence in both languages, so per-word ratio == corpus ratio
    assert math.isclose(r["per_word_ratio"], 1.5), r["per_word_ratio"]
    print("Self-test passed: corpus, mean, median, threshold share, CI and per-word "
          "measures all match hand-computed values.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flores-dir", default="data/flores200_dataset",
                        help="Extracted FLORES-200 folder (contains dev/)")
    parser.add_argument("--languages", nargs="+", default=list(FLORES_CODES),
                        help="Languages to measure (English is always included)")
    parser.add_argument("--include-hf", action="store_true",
                        help="Also run the Qwen2.5-7B and Mistral-7B tokenizers TEA uses")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Per-sentence ratio threshold to report (default 1.5)")
    parser.add_argument("--reference-csv", default=str(DEFAULT_REFERENCE_CSV),
                        help="Optional CSV of reference ratios for the sanity check")
    parser.add_argument("--self-test", action="store_true",
                        help="Check the statistics on synthetic data and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return None

    languages = ["English"] + [l for l in args.languages if l != "English"]
    unknown = [l for l in languages if l not in FLORES_CODES]
    if unknown:
        sys.exit(f"Unknown language(s): {unknown}. Choose from {list(FLORES_CODES)}")

    print("\nLoading FLORES-200 dev split ...")
    df = load_flores(Path(args.flores_dir), languages)

    print("\nLoading tokenizers ...")
    counters = load_tiktoken_counters()
    if args.include_hf:
        counters.update(load_hf_counters())

    print("\nCounting tokens ...")
    counts = count_all(df, languages, counters)

    print("\nComputing summary statistics ...")
    summary = summarize(counts, languages, args.threshold,
                        BOOTSTRAP_RESAMPLES, RANDOM_SEED)
    sanity = sanity_check(summary, load_reference(Path(args.reference_csv)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT_DIR / "per_sentence_counts.csv", index=False, encoding="utf-8")
    summary.round(4).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8")
    if not sanity.empty:
        sanity.to_csv(OUT_DIR / "sanity_check.csv", index=False, encoding="utf-8")
    write_review_table(summary, args.threshold, OUT_DIR / "review_table.md")

    print_summary(summary, sanity, args.threshold)
    print(f"\nOutput files written to {OUT_DIR}/")
    for name in (["per_sentence_counts.csv", "summary.csv", "review_table.md"]
                 + (["sanity_check.csv"] if not sanity.empty else [])):
        print(f"  {name}")

    return {
        "sentences":  len(df),
        "languages":  len(languages),
        "tokenizers": len(counters),
    }


if __name__ == "__main__":
    stats = main()
    if stats is not None:
        from log_run import append_run_entry
        append_run_entry(
            script="03_flores_same_text.py",
            note="FLORES-200 dev same-text ratios (general-domain comparison).",
            stats=stats,
        )
