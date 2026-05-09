"""
02_tokenize.py
--------------
TEA Benchmark — Day 3: Tokenization Pipeline

For every item in tea_corpus_flagged.csv, tokenizes each of the six language
columns using three tokenizers (GPT-4o via tiktoken, Qwen2.5-7B, Mistral-7B),
computes Token Fertility Ratios, and writes five result files.

Usage
-----
    python scripts/02_tokenize.py
"""

import os
import math
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve repository root ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

LANGUAGES = ["English", "Bengali", "Hindi", "Arabic", "Tamil", "Yoruba"]
NOMINAL_WINDOW = 128_000
GPT4O_PRICE_PER_1M = 2.50   # USD, input tokens


# ── Tokenizer loaders ────────────────────────────────────────────────────────

def load_tiktoken(encoding_name: str = "o200k_base"):
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    print(f"  [tiktoken] Loaded encoding '{encoding_name}'")
    return enc


def load_hf_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    print(f"  [HuggingFace] Loaded '{model_id}'")
    return tok


# ── Token counting ───────────────────────────────────────────────────────────

def count_tokens_tiktoken(enc, text: str) -> int:
    return len(enc.encode(text))


def count_tokens_hf(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def safe_count(counter_fn, text) -> float:
    """Return NaN if text is missing/empty, else the token count."""
    if not isinstance(text, str) or not text.strip():
        return float("nan")
    try:
        return float(counter_fn(text))
    except Exception:
        return float("nan")


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    # 1. Load corpus
    corpus_path = "data/tea_corpus_flagged.csv"
    print(f"\nLoading corpus from {corpus_path} ...")
    df = pd.read_csv(corpus_path, encoding="utf-8")
    # Drop any stray unnamed columns from the source CSV
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
    print(f"  {len(df)} items loaded. Columns: {df.columns.tolist()}")

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 2. Load tokenizers
    print("\nLoading tokenizers ...")
    tiktoken_enc = load_tiktoken("o200k_base")

    print("  Downloading Qwen/Qwen2.5-7B tokenizer (first run may take a moment) ...")
    qwen_tok = load_hf_tokenizer("Qwen/Qwen2.5-7B")

    print("  Downloading mistralai/Mistral-7B-v0.1 tokenizer ...")
    mistral_tok = load_hf_tokenizer("mistralai/Mistral-7B-v0.1")

    tokenizers = {
        "tiktoken_o200k": ("tiktoken", tiktoken_enc, count_tokens_tiktoken),
        "qwen2.5_7b":     ("hf",       qwen_tok,    count_tokens_hf),
        "mistral_7b_v0.1":("hf",       mistral_tok, count_tokens_hf),
    }

    # 3 & 4. Tokenize every item × language × tokenizer; compute TFR
    print("\nTokenizing corpus ...")
    raw_records = []
    tfr_records = []

    for _, row in df.iterrows():
        key      = row["Key"]
        tier     = row["tier"]
        bn_qual  = row["bn_quality"]

        # Word count from English source (whitespace splitting)
        en_text      = row["English"] if isinstance(row["English"], str) else ""
        word_count_en = len(en_text.split())

        for tok_name, (tok_type, tok_obj, count_fn) in tokenizers.items():
            # Token counts per language for this tokenizer
            counts = {}
            for lang in LANGUAGES:
                text = row.get(lang)
                if tok_type == "tiktoken":
                    cnt = safe_count(lambda t, e=tok_obj: count_tokens_tiktoken(e, t), text)
                else:
                    cnt = safe_count(lambda t, e=tok_obj: count_tokens_hf(e, t), text)
                counts[lang] = cnt

                raw_records.append({
                    "key":          key,
                    "tier":         tier,
                    "bn_quality":   bn_qual,
                    "language":     lang,
                    "tokenizer":    tok_name,
                    "token_count":  cnt,
                    "word_count_en": word_count_en,
                })

            # TFR relative to English
            en_count = counts["English"]
            for lang in LANGUAGES:
                lang_count = counts[lang]
                if math.isnan(en_count) or math.isnan(lang_count) or en_count == 0:
                    tfr = float("nan")
                else:
                    tfr = lang_count / en_count

                tfr_records.append({
                    "key":        key,
                    "tier":       tier,
                    "bn_quality": bn_qual,
                    "language":   lang,
                    "tokenizer":  tok_name,
                    "tfr":        tfr,
                })

        print(f"  Processed {key}")

    raw_df = pd.DataFrame(raw_records)
    tfr_df = pd.DataFrame(tfr_records)

    # 5. Write raw token counts
    out_raw = results_dir / "raw_token_counts.csv"
    raw_df.to_csv(out_raw, index=False, encoding="utf-8")

    # 6. Write TFR by item
    out_tfr = results_dir / "tfr_by_item.csv"
    tfr_df.to_csv(out_tfr, index=False, encoding="utf-8")

    # 7. Summary stats — two subsets stacked
    print("\nComputing summary statistics ...")
    summary_rows = []

    def compute_group_stats(sub_df, subset_label):
        grouped = sub_df.groupby(["language", "tokenizer", "tier"])["tfr"]
        for (lang, tok, tier), grp in grouped:
            vals = grp.dropna()
            summary_rows.append({
                "subset":           subset_label,
                "language":         lang,
                "tokenizer":        tok,
                "tier":             tier,
                "mean_tfr":         vals.mean(),
                "median_tfr":       vals.median(),
                "std_tfr":          vals.std(),
                "mean_token_count": raw_df[
                    (raw_df["language"] == lang) &
                    (raw_df["tokenizer"] == tok) &
                    (raw_df["tier"] == tier)
                ]["token_count"].mean(),
                "n_items":          len(vals),
            })

    compute_group_stats(tfr_df, "all")

    # clean_only: for Bengali use bn_quality=="clean"; for others use all items
    clean_tfr_df = tfr_df.copy()
    # Mask Bengali rows where bn_quality != clean
    bn_mask = (clean_tfr_df["language"] == "Bengali") & (clean_tfr_df["bn_quality"] != "clean")
    clean_tfr_df = clean_tfr_df[~bn_mask]
    compute_group_stats(clean_tfr_df, "clean_only")

    summary_df = pd.DataFrame(summary_rows)[[
        "subset", "language", "tokenizer", "tier",
        "mean_tfr", "median_tfr", "std_tfr", "mean_token_count", "n_items"
    ]]
    out_summary = results_dir / "summary_stats.csv"
    summary_df.to_csv(out_summary, index=False, encoding="utf-8")

    # 8. ACM table — GPT-4o (tiktoken) only, all items, across all tiers
    print("Computing ACM table ...")
    gpt4o_tfr = tfr_df[tfr_df["tokenizer"] == "tiktoken_o200k"].copy()

    # Mean English token count across all items for cost baseline
    mean_en_tokens = raw_df[
        (raw_df["tokenizer"] == "tiktoken_o200k") &
        (raw_df["language"] == "English")
    ]["token_count"].mean()

    acm_rows = []
    for lang in LANGUAGES:
        lang_tfr = gpt4o_tfr[gpt4o_tfr["language"] == lang]["tfr"].dropna()
        mean_tfr = lang_tfr.mean()
        acm      = mean_tfr                          # TFR IS the multiplier vs English
        # Extra cost for 1000 requests at the language's token volume vs English
        # cost_lang_per_request = (mean_en_tokens * mean_tfr) / 1e6 * GPT4O_PRICE_PER_1M
        # cost_en_per_request   = mean_en_tokens / 1e6 * GPT4O_PRICE_PER_1M
        # cost_per_1k = (cost_lang - cost_en) * 1000
        extra_tokens_per_req = mean_en_tokens * (mean_tfr - 1.0)
        cost_per_1k = (extra_tokens_per_req / 1e6) * GPT4O_PRICE_PER_1M * 1000

        acm_rows.append({
            "language":              lang,
            "mean_tfr_gpt4o":        round(mean_tfr, 4),
            "acm":                   round(acm, 4),
            "cost_per_1k_requests_usd": round(cost_per_1k, 4),
        })

    acm_df = pd.DataFrame(acm_rows)
    out_acm = results_dir / "acm_table.csv"
    acm_df.to_csv(out_acm, index=False, encoding="utf-8")

    # 9. ECW table
    print("Computing ECW table ...")
    ecw_rows = []
    for lang in LANGUAGES:
        mean_tfr = acm_df.loc[acm_df["language"] == lang, "mean_tfr_gpt4o"].values[0]
        if mean_tfr and not math.isnan(mean_tfr) and mean_tfr > 0:
            ecw_tokens = int(NOMINAL_WINDOW / mean_tfr)
            ecw_pct    = round(100.0 * ecw_tokens / NOMINAL_WINDOW, 2)
        else:
            ecw_tokens = None
            ecw_pct    = None
        ecw_rows.append({
            "language":           lang,
            "mean_tfr_gpt4o":     mean_tfr,
            "ecw_tokens":         ecw_tokens,
            "ecw_pct_of_nominal": ecw_pct,
        })

    ecw_df = pd.DataFrame(ecw_rows)
    out_ecw = results_dir / "ecw_table.csv"
    ecw_df.to_csv(out_ecw, index=False, encoding="utf-8")

    # 10. Terminal summary
    print("\n" + "=" * 72)
    print("TEA BENCHMARK — MEAN TOKEN FERTILITY RATIO (TFR) BY LANGUAGE & TOKENIZER")
    print("=" * 72)
    pivot = (
        tfr_df.groupby(["language", "tokenizer"])["tfr"]
        .mean()
        .unstack("tokenizer")
        .reindex(LANGUAGES)
    )
    tok_cols = ["tiktoken_o200k", "qwen2.5_7b", "mistral_7b_v0.1"]
    pivot = pivot.reindex(columns=tok_cols)
    header = f"{'Language':<12}" + "".join(f"  {c:>18}" for c in tok_cols)
    print(header)
    print("-" * 72)
    for lang, row_data in pivot.iterrows():
        vals = "".join(
            f"  {v:>18.4f}" if not math.isnan(v) else f"  {'N/A':>18}"
            for v in row_data
        )
        print(f"{lang:<12}{vals}")
    print("=" * 72)

    print("\nACM TABLE (GPT-4o / tiktoken_o200k):")
    print(f"  {'Language':<12}  {'Mean TFR':>10}  {'ACM':>10}  {'Extra cost / 1k req (USD)':>26}")
    print(f"  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*26}")
    for _, row_data in acm_df.iterrows():
        print(
            f"  {row_data['language']:<12}  {row_data['mean_tfr_gpt4o']:>10.4f}"
            f"  {row_data['acm']:>10.4f}  {row_data['cost_per_1k_requests_usd']:>26.4f}"
        )

    print("\nEFFECTIVE CONTEXT WINDOW TABLE (GPT-4o, nominal=128k):")
    print(f"  {'Language':<12}  {'Mean TFR':>10}  {'ECW (tokens)':>14}  {'ECW % of 128k':>14}")
    print(f"  {'-'*11}  {'-'*10}  {'-'*14}  {'-'*14}")
    for _, row_data in ecw_df.iterrows():
        print(
            f"  {row_data['language']:<12}  {row_data['mean_tfr_gpt4o']:>10.4f}"
            f"  {row_data['ecw_tokens']:>14,}  {row_data['ecw_pct_of_nominal']:>13.2f}%"
        )

    # 11. File confirmation
    output_files = [
        (out_raw,     raw_df),
        (out_tfr,     tfr_df),
        (out_summary, summary_df),
        (out_acm,     acm_df),
        (out_ecw,     ecw_df),
    ]
    print("\nOutput files written:")
    for path, frame in output_files:
        print(f"  {str(path):<40}  {len(frame):>5} rows")


if __name__ == "__main__":
    main()
