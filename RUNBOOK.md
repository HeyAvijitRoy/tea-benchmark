# TEA Benchmark — Runbook

**Project:** Tokenization Equity Audit (TEA)  
**Paper:** *Measuring the Tokenization Premium: A Cross-Script Cost Audit for Underserved Language Communities*

This runbook is the single source of truth for reproducing every pipeline step from a clean checkout through final output tables. Follow the steps in order; each script depends on the output of the previous one.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Repository Layout](#2-repository-layout)
3. [Environment Setup](#3-environment-setup)
4. [Step 1 — Bengali Quality Check (`01_quality_chec.py`)](#4-step-1--bengali-quality-check)
5. [Step 2 — Tokenization Pipeline (`02_tokenize.py`)](#5-step-2--tokenization-pipeline)
6. [Output File Reference](#6-output-file-reference)
7. [Interpreting the Results](#7-interpreting-the-results)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| Python | 3.10 + | Script uses `pathlib`, f-strings, walrus operator |
| pip | any recent | Used to install dependencies |
| Internet access | — | Required on first run to download HuggingFace tokenizers (~few MB each, no model weights) |
| Disk space | ~500 MB | HuggingFace tokenizer cache |

No GPU is needed. Tokenizer-only downloads do not pull model weights.

---

## 2. Repository Layout

```
tea-benchmark/
├── data/
│   ├── tea_corpus.csv            # Raw 70-item corpus (source of truth)
│   ├── tea_corpus_flagged.csv    # Step 1
│   └── quality_report.txt        # Step 1
├── results/                      # Step 2 (created automatically)
│   ├── raw_token_counts.csv
│   ├── tfr_by_item.csv
│   ├── summary_stats.csv
│   ├── acm_table.csv
│   └── ecw_table.csv
├── scripts/
│   ├── 01_quality_chec.py        # Step 1: Bengali quality classification
│   └── 02_tokenize.py            # Step 2: Tokenization pipeline
├── requirements.txt
├── README.md
├── RUNBOOK.md                    # This file
└── Notes.md
```

**Dependency chain:**

```
tea_corpus.csv
      │
      ▼  (Step 1)
tea_corpus_flagged.csv
      │
      ▼  (Step 2)
results/*.csv
```

---

## 3. Environment Setup

All scripts resolve paths relative to their own location, so they run correctly from any working directory. A virtual environment is required — do not install into the system Python.

### Create and activate the virtual environment

```bash
# From the repository root
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)
```

### Install all dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins every dependency to the exact versions used when the benchmark was run, ensuring reproducible token counts across machines.

### Verify the environment

```bash
python -c "import tiktoken, transformers, pandas; print('OK')"
# Expected output: OK
```

### Deactivate when done

```bash
deactivate
```

> **Note:** All `python` commands in this runbook assume the venv is active. If you see `ModuleNotFoundError`, the venv is likely not activated.

### Pinned dependency set (`requirements.txt`)

```
pandas==3.0.2
tiktoken==0.12.0
transformers==5.8.0
tokenizers==0.22.2
huggingface-hub==1.14.0
numpy==2.4.4
regex==2026.5.9
safetensors==0.7.0
tqdm==4.67.3
```

---

## 4. Step 1 — Bengali Quality Check

**Script:** `scripts/01_quality_chec.py`
**Input:** `data/tea_corpus.csv`
**Outputs:** `data/tea_corpus_flagged.csv`, `data/quality_report.txt`

### What it does

For each of the 70 corpus items, the script computes the ratio of Bengali Unicode characters (U+0980–U+09FF) to total alphabetic characters in the Bengali translation column. Items are classified into three quality tiers:

| Label | Condition | Meaning |
|---|---|---|
| `clean` | bn_ratio ≥ 0.75 | Predominantly Bengali |
| `mixed` | 0.40 ≤ bn_ratio < 0.75 | Significant code-switching |
| `english_retained` | bn_ratio < 0.40 | Predominantly English/Latin |

These labels drive the `clean_only` sensitivity subset in Step 2.

### How to run

```bash
python scripts/01_quality_chec.py
```

Expected runtime: **< 5 seconds** (pure Python, no downloads).

### Expected terminal output

```
Loading corpus from data/tea_corpus.csv ...
Computing Bengali character ratios ...
Flagged corpus saved to data/tea_corpus_flagged.csv

============================================================
TEA BENCHMARK — BENGALI TRANSLATION QUALITY REPORT
============================================================

Thresholds applied:
  clean            : bn_ratio >= 0.75
  mixed            : 0.40 <= bn_ratio < 0.75
  english_retained : bn_ratio < 0.40

── Overall Distribution ──────────────────────────────
  clean               :  58 items  (82.9%)
  mixed               :  11 items  (15.7%)
  english_retained    :   1 items  ( 1.4%)
  TOTAL               :  70 items

── Ratio Statistics ─────────────────────────────────
  Mean bn_ratio  : 0.900
  Median bn_ratio: 0.955
  ...

Report saved to data/quality_report.txt
```

### Output columns added to `tea_corpus_flagged.csv`

| Column | Type | Description |
|---|---|---|
| `bn_ratio` | float | Bengali char / total alpha chars |
| `bn_quality` | string | `clean` / `mixed` / `english_retained` |
| `tier` | string | `T1` / `T2` / `T3` extracted from Key |

---

## 5. Step 2 — Tokenization Pipeline

**Script:** `scripts/02_tokenize.py`
**Input:** `data/tea_corpus_flagged.csv` (must exist — run Step 1 first)
**Outputs:** five CSV files under `results/`

### What it does

1. Loads `tea_corpus_flagged.csv` (70 items).
2. Loads three tokenizers:
   - **tiktoken `o200k_base`** — the vocabulary used by GPT-4o.
   - **`Qwen/Qwen2.5-7B`** — Alibaba's dense 7B model tokenizer (151,936-token vocabulary).
   - **`mistralai/Mistral-7B-v0.1`** — Mistral's 7B model tokenizer (32,000-token vocabulary).
3. Tokenizes all six language columns (`English`, `Bengali`, `Hindi`, `Arabic`, `Tamil`, `Yoruba`) for every item under every tokenizer. Missing/empty cells produce `NaN` rather than errors.
4. Computes **Token Fertility Ratio (TFR)** per item per language per tokenizer: `TFR = token_count_language / token_count_english`.
5. Computes English **word count** via whitespace splitting (used as denominator baseline).
6. Writes five output files, prints summary tables, and confirms row counts.

### How to run

```bash
python scripts/02_tokenize.py
```

### Expected runtime

| Phase | Time |
|---|---|
| Corpus load | < 1 s |
| tiktoken load | < 1 s |
| Qwen tokenizer download (first run) | 10–60 s (network-dependent) |
| Mistral tokenizer download (first run) | 10–60 s |
| Tokenizing 70 items × 6 languages × 3 tokenizers | 30–120 s |
| Writing output files | < 2 s |

Subsequent runs use the HuggingFace local cache and are significantly faster.

### Expected terminal output (abridged)

```
Loading corpus from data/tea_corpus_flagged.csv ...
  70 items loaded.

Loading tokenizers ...
  [tiktoken] Loaded encoding 'o200k_base'
  [HuggingFace] Loaded 'Qwen/Qwen2.5-7B'
  [HuggingFace] Loaded 'mistralai/Mistral-7B-v0.1'

Tokenizing corpus ...
  Processed T1-01
  Processed T1-02
  ...
  Processed T3-20

Computing summary statistics ...
Computing ACM table ...
Computing ECW table ...

========================================================================
TEA BENCHMARK — MEAN TOKEN FERTILITY RATIO (TFR) BY LANGUAGE & TOKENIZER
========================================================================
Language          tiktoken_o200k          qwen2.5_7b     mistral_7b_v0.1
------------------------------------------------------------------------
English                   1.0000              1.0000              1.0000
Bengali                   1.8044              5.2929              5.2406
Hindi                     1.6984              4.7238              5.0657
Arabic                    1.4112              1.6638              3.7873
Tamil                     2.0685              6.4121              6.4328
Yoruba                    2.3767              3.1328              3.2966
========================================================================

ACM TABLE (GPT-4o / tiktoken_o200k):
  Language        Mean TFR         ACM   Extra cost / 1k req (USD)
  -----------  ----------  ----------  --------------------------
  English           1.0000      1.0000                      0.0000
  Bengali           1.8044      1.8044                      0.1161
  Hindi             1.6984      1.6984                      0.1008
  Arabic            1.4112      1.4112                      0.0593
  Tamil             2.0685      2.0685                      0.1542
  Yoruba            2.3767      2.3767                      0.1986

EFFECTIVE CONTEXT WINDOW TABLE (GPT-4o, nominal=128k):
  Language        Mean TFR    ECW (tokens)   ECW % of 128k
  -----------  ----------  --------------  --------------
  English           1.0000         128,000         100.00%
  Bengali           1.8044          70,937          55.42%
  Hindi             1.6984          75,365          58.88%
  Arabic            1.4112          90,702          70.86%
  Tamil             2.0685          61,880          48.34%
  Yoruba            2.3767          53,856          42.08%

Output files written:
  results/raw_token_counts.csv               1260 rows
  results/tfr_by_item.csv                    1260 rows
  results/summary_stats.csv                   108 rows
  results/acm_table.csv                         6 rows
  results/ecw_table.csv                         6 rows
```

---

## 6. Output File Reference

All files are written to `results/` with UTF-8 encoding. Row counts assume all 70 items with no missing data.

---

### `raw_token_counts.csv` — 1,260 rows

One row per item × language × tokenizer combination.

| Column | Type | Description |
|---|---|---|
| `key` | string | Corpus item identifier (e.g., `T1-01`) |
| `tier` | string | Difficulty tier: `T1`, `T2`, or `T3` |
| `bn_quality` | string | Bengali quality label for this item |
| `language` | string | One of: `English`, `Bengali`, `Hindi`, `Arabic`, `Tamil`, `Yoruba` |
| `tokenizer` | string | `tiktoken_o200k`, `qwen2.5_7b`, or `mistral_7b_v0.1` |
| `token_count` | float | Number of tokens produced (NaN if source text missing) |
| `word_count_en` | int | Whitespace word count of the English source text |

**Row count derivation:** 70 items × 6 languages × 3 tokenizers = 1,260.

---

### `tfr_by_item.csv` — 1,260 rows

Token Fertility Ratio for each item, computed per tokenizer.

| Column | Type | Description |
|---|---|---|
| `key` | string | Corpus item identifier |
| `tier` | string | Difficulty tier |
| `bn_quality` | string | Bengali quality label |
| `language` | string | Language column |
| `tokenizer` | string | Tokenizer name |
| `tfr` | float | `token_count_language / token_count_english` for this item. Always `1.0` for English. NaN if either count is missing or English count is zero. |

**Interpretation:** A TFR of 2.0 means this language required twice as many tokens as the English equivalent for this specific item under this tokenizer.

---

### `summary_stats.csv` — 108 rows

Aggregate statistics grouped by language × tokenizer × tier, stacked for two subsets.

| Column | Type | Description |
|---|---|---|
| `subset` | string | `all` or `clean_only` |
| `language` | string | Language |
| `tokenizer` | string | Tokenizer name |
| `tier` | string | `T1`, `T2`, or `T3` |
| `mean_tfr` | float | Mean TFR across items in this group |
| `median_tfr` | float | Median TFR |
| `std_tfr` | float | Standard deviation of TFR |
| `mean_token_count` | float | Mean raw token count for this language/tokenizer/tier |
| `n_items` | int | Number of non-NaN items in group |

**Subsets:**

- `all` — all 70 items included for every language.
- `clean_only` — for Bengali, only items where `bn_quality == "clean"` are included (sensitivity check for translation quality). All other languages are unaffected and retain all 70 items.

**Row count derivation:** 2 subsets × 6 languages × 3 tokenizers × 3 tiers = 108.

---

### `acm_table.csv` — 6 rows

API Cost Multiplier table using GPT-4o pricing ($2.50 per 1M input tokens).

| Column | Type | Description |
|---|---|---|
| `language` | string | Language |
| `mean_tfr_gpt4o` | float | Mean TFR under `tiktoken_o200k` across all 70 items |
| `acm` | float | API Cost Multiplier — equal to `mean_tfr_gpt4o` by definition, since TFR is already normalized to English |
| `cost_per_1k_requests_usd` | float | Additional cost (USD) for 1,000 requests in this language vs English, assuming each request contains one item of average English token length |

**Formula:**

```
extra_tokens_per_request = mean_en_tokens × (mean_tfr - 1.0)
cost_per_1k_requests_usd = (extra_tokens_per_request / 1,000,000) × $2.50 × 1,000
```

**Interpretation:** An ACM of 2.38 for Yoruba means Yoruba API calls cost 2.38× more than equivalent English calls under GPT-4o pricing.

---

### `ecw_table.csv` — 6 rows

Effective Context Window table showing usable context after accounting for tokenization overhead.

| Column | Type | Description |
|---|---|---|
| `language` | string | Language |
| `mean_tfr_gpt4o` | float | Mean TFR under `tiktoken_o200k` |
| `ecw_tokens` | int | `floor(128,000 / mean_tfr_gpt4o)` |
| `ecw_pct_of_nominal` | float | `ecw_tokens / 128,000 × 100` |

**Nominal window:** 128,000 tokens (GPT-4o).

**Interpretation:** A Tamil user fitting a document into GPT-4o's 128k window effectively has access to only ~61,880 tokens of content capacity — 48% of what an English user receives for the same nominal price.

---

## 7. Interpreting the Results

### TFR across tokenizers

- **tiktoken (GPT-4o)** has the best multilingual coverage of the three tokenizers tested, reflecting Anthropic's and OpenAI's investment in multilingual vocabulary. Even so, all non-English languages show TFR > 1.
- **Qwen2.5-7B** and **Mistral-7B-v0.1** show much higher TFRs for Indic scripts (Bengali, Hindi, Tamil), indicating their vocabularies were built primarily around Latin-script and CJK text. Tamil hits 6.4× under both models.
- **Arabic** is a partial exception: tiktoken (1.41×) vs Mistral (3.79×), showing that GPT-4o specifically invested in Arabic vocabulary while Mistral did not.

### Tiers

- **T1** — short error messages and single-line snippets (≤ 10 English words).
- **T2** — multi-sentence explanations and fix recommendations (10–50 English words).
- **T3** — longer conceptual explanations and code blocks (50+ English words).

TFR is relatively stable across tiers for most languages, indicating the tokenization penalty is a structural property of the vocabulary, not a length artifact.

### Bengali sensitivity analysis (`clean_only`)

Eleven Bengali items (15.7%) were classified as `mixed` or `english_retained` due to code-switching (retained Python identifiers like `TypeError`, `NoneType`). The `clean_only` subset excludes these from Bengali TFR computation to check whether they inflate or deflate the overall Bengali penalty. Compare `mean_tfr` values between `subset=all` and `subset=clean_only` in `summary_stats.csv` to assess the sensitivity.

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'pandas'` (or `tiktoken`, `transformers`)

The venv is not active. Activate it and install from `requirements.txt`:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If `.venv` does not exist yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### HuggingFace download stalls or times out

The tokenizer configs are small (~5–50 MB). If downloads are slow, set a mirror or authenticate:

```bash
export HF_TOKEN=your_token_here   # optional, for higher rate limits
python scripts/02_tokenize.py
```

Alternatively, pre-download the tokenizers (venv must be active):

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', use_fast=True)"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', use_fast=True)"
```

### `FileNotFoundError: data/tea_corpus_flagged.csv`

You must run Step 1 before Step 2:

```bash
python scripts/01_quality_chec.py
python scripts/02_tokenize.py
```

### `KeyError` or unexpected column names

The source CSV `data/tea_corpus.csv` contains two trailing unnamed columns. Both scripts strip these automatically. If you add columns to the source corpus, ensure `Key`, `English`, `Bengali`, `Hindi`, `Arabic`, `Tamil`, `Yoruba` are preserved with those exact names (case-sensitive).

### Re-running from scratch

```bash
source .venv/bin/activate
rm -f data/tea_corpus_flagged.csv data/quality_report.txt
rm -rf results/
python scripts/01_quality_chec.py
python scripts/02_tokenize.py
```

---

*Last updated: 2026-05-09*
