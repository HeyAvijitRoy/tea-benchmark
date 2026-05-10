# TEA Benchmark Runbook

This runbook explains how to reproduce the Tokenization Equity Audit (TEA) from a clean checkout. The artifact measures tokenization overhead across languages for a 120-item Python debugging and tutoring corpus.

The runbook is intentionally anonymous. It does not include author names, institutions, compute allocations, local machine paths, or internal development history.

---

## 1. Requirements

| Requirement | Tested With | Notes |
|---|---|---|
| Python | 3.10+ | Python 3.11 recommended |
| pip | Recent version | Used for dependency installation |
| Internet access | First run only | Needed to download tokenizer files from Hugging Face and `tiktoken` resources |
| GPU | Not required | This benchmark uses tokenizers only |
| Disk space | < 1 GB | Hugging Face tokenizer cache may vary by system |

No model weights are downloaded or used. The benchmark only loads tokenizer files.

---

## 2. Expected Repository Layout

```text
tea-benchmark/
├── data/
│   ├── tea_corpus.csv
│   ├── tea_corpus_flagged.csv
│   └── quality_report.txt
├── results/
│   ├── raw_token_counts.csv
│   ├── tfr_by_item.csv
│   ├── summary_stats.csv
│   ├── acm_table.csv
│   └── ecw_table.csv
├── figures/
│   ├── tfr_by_language_tokenizer.pdf
│   ├── tfr_by_language_tokenizer.png
│   ├── ecw_by_language.pdf
│   └── ecw_by_language.png
├── scripts/
│   ├── 01_quality_check.py
│   ├── 02_tokenize.py
│   ├── generate_tfr_figure.py
│   └── log_run.py
├── README.md
├── RUNBOOK.md
├── REPRODUCIBILITY_NOTES.md
└── requirements.txt
```

`tea_corpus.csv` is the source file. All other data, result, and figure files can be regenerated from the scripts.

---

## 3. Environment Setup

Create a clean virtual environment from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify imports:

```bash
python -c "import pandas, tiktoken, transformers; print('environment ok')"
```

---

## 4. Step 1 — Bengali Quality Check

**Script:** `scripts/01_quality_check.py`  
**Input:** `data/tea_corpus.csv`  
**Outputs:**

- `data/tea_corpus_flagged.csv`
- `data/quality_report.txt`

Run:

```bash
python scripts/01_quality_check.py
```

### What the script does

The script computes the ratio of Bengali Unicode characters to total alphabetic characters in the Bengali column. It then assigns one of three labels:

| Label | Rule | Interpretation |
|---|---|---|
| `clean` | `bn_ratio >= 0.75` | Predominantly Bengali text |
| `mixed` | `0.40 <= bn_ratio < 0.75` | Significant code-switching |
| `english_retained` | `bn_ratio < 0.40` | Mostly English or symbolic technical text |

These labels support the clean-only Bengali sensitivity analysis reported in the paper.

---

## 5. Step 2 — Tokenization Pipeline

**Script:** `scripts/02_tokenize.py`  
**Input:** `data/tea_corpus_flagged.csv`  
**Outputs:**

- `results/raw_token_counts.csv`
- `results/tfr_by_item.csv`
- `results/summary_stats.csv`
- `results/acm_table.csv`
- `results/ecw_table.csv`

Run:

```bash
python scripts/02_tokenize.py
```

### Tokenizers loaded

| Internal Label | Source |
|---|---|
| `tiktoken_o200k` | `o200k_base` through `tiktoken` |
| `qwen2.5_7b` | `Qwen/Qwen2.5-7B` through Hugging Face `AutoTokenizer` |
| `mistral_7b_v0.1` | `mistralai/Mistral-7B-v0.1` through Hugging Face `AutoTokenizer` |

The script tokenizes all items across all six languages and all three tokenizers.

Expected tokenizer rows:

```text
120 items × 6 languages × 3 tokenizers = 2,160 rows
```

---

## 6. Step 3 — Figure Generation

**Script:** `scripts/generate_tfr_figure.py`  
**Inputs:** `results/acm_table.csv`, `results/ecw_table.csv`, and/or `results/summary_stats.csv` depending on script implementation  
**Outputs:**

- `figures/tfr_by_language_tokenizer.pdf`
- `figures/tfr_by_language_tokenizer.png`
- `figures/ecw_by_language.pdf`
- `figures/ecw_by_language.png`

Run:

```bash
python scripts/generate_tfr_figure.py
```

The PDF files are intended for LaTeX submission. PNG files are included only for convenient preview.

---

## 7. Output File Reference

### `data/tea_corpus_flagged.csv`

Adds quality-control columns to the original corpus.

| Column | Description |
|---|---|
| `bn_ratio` | Bengali-script character ratio |
| `bn_quality` | `clean`, `mixed`, or `english_retained` |
| `tier` | Corpus tier derived from item ID |

### `results/raw_token_counts.csv`

One row per item, language, and tokenizer.

| Column | Description |
|---|---|
| `key` | Corpus item ID |
| `tier` | T1, T2, or T3 |
| `bn_quality` | Bengali quality label |
| `language` | Language of the item |
| `tokenizer` | Tokenizer label |
| `token_count` | Number of tokens |
| `word_count_en` | English source word count |

### `results/tfr_by_item.csv`

Contains per-item Token Fertility Ratio values.

```text
TFR = token_count_target_language / token_count_English
```

English rows have TFR = 1.0 by definition.

### `results/summary_stats.csv`

Aggregates TFR statistics by subset, language, tokenizer, and tier.

| Column | Description |
|---|---|
| `subset` | `all` or `clean_only` |
| `language` | Language |
| `tokenizer` | Tokenizer label |
| `tier` | T1, T2, or T3 |
| `mean_tfr` | Mean Token Fertility Ratio |
| `median_tfr` | Median Token Fertility Ratio |
| `std_tfr` | Standard deviation |
| `mean_token_count` | Mean raw token count |
| `n_items` | Number of items |

### `results/acm_table.csv`

Contains GPT-4o-family tokenizer cost multipliers and illustrative cost premiums. Dollar values depend on the pricing constant used in the script and should be treated as reproducible calculations, not permanent provider pricing.

### `results/ecw_table.csv`

Contains effective context-window estimates for a nominal 128,000-token context window.

```text
ECW = 128000 / mean_TFR
```

---

## 8. Re-running From Scratch

To regenerate all outputs:

```bash
source .venv/bin/activate
rm -f data/tea_corpus_flagged.csv data/quality_report.txt
rm -rf results figures
python scripts/01_quality_check.py
python scripts/02_tokenize.py
python scripts/generate_tfr_figure.py
```

Windows PowerShell equivalent:

```powershell
.venv\Scripts\activate
Remove-Item data\tea_corpus_flagged.csv, data\quality_report.txt -ErrorAction SilentlyContinue
Remove-Item results, figures -Recurse -Force -ErrorAction SilentlyContinue
python scripts\01_quality_check.py
python scripts\02_tokenize.py
python scripts\generate_tfr_figure.py
```

---

## 9. Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and reinstall dependencies.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Hugging Face tokenizer download fails

The first run requires network access. Re-run after confirming internet access. If needed, authenticate with Hugging Face using standard local tooling. Do not commit authentication tokens to the repository.

### `FileNotFoundError: data/tea_corpus_flagged.csv`

Run Step 1 before Step 2.

```bash
python scripts/01_quality_check.py
python scripts/02_tokenize.py
```

### Unexpected column-name errors

The source corpus must include these columns exactly:

```text
Key, English, Bengali, Hindi, Arabic, Tamil, Yoruba
```

Column names are case-sensitive.
