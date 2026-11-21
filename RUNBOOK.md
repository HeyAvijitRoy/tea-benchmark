# TEA Benchmark Runbook

This runbook explains how to reproduce the Tokenization Equity Audit (TEA) from a clean checkout. The artifact measures tokenization overhead across languages for a 120-item Python debugging and tutoring corpus.

The runbook focuses on the commands, inputs, and outputs needed to reproduce the published benchmark results.

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

For the current repository tree, see the canonical structure section in `README.md`.

---

## 2. Environment Setup

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

## 3. Step 1 — Bengali Quality Check

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

## 4. Step 2 — Tokenization Pipeline

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

| Internal Label | Source | Pinned revision |
|---|---|---|
| `tiktoken_o200k` | `o200k_base` through `tiktoken` | Provided by pinned `tiktoken` package |
| `qwen2.5_7b` | `Qwen/Qwen2.5-7B` through Hugging Face `AutoTokenizer` | `d149729398750b98c0af14eb82c78cfe92750796` |
| `mistral_7b_v0.1` | `mistralai/Mistral-7B-v0.1` through Hugging Face `AutoTokenizer` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |

The script tokenizes all items across all six languages and all three tokenizers.

Expected tokenizer rows:

```text
120 items × 6 languages × 3 tokenizers = 2,160 rows
```

---

## 5. Step 3 — Figure Generation

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

## 6. Step 4 — Tokenizer Example Figure

**Script:** `scripts/token_split_table.py`
**Input:** `data/tea_corpus.csv` and an interactively selected item key
**Outputs:**

- `results/token_split_<key>.csv`
- `figures/tokenizer_fragmentation_example.png`

Run:

```bash
python scripts/token_split_table.py
```

The committed example uses item `T1-10`. The figure plots the exact token counts written to the corresponding CSV and can be regenerated without model inference.

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

---

## 10. Extension — FLORES-200 Same-Text Ratios

This optional step compares TEA's technical-domain results with general-domain text. It does not change any of the published TEA outputs.

### Download FLORES-200

FLORES-200 is distributed separately under its own license and is git-ignored in this repository.

```bash
# Linux / macOS
curl -L -o data/flores200_dataset.tar.gz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar -xzf data/flores200_dataset.tar.gz -C data
```

Windows PowerShell:

```powershell
Invoke-WebRequest https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz -OutFile data\flores200_dataset.tar.gz
tar -xzf data\flores200_dataset.tar.gz -C data
```

### Run

```bash
# Check the statistics code on synthetic data (no network needed)
python scripts/03_flores_same_text.py --self-test

# OpenAI encodings (r50k_base, cl100k_base, o200k_base)
python scripts/03_flores_same_text.py --flores-dir data/flores200_dataset

# Also include the Qwen2.5-7B and Mistral-7B tokenizers pinned in Step 2
python scripts/03_flores_same_text.py --flores-dir data/flores200_dataset --include-hf
```

Options: `--languages` restricts the language set (English is always included), `--threshold` changes the per-sentence threshold (default 1.5), and `--reference-csv` points to an optional CSV of externally reported ratios (`language,tokenizer,reference`) for a reproduction check. The default reference file, `data/reference_ratios.local.csv`, is git-ignored.

### Outputs (`results/flores_same_text/`)

| File | Contents |
|---|---|
| `per_sentence_counts.csv` | Token and whitespace-word counts per sentence, language, and tokenizer |
| `summary.csv` | Corpus ratio with 95% bootstrap CI, mean/median/SD of per-sentence ratios, share of sentences above the threshold, tokens per word, and per-word ratio |
| `review_table.md` | Compact Markdown table of the main results |
| `sanity_check.csv` | Comparison with reference values (only written when a reference CSV is supplied; git-ignored) |

The bootstrap uses 2,000 resamples of sentence pairs with a fixed seed (42), so results are deterministic for a given tokenizer version. The run reported in the README used `tiktoken` 0.12.0.

---

## 11. Extension — Open-Weight Tokenizers on FLORES-200

This step extends Section 10 to a pinned registry of current tokenizers. It needs the FLORES-200 download from Section 10 and does not change any published TEA output.

**Scripts:** `scripts/04_tokenizer_registry.py` (shared registry, imported by Sections 11–13) and `scripts/05_flores_open_weight.py`  
**Input:** `data/flores200_dataset/dev/` (997 sentences per language)

### Registered tokenizers

| Label | Source | Revision |
|---|---|---|
| `r50k_base`, `cl100k_base`, `o200k_base`, `o200k_harmony` | `tiktoken` encodings | fixed by the pinned `tiktoken` package |
| `qwen2.5_7b` | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` |
| `qwen3_8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `llama3.1_8b` | `meta-llama/Llama-3.1-8B` (gated) | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` |
| `gemma3_4b` | `google/gemma-3-4b-it` (gated) | `093f9f388b31de276ce2de164bdc2081324b9767` |
| `mistral_7b_v0.1` | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |
| `mistral_nemo_tekken` | `mistralai/Mistral-Nemo-Base-2407` | `a4477a2f977929a969745b69bbd62e03043551a5` |
| `deepseek_v3` | `deepseek-ai/DeepSeek-V3` | `e815299b0bcbac849fa540c768ef21845365c9eb` |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16` | `434456c9a6753f29d24e23c95d622aaf17111b3b` |

`o200k_base` is the encoding TEA labels `tiktoken_o200k`. Only tokenizer files are downloaded, never model weights.

### Gated tokenizers

`llama3.1_8b` and `gemma3_4b` are gated. To include them, accept each model's licence on its Hugging Face page, then log in once:

```bash
huggingface-cli login        # or: hf auth login
```

Alternatively, set `HF_TOKEN` in the environment for the session. Never commit a token; `.env` is git-ignored.

### Run

```bash
# Offline checks
python scripts/04_tokenizer_registry.py --self-test
python scripts/05_flores_open_weight.py --self-test

# Full run (every registered tokenizer)
python scripts/05_flores_open_weight.py --flores-dir data/flores200_dataset

# A subset
python scripts/05_flores_open_weight.py --labels o200k_base qwen3_8b deepseek_v3
```

Each download is retried up to 5 times with exponential backoff (5, 10, 20, 40 s). A tokenizer that still fails, or whose gated access is refused, is marked `skipped` in the manifest with the reason, and the run continues with the rest. Expected runtime: under 1 minute with a warm Hugging Face cache; a few minutes on first download.

### Outputs

| File | Contents |
|---|---|
| `results/extension/tokenizer_manifest.csv` | Label, source, revision, vocabulary size, tokenizer type, library version, status (`ran` / `skipped`), skip reason, note |
| `results/extension/flores_open_weight/summary.csv` | Per language × tokenizer: corpus ratio with 95% bootstrap CI, mean/median/SD of per-sentence ratios, share of sentences above 1.5× and 2.0×, tokens per word, per-word ratio, mean tokens per sentence (language and English) |
| `results/extension/flores_open_weight/per_sentence_counts.csv` | Token and whitespace-word counts per sentence, language, and tokenizer |
| `results/extension/flores_open_weight/review_table.md` | Compact Markdown table, sorted by tokenizer then language |
| `figures/extension/flores_open_weight_ratio.png` / `.pdf` | Corpus ratio by language and tokenizer with CI error bars and a 1.5× reference line |

The loading, counting and summary functions are imported from `03_flores_same_text.py`, so the `o200k_base`, `cl100k_base` and `r50k_base` rows reproduce `results/flores_same_text/summary.csv` exactly.

---

## 12. Extension — Hindi Script-Share Sensitivity

This step extends the Bengali clean-only analysis (Step 1) to Hindi. It reads the corpus and never modifies it.

**Script:** `scripts/06_hindi_script_share.py`  
**Inputs:** `data/tea_corpus_flagged.csv`, `results/extension/tokenizer_manifest.csv` (from Section 11)

### What the script does

1. Computes `hi_script_share` = Devanagari characters / (Devanagari characters + Latin letters). Devanagari characters are letters and combining marks (vowel signs, virama, nukta) in U+0900–U+097F and U+A8E0–U+A8FF; digits and punctuation such as the danda are ignored, as is every non-alphabetic character.
2. Assigns `hi_quality` with the thresholds and function from `01_quality_check.py`: `clean` (≥ 0.75), `mixed` (0.40–0.75), `english_retained` (< 0.40).
3. For every tokenizer marked `ran` in the manifest, computes per-item Hindi TFR exactly as `02_tokenize.py` does, and summarizes all items vs clean-only items, overall and by tier.
4. Checks its per-item TFRs against `results/tfr_by_item.csv` for the three TEA tokenizers, and reports how many labels would change if every code point in the Devanagari block were counted (the way Step 1 counts the Bengali block).

### Run

```bash
python scripts/06_hindi_script_share.py --self-test
python scripts/06_hindi_script_share.py

# Restrict tokenizers
python scripts/06_hindi_script_share.py --labels o200k_base qwen2.5_7b
```

Gated tokenizers follow the same rule as Section 11 (`huggingface-cli login` and licence acceptance); a tokenizer that cannot be loaded is skipped with a warning. Expected runtime: under 1 minute with a warm cache.

### Outputs

| File | Contents |
|---|---|
| `data/extension/tea_hindi_flagged.csv` | `Key`, `tier`, `hi_script_share`, `hi_quality` |
| `results/extension/hindi_script_share/item_level.csv` | Per item and tokenizer: script share, label, Hindi and English token counts, TFR |
| `results/extension/hindi_script_share/summary.csv` | Mean, median, SD and item count of Hindi TFR for `all` and `clean_only`, overall and per tier |
| `results/extension/hindi_script_share/label_distribution.csv` | Items per label, overall and per tier |
| `results/extension/hindi_script_share/review_table.md` | Label distribution and all-items vs clean-only table |

---

## 13. Extension — Technical vs General Text (Matched Aggregation)

This step compares the TEA corpus with FLORES-200 using the same aggregation on both sides: the mean of per-item (TEA) or per-sentence (FLORES-200) ratios of tokens in the language to tokens in English.

**Script:** `scripts/07_domain_comparison.py`  
**Inputs:** `data/tea_corpus_flagged.csv`, `data/extension/tea_hindi_flagged.csv` (Section 12), `results/extension/tokenizer_manifest.csv` and `results/extension/flores_open_weight/` (Section 11), `results/flores_same_text/summary.csv` (Section 10)

### What the script does

1. Computes per-item TFR on the 120-item TEA corpus for every tokenizer marked `ran`, exactly as `02_tokenize.py` does.
2. Aggregates TEA by mean per-item TFR for all items in every language, plus clean-only items for Bengali (`bn_quality == "clean"`) and Hindi (`hi_quality == "clean"`).
3. Joins with the Task 1 FLORES-200 mean per-sentence ratios (corpus ratio included for reference) and reports the domain gap, TEA mean − FLORES mean, with its sign and a 95% bootstrap CI (TEA items and FLORES sentences resampled independently; 2,000 resamples; seed 42).
4. Verifies, before writing any output, that the `o200k_base` TEA means round to the published TEA values (Bengali 1.56, Hindi 1.72, Arabic 1.44, Tamil 2.09, Yoruba 2.37) and that the `o200k_base` FLORES means equal `results/flores_same_text/summary.csv`. If either check fails, the script prints the discrepancy and exits without writing results.

### Run

```bash
python scripts/07_domain_comparison.py --self-test
python scripts/07_domain_comparison.py
```

Run Sections 11 and 12 first. Gated tokenizers follow the same rule as Section 11. Expected runtime: under 1 minute with a warm cache.

### Outputs

| File | Contents |
|---|---|
| `results/extension/domain_comparison/domain_comparison.csv` | Per tokenizer × language × TEA subset: TEA n, mean and median TFR; FLORES n, mean and median per-sentence ratio and corpus ratio; domain gap with 95% CI; sign of the gap (marked when the CI includes 0) |
| `results/extension/domain_comparison/review_table.md` | The same comparison as a Markdown table |
| `figures/extension/domain_gap_bengali_hindi.png` / `.pdf` | Paired points per tokenizer for Bengali and Hindi: FLORES mean, TEA all-items mean, TEA clean-only mean |
