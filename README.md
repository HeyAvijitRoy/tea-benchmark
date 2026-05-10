<div align="center">

# Tokenization Equity Audit (TEA)

### Measuring the Tokenization Premium for Underserved Language Communities

<p>
  <strong>A reproducible benchmark for auditing tokenization cost, context-window loss, and sequence-length overhead in multilingual technical tutoring content.</strong>
</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="pandas" src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000" />
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" />
</p>

<p>
  <a href="#quickstart">Quickstart</a> •
  <a href="#benchmark-summary">Benchmark Summary</a> •
  <a href="#repository-structure">Repository Structure</a> •
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#license">License</a>
</p>

</div>

---

## Overview

Large language models process text as tokens, not as words, characters, or semantic units. When semantically equivalent content requires more tokens in one language than another, users of that language face measurable overhead: higher API cost, shorter effective context windows, and longer sequences for local inference.

The **Tokenization Equity Audit (TEA)** measures this overhead in a focused technical education setting. The benchmark evaluates three tokenizers across a 120-item Python debugging and tutoring corpus translated from English into Bengali, Hindi, Arabic, Tamil, and Yoruba.

Bengali is the primary validated case in this artifact. Hindi, Arabic, Tamil, and Yoruba are included as exploratory comparison languages to test whether tokenization penalties vary across scripts and language families.

---

## Benchmark Summary

### Main GPT-4o Tokenization Results

| Language | Mean TFR | Effective Context of 128k | Extra Cost / 1k Requests |
|---|---:|---:|---:|
| English | 1.00× | 128,000 tokens (100.0%) | $0.00 |
| Arabic | 1.44× | 89,148 tokens (69.6%) | $0.06 |
| Bengali | 1.56× | 81,967 tokens (64.0%) | $0.07 |
| Hindi | 1.72× | 74,461 tokens (58.2%) | $0.09 |
| Tamil | 2.09× | 61,121 tokens (47.8%) | $0.14 |
| Yoruba | 2.37× | 53,951 tokens (42.2%) | $0.18 |

Under open-weight tokenizers, the disparity is substantially larger: Tamil exceeds 6.5× under both Qwen2.5 and Mistral, while Bengali and Hindi exceed 4–5× under multiple open-weight vocabularies.

**TFR** means **Token Fertility Ratio**: the token count in a target language divided by the token count of the English version of the same item, using the same tokenizer.

A TFR of 1.56× means the target-language version requires 56% more tokens than the English equivalent.

---

## Corpus

The TEA corpus contains **120 Python debugging and tutoring items** across three tiers.

| Tier | Items | Description |
|---|---:|---|
| T1 | 35 | Short error messages, diagnostic phrases, and identifiers |
| T2 | 50 | Short bug explanations and fix recommendations |
| T3 | 35 | Longer conceptual explanations and code examples |

Each item is represented in six languages:

- English
- Bengali
- Hindi
- Arabic
- Tamil
- Yoruba

Python-specific symbols, class names, and identifiers such as `TypeError`, `IndexError`, `NoneType`, and code variables are intentionally retained in English where appropriate. This reflects common multilingual programming practice and avoids mistranslating language-specific programming symbols.

---

## Tokenizers Evaluated

| Tokenizer Label | Source | Notes |
|---|---|---|
| `tiktoken_o200k` | OpenAI `o200k_base` via `tiktoken` | Used as the GPT-4o-family tokenizer in this audit |
| `qwen2.5_7b` | `Qwen/Qwen2.5-7B` via Hugging Face `AutoTokenizer` | Tokenizer only; no model weights required |
| `mistral_7b_v0.1` | `mistralai/Mistral-7B-v0.1` via Hugging Face `AutoTokenizer` | Tokenizer only; no model weights required |

No model inference is required to reproduce the benchmark.

---

## Metrics

### Token Fertility Ratio (TFR)

```text
TFR = token_count(target_language_item) / token_count(English_item)
```

TFR is computed per item, language, and tokenizer before aggregation. This avoids artifacts caused by simply summing all tokens across a corpus.

### API Cost Multiplier (ACM)

For token-priced APIs, ACM is numerically equivalent to mean TFR. It estimates how much more input-token cost a target-language request incurs compared with an equivalent English request.

### Effective Context Window (ECW)

```text
ECW = nominal_context_window / mean_TFR
```

ECW estimates how much semantically equivalent content fits into the same token budget. It does not claim that the model architecture changes by language.

---

## Repository Structure

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
├── requirements.txt
└── LICENSE
```

Generated files are included to support review-time verification. They can also be regenerated from the source corpus and scripts.

Generated outputs are deterministic given the same tokenizer versions and source corpus.

---
## Quickstart
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell alternative:
# .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Bengali quality classification
python scripts/01_quality_check.py

# 4. Run tokenizer audit
python scripts/02_tokenize.py

# 5. Generate paper figures
python scripts/generate_tfr_figure.py
```
For a full step-by-step guide, see [`RUNBOOK.md`](RUNBOOK.md).

## Reproducibility

This artifact is designed to support anonymous review and independent reproduction.

The pipeline performs the following steps:

1. Loads the multilingual TEA corpus.
2. Classifies Bengali items by script-ratio quality labels.
3. Loads tokenizer vocabularies without downloading model weights.
4. Computes token counts for each item, language, and tokenizer.
5. Computes per-item Token Fertility Ratio values.
6. Aggregates summary statistics.
7. Produces cost and context-window tables.
8. Generates publication-ready figures.

The benchmark does not require GPU access.

## Translation and Validation Notes

The Bengali subset was manually reviewed by Bengali-speaking reviewers with programming experience. 

Hindi, Arabic, Tamil, and Yoruba are included as exploratory comparison languages and should not be interpreted as fully validated pedagogical translations.

The benchmark’s strongest claim concerns Bengali technical tutoring content. Cross-language comparisons are intended to motivate broader tokenizer audits rather than claim final language-wide conclusions for every included language.

---

## Data and Privacy Notes

The corpus contains synthetic or benchmark-style programming education examples. It does not contain personal data, classroom logs, student submissions, names, emails, or institutional identifiers.

The repository is prepared for anonymous review. Please avoid adding author names, institutional paths, compute-allocation identifiers, personal Git history, or acknowledgments until camera-ready release.

---


## License

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC-BY-NC 4.0).

You are free to:
- Share and adapt the material for non-commercial purposes
- Give appropriate credit to the original authors

See the [LICENSE](LICENSE) file for full details.