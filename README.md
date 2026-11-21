<div align="center">

**Accepted for Oral Presentation and Poster at the Language Models for Underserved Communities (LM4UC) Workshop · IJCAI 2026**  
Bremen, Germany · August 16, 2026

[**arXiv Paper**](https://arxiv.org/abs/2608.09046) · [**arXiv PDF**](https://arxiv.org/pdf/2608.09046) · [**Repository PDF**](Tokenization%20Equity%20Audit.pdf)

# Tokenization Equity Audit (TEA)

### Measuring the Tokenization Premium: A Cost Audit for Underserved Language Communities

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
<a href="#paper">Paper</a> •
<a href="#overview">Overview</a> •
<a href="#benchmark-summary">Benchmark Summary</a> •
<a href="#corpus">Corpus</a> •
<a href="#tokenizers-evaluated">Tokenizers</a> •
<a href="#metrics">Metrics</a> •
<a href="#quickstart">Quickstart</a> •
<a href="#repository-structure">Repository Structure</a> •
<a href="#reproducibility">Reproducibility</a> •
<a href="#citation">Citation</a> •
<a href="#license">License</a>
</p>

</div>

---

## Paper

**Measuring the Tokenization Premium: A Cost Audit for Underserved Language Communities**  
Avijit Roy, Proma Roy, Hrishitva Patel

**Language Models for Underserved Communities (LM4UC) Workshop at IJCAI 2026**  
Bremen, Germany · August 16, 2026  
**Oral Presentation and Poster**

- **arXiv:** [arXiv:2608.09046](https://arxiv.org/abs/2608.09046)
- **arXiv PDF:** [2608.09046.pdf](https://arxiv.org/pdf/2608.09046)
- **Repository PDF:** [`Tokenization Equity Audit.pdf`](Tokenization%20Equity%20Audit.pdf)

### Versions

| Version | Where | Contents |
|---|---|---|
| Paper version (LM4UC @ IJCAI 2026) | branch [`ijcai-2026`](https://github.com/HeyAvijitRoy/tea-benchmark/tree/ijcai-2026) (commit `91c5a80`) | Exactly the corpus, scripts, results and figures reported in the paper |
| Current version | `main` | The paper version plus post-publication extensions (see [Extensions](#extension-flores-200-same-text-comparison)) |

The paper's results are unchanged on `main`: the sections from [Overview](#overview) to [Metrics](#metrics), the files directly under `results/` and `figures/`, and scripts `01`–`02` are identical to the paper version. Everything under `results/extension/`, `figures/extension/`, `results/flores_same_text/`, and scripts `03`–`08` was added after publication and is not part of the paper. To cite or reproduce the paper exactly, use the `ijcai-2026` branch.

---

## Overview

Large language models process text as tokens, not as words, characters, or semantic units. When semantically equivalent content requires more tokens in one language than another, users of that language face measurable overhead: higher API cost, shorter effective context windows, and longer sequences for local inference.

The **Tokenization Equity Audit (TEA)** measures this overhead in a focused technical education setting. The benchmark evaluates three tokenizers across a **120-item Python debugging and tutoring corpus** translated from English into Bengali, Hindi, Arabic, Tamil, and Yoruba.

Bengali and Hindi are the primary validated cases in this artifact. Arabic, Tamil, and Yoruba are included as exploratory comparison languages to test whether tokenization penalties vary across scripts and language families.

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

Under open-weight tokenizers, the disparity is substantially larger. Tamil exceeds **6.5×** under both Qwen2.5 and Mistral, while Bengali and Hindi exceed **4–5×** under multiple open-weight vocabularies.

**TFR** means **Token Fertility Ratio**: the token count in a target language divided by the token count of the English version of the same item, using the same tokenizer.

A TFR of **1.56×** means the target-language version requires 56% more tokens than the English equivalent.

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

Python-specific symbols, canonical exception messages, class names, and identifiers such as `TypeError`, `IndexError`, `NoneType`, and code variables are intentionally retained in English where appropriate. This reflects common multilingual programming practice and avoids mistranslating language-specific programming symbols.

---

## Tokenizers Evaluated

| Tokenizer Label | Source | Revision |
|---|---|---|
| `tiktoken_o200k` | OpenAI `o200k_base` via `tiktoken` | Provided by pinned `tiktoken` package |
| `qwen2.5_7b` | `Qwen/Qwen2.5-7B` via Hugging Face `AutoTokenizer` | `d149729398750b98c0af14eb82c78cfe92750796` |
| `mistral_7b_v0.1` | `mistralai/Mistral-7B-v0.1` via Hugging Face `AutoTokenizer` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |

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

## Extension: FLORES-200 Same-Text Comparison

> **Post-publication extension.** This section and the subsections below were added after the LM4UC @ IJCAI 2026 paper and are not part of it. The paper version is preserved on the [`ijcai-2026`](https://github.com/HeyAvijitRoy/tea-benchmark/tree/ijcai-2026) branch.

TEA measures a technical tutoring domain. To compare it with general-domain text, `scripts/03_flores_same_text.py` measures the same-text token ratio on the 997 parallel sentences of the FLORES-200 development set (NLLB Team, 2022) for the same six languages, using OpenAI's public `r50k_base`, `cl100k_base`, and `o200k_base` encodings (and, optionally, the two open-weight tokenizers TEA pins).

For each language and tokenizer, the script reports the corpus-level ratio (total tokens / total English tokens) with a 95% bootstrap confidence interval, the mean and median per-sentence ratio (the TEA measure), the share of sentences above a threshold (default 1.5×), and the tokens-per-word ratio relative to English.

### FLORES-200 Results on `o200k_base`

| Language | Corpus ratio (95% CI) | Mean / median per-sentence | Sentences above 1.5× | Per-word ratio | TEA mean TFR |
|---|---:|---:|---:|---:|---:|
| Arabic | 1.38× (1.37–1.40) | 1.39× / 1.38× | 29% | 1.54× | 1.44× |
| Hindi | 1.58× (1.56–1.59) | 1.59× / 1.57× | 59% | 1.34× | 1.72× |
| Bengali | 1.71× (1.69–1.73) | 1.73× / 1.69× | 78% | 1.91× | 1.56× |
| Tamil | 1.98× (1.95–2.00) | 1.99× / 1.96× | 95% | 2.57× | 2.09× |
| Yoruba | 2.18× (2.14–2.22) | 2.20× / 2.16× | 85% | 1.88× | 2.37× |

Two observations follow. First, a corpus ratio below a threshold can coexist with a substantial share of sentences above it (Arabic: 1.38× overall, 29% of sentences above 1.5×), so distributions should be reported beside averages. Second, the effect of domain is not uniform in direction: Hindi is higher on TEA's technical corpus than on FLORES-200, while Bengali is lower, consistent with the suppression effect of retained English identifiers reported for technical Bengali (Section 5.3 of the paper; TEA's clean-only Bengali items run 1.79–2.11×).

Full results for all three encodings are in `results/flores_same_text/`. FLORES-200 is not redistributed here; see [`RUNBOOK.md`](RUNBOOK.md) for download instructions.

### Open-Weight Base-Model Tokenizers on FLORES-200

`scripts/05_flores_open_weight.py` repeats the FLORES-200 measurement on a pinned registry of tokenizers (`scripts/04_tokenizer_registry.py`; revisions in `results/extension/tokenizer_manifest.csv`). The Llama 3.1 tokenizer is registered but gated, and is skipped until Hugging Face access is granted. The Gemma 3 (gated), Gemma 4 and Qwen3.8 tokenizers are loaded from local copies of their files at the pinned revisions (see [`RUNBOOK.md`](RUNBOOK.md), Section 11); those files are not redistributed here.

| Tokenizer | Bengali corpus ratio (95% CI) | Bengali > 1.5× | Hindi corpus ratio (95% CI) | Hindi > 1.5× | English tokens / sentence |
|---|---:|---:|---:|---:|---:|
| `r50k_base` | 9.61× (9.49–9.72) | 100% | 7.44× (7.36–7.53) | 100% | 25.8 |
| `cl100k_base` | 5.84× (5.77–5.90) | 100% | 4.79× (4.74–4.85) | 100% | 25.9 |
| `o200k_base` | 1.71× (1.69–1.73) | 78% | 1.58× (1.56–1.59) | 59% | 25.6 |
| `o200k_harmony` | 1.71× (1.69–1.73) | 78% | 1.58× (1.56–1.59) | 59% | 25.6 |
| `qwen2.5_7b` | 5.05× (4.99–5.12) | 100% | 4.45× (4.39–4.50) | 100% | 26.3 |
| `qwen3_8b` | 5.05× (4.99–5.12) | 100% | 4.45× (4.39–4.50) | 100% | 26.3 |
| `qwen3.8_2.4t` | 3.13× (3.09–3.17) | 100% | 3.12× (3.09–3.16) | 100% | 26.2 |
| `gemma3_4b` | 1.21× (1.20–1.22) | 9% | 1.32× (1.30–1.33) | 19% | 26.0 |
| `gemma4_e2b` | 1.21× (1.20–1.22) | 9% | 1.32× (1.30–1.33) | 19% | 26.0 |
| `mistral_7b_v0.1` | 4.93× (4.87–4.99) | 100% | 4.58× (4.52–4.64) | 100% | 28.6 |
| `mistral_nemo_tekken` | 2.06× (2.03–2.08) | 95% | 1.79× (1.78–1.81) | 85% | 26.6 |
| `deepseek_v3` | 2.03× (2.00–2.05) | 95% | 2.82× (2.79–2.85) | 100% | 25.8 |
| `nemotron` | 2.06× (2.03–2.08) | 95% | 1.79× (1.78–1.81) | 85% | 26.6 |

English costs between 25.6 and 28.6 tokens per sentence under every tokenizer, so the differences in the table come from how each vocabulary handles the target language. Gemma 3 and Gemma 4 give the lowest Bengali and Hindi ratios measured (1.21× and 1.32×; also the lowest for Tamil, 1.47×). Most other tokenizers fall into two groups for Bengali and Hindi: about 1.6–2.1× (`o200k`, Tekken, Nemotron, DeepSeek-V3 for Bengali) and about 4.5–5× or more (`cl100k_base`, Qwen2.5/Qwen3, Mistral-7B). Qwen3.8, with a new 248k-token vocabulary, sits between them at about 3.1× for both languages. Hindi costs more than Bengali under DeepSeek-V3 (2.82× vs 2.03×) and Gemma 3/Gemma 4 (1.32× vs 1.21×); the two are about equal under Qwen3.8, and Bengali costs more under every other tokenizer. Four pairs produce identical counts on every sentence: `o200k_harmony` and `o200k_base`, `qwen3_8b` and `qwen2.5_7b`, `nemotron` and `mistral_nemo_tekken`, and `gemma4_e2b` and `gemma3_4b`.

Because each ratio uses the tokenizer's own English count, `scripts/08_fixed_baseline.py` also divides by a fixed English count (`o200k_base`) for every tokenizer. English costs between 1.000× and 1.116× the `o200k_base` count, so the fixed-baseline ratios are at most about 12% higher than the ratios above (largest for Mistral-7B); results are in `results/extension/fixed_baseline/` and RUNBOOK Section 14.

Results for all five languages are in `results/extension/flores_open_weight/` and `figures/extension/flores_open_weight_ratio.png`.

### Hindi Script-Share Sensitivity on the TEA Corpus

The paper's code-switching sensitivity analysis covers Bengali only. `scripts/06_hindi_script_share.py` applies the same labels and thresholds to Hindi, using the Devanagari share of alphabetic characters (letters and combining marks in U+0900–U+097F and U+A8E0–U+A8FF, against Latin letters). Labels are written to `data/extension/tea_hindi_flagged.csv`; the original corpus files are unchanged.

| Label | Hindi items | T1 | T2 | T3 | Bengali items (for comparison) |
|---|---:|---:|---:|---:|---:|
| clean (≥ 0.75) | 110 | 25 | 50 | 35 | 62 |
| mixed (0.40–0.75) | 10 | 10 | 0 | 0 | 50 |
| english_retained (< 0.40) | 0 | 0 | 0 | 0 | 8 |

| Tokenizer | Hindi mean TFR, all items (n = 120) | Clean-only (n = 110) | T1 all (n = 35) | T1 clean-only (n = 25) |
|---|---:|---:|---:|---:|
| `o200k_base` | 1.72× | 1.75× | 1.86× | 2.03× |
| `gemma3_4b` / `gemma4_e2b` | 1.42× | 1.45× | 1.55× | 1.69× |
| `mistral_nemo_tekken` / `nemotron` | 1.96× | 2.00× | 2.11× | 2.36× |
| `deepseek_v3` | 3.02× | 3.11× | 3.10× | 3.52× |
| `qwen3.8_2.4t` | 3.45× | 3.55× | 3.50× | 3.97× |
| `qwen2.5_7b` / `qwen3_8b` | 4.87× | 5.02× | 4.79× | 5.46× |
| `mistral_7b_v0.1` | 5.20× | 5.37× | 5.08× | 5.79× |

Hindi translations retain far less English than the Bengali ones: all ten mixed items are short T1 items, and none is English-retained. Excluding them raises mean Hindi TFR by 0.03 under `o200k_base` overall and by 0.17 within T1, so Hindi's published TEA figure is only slightly suppressed by retained English, unlike Bengali's.

Full results, including medians, standard deviations and every tokenizer, are in `results/extension/hindi_script_share/`.

### Technical vs General Text with Matched Aggregation

The comparison table above sets FLORES-200 corpus ratios beside TEA mean TFRs. `scripts/07_domain_comparison.py` uses the same aggregation on both sides, the mean of per-item (TEA) or per-sentence (FLORES-200) ratios, and reports the domain gap (TEA − FLORES) with a 95% bootstrap CI. Before writing results it verifies that its `o200k_base` TEA means reproduce the published values (1.56, 1.72, 1.44, 2.09, 2.37) and that its FLORES means reproduce `results/flores_same_text/summary.csv`; both checks pass.

| Tokenizer | Bengali TEA all / clean-only | Bengali FLORES | Bengali gap, all / clean-only | Hindi TEA all / clean-only | Hindi FLORES | Hindi gap, all / clean-only |
|---|---:|---:|---:|---:|---:|---:|
| `o200k_base` | 1.56× / 1.85× | 1.73× | −0.16 / +0.12 | 1.72× / 1.75× | 1.59× | +0.13 / +0.16 |
| `gemma3_4b` / `gemma4_e2b` | 1.29× / 1.45× | 1.22× | +0.07 / +0.23 | 1.42× / 1.45× | 1.33× | +0.09 / +0.12 |
| `qwen3.8_2.4t` | 2.83× / 3.44× | 3.18× | −0.35 / +0.27 | 3.45× / 3.55× | 3.16× | +0.28 / +0.38 |
| `mistral_nemo_tekken` / `nemotron` | 1.82× / 2.15× | 2.08× | −0.26 / +0.07† | 1.96× / 2.00× | 1.81× | +0.15 / +0.19 |
| `deepseek_v3` | 1.79× / 2.14× | 2.04× | −0.25 / +0.09 | 3.02× / 3.11× | 2.85× | +0.17 / +0.26 |
| `qwen2.5_7b` / `qwen3_8b` | 4.50× / 5.63× | 5.13× | −0.63 / +0.50 | 4.87× / 5.02× | 4.51× | +0.36 / +0.52 |
| `mistral_7b_v0.1` | 4.44× / 5.57× | 5.02× | −0.58 / +0.55 | 5.20× / 5.37× | 4.66× | +0.54 / +0.71 |
| `cl100k_base` | 5.18× / 6.51× | 5.91× | −0.73 / +0.60 | 5.25× / 5.43× | 4.85× | +0.40 / +0.58 |

† 95% CI includes 0; every other gap in this table has a CI that excludes 0. Clean-only n = 62 (Bengali) and 110 (Hindi) of 120.

With all items, Bengali is the only TEA language whose technical text costs fewer tokens than FLORES-200, under every tokenizer except Gemma 3 and Gemma 4 (+0.07); restricted to clean items, the gap is positive for every tokenizer, and Bengali technical text costs more, as Hindi, Tamil and Yoruba do in all cases. The lower all-item Bengali figure therefore reflects retained English in the translations rather than a property of technical text. Arabic gaps are small (+0.04 to +0.34) and their CIs include 0 for 7 of the 13 tokenizers.

Full results for all five languages are in `results/extension/domain_comparison/` and `figures/extension/domain_gap_bengali_hindi.png`. A summary of all three extension tasks, including the tokenizer manifest, is in [`results/extension/EXTENSION_SUMMARY.md`](results/extension/EXTENSION_SUMMARY.md).

---

## Repository Structure

```text
tea-benchmark/
├── data/
│   ├── tea_corpus.csv
│   ├── tea_corpus_flagged.csv
│   ├── quality_report.txt
│   └── extension/
│       └── tea_hindi_flagged.csv
├── results/
│   ├── raw_token_counts.csv
│   ├── tfr_by_item.csv
│   ├── summary_stats.csv
│   ├── acm_table.csv
│   ├── ecw_table.csv
│   ├── token_split_T1-10.csv
│   ├── flores_same_text/
│   │   ├── per_sentence_counts.csv
│   │   ├── summary.csv
│   │   └── review_table.md
│   └── extension/
│       ├── EXTENSION_SUMMARY.md
│       ├── tokenizer_manifest.csv
│       ├── flores_open_weight/
│       │   ├── per_sentence_counts.csv
│       │   ├── summary.csv
│       │   └── review_table.md
│       ├── hindi_script_share/
│       │   ├── item_level.csv
│       │   ├── label_distribution.csv
│       │   ├── summary.csv
│       │   └── review_table.md
│       ├── domain_comparison/
│       │   ├── domain_comparison.csv
│       │   └── review_table.md
│       └── fixed_baseline/
│           ├── summary.csv
│           └── review_table.md
├── figures/
│   ├── tfr_by_language_tokenizer.pdf
│   ├── tfr_by_language_tokenizer.png
│   ├── ecw_by_language.pdf
│   ├── ecw_by_language.png
│   ├── tokenizer_fragmentation_example.png
│   └── extension/
│       ├── domain_gap_bengali_hindi.pdf
│       ├── domain_gap_bengali_hindi.png
│       ├── flores_open_weight_ratio.pdf
│       └── flores_open_weight_ratio.png
├── scripts/
│   ├── 01_quality_check.py
│   ├── 02_tokenize.py
│   ├── 03_flores_same_text.py
│   ├── 04_tokenizer_registry.py
│   ├── 05_flores_open_weight.py
│   ├── 06_hindi_script_share.py
│   ├── 07_domain_comparison.py
│   ├── 08_fixed_baseline.py
│   ├── generate_tfr_figure.py
│   ├── log_run.py
│   └── token_split_table.py
├── Tokenization Equity Audit.pdf
├── README.md
├── RUNBOOK.md
├── requirements.txt
└── LICENSE
```

Generated files are included to support independent verification. They can also be regenerated from the source corpus and scripts.

Generated outputs are deterministic given the same tokenizer versions and source corpus.

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell alternatives:
# python -m venv .venv
# .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Step 1: Bengali quality check
python scripts/01_quality_check.py

# 4. Step 2: tokenization pipeline
python scripts/02_tokenize.py

# 5. Step 3: figure generation
python scripts/generate_tfr_figure.py

# 6. Optional extension: FLORES-200 same-text comparison
#    (download FLORES-200 first; see RUNBOOK.md, Section 10)
python scripts/03_flores_same_text.py --flores-dir data/flores200_dataset
```

For full reproducible procedures, output documentation, and troubleshooting, see [`RUNBOOK.md`](RUNBOOK.md).

---

## Reproducibility

This artifact is designed to support independent reproduction of the published results.

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

---

## Translation and Validation Notes

The Bengali subset was manually reviewed by two Bengali-speaking reviewers with programming experience. The Hindi subset was reviewed by a Hindi-speaking contributor familiar with technical programming terminology and instructional usage.

Arabic, Tamil, and Yoruba were generated with machine translation and manually inspected for obvious semantic or formatting failures. They are exploratory comparisons and should not be interpreted as fully validated pedagogical translations.

The benchmark’s strongest claim concerns Bengali technical tutoring content. Cross-language comparisons are intended to motivate broader tokenizer audits rather than claim final language-wide conclusions for every included language.

---

## Data and Privacy Notes

The corpus contains synthetic or benchmark-style programming education examples. It does not contain personal data, classroom logs, student submissions, names, emails, or institutional identifiers.

The paper is publicly available on **arXiv as [arXiv:2608.09046](https://arxiv.org/abs/2608.09046)**. A repository copy is also provided as [`Tokenization Equity Audit.pdf`](Tokenization%20Equity%20Audit.pdf) at the repository root.

The local camera-ready source package and LaTeX build files are excluded through `.gitignore`.

---

## Citation

If you use TEA, the benchmark data, or the accompanying code in academic work, please cite the paper:

```bibtex
@article{roy2026tokenizationpremium,
  title   = {Measuring the Tokenization Premium: A Cost Audit for Underserved Language Communities},
  author  = {Roy, Avijit and Roy, Proma and Patel, Hrishitva},
  year    = {2026},
  journal = {arXiv preprint arXiv:2608.09046},
  url     = {https://arxiv.org/abs/2608.09046}
}
```

The paper was accepted for **Oral Presentation and Poster** at the **Language Models for Underserved Communities (LM4UC) Workshop at IJCAI 2026**.

---

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

You are free to:

- Share and adapt the material for non-commercial purposes
- Give appropriate credit to the original authors

See the [`LICENSE`](LICENSE) file for full details.
