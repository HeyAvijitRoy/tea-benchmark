# Measuring the Tokenization Premium
## A Cross-Script Cost Audit for Underserved Language Communities

**Tokenization Equity Audit (TEA) Benchmark**

---

## Overview

Large language models charge for API access by the token. Tokenizers — the components that convert raw text into token sequences — are overwhelmingly optimized for English and Latin-script languages. This creates a hidden structural tax: identical semantic content expressed in Bengali, Tamil, Yoruba, or Arabic consumes significantly more tokens than its English equivalent, directly translating into higher API costs and reduced effective context windows for users in underserved language communities.

The **TEA Benchmark** quantifies this disparity systematically. It audits three production tokenizers across six languages on a 120-item corpus of Python debugging content, computing the **Token Fertility Ratio (TFR)** — the multiplier by which a given language exceeds English token consumption — and derives downstream metrics: API Cost Multipliers (ACM) and Effective Context Windows (ECW).

---

## Key Findings

| Language | TFR (GPT-4o) | Effective Context (of 128k) | Extra cost / 1k requests |
|---|---|---|---|
| English | 1.00× | 128,000 tokens (100.0%) | $0.00 |
| Arabic | 1.44× | 89,148 tokens (69.6%) | $0.06 |
| Bengali | 1.56× | 81,967 tokens (64.0%) | $0.07 |
| Hindi | 1.72× | 74,461 tokens (58.2%) | $0.09 |
| Tamil | 2.09× | 61,121 tokens (47.8%) | $0.14 |
| Yoruba | 2.37× | 53,951 tokens (42.2%) | $0.18 |

Under open-weight models (Qwen2.5-7B, Mistral-7B-v0.1), the disparity is dramatically worse: Tamil reaches 6.57×, Hindi 5.20× under Mistral, and Bengali 4.50× under Qwen, reflecting far weaker multilingual vocabulary investment.

---

## Corpus

The TEA corpus consists of **120 items** across three difficulty tiers, drawn from Python debugging and error-explanation content — a domain where developer tools are disproportionately used in English.

| Tier | Items | Description |
|---|---|---|
| T1 | 35 | Short error messages and single-line identifiers (≤ 10 words) |
| T2 | 50 | Multi-sentence explanations and fix recommendations (10–50 words) |
| T3 | 35 | Longer conceptual explanations and code blocks (50+ words) |

Each item is provided in six languages: **English**, **Bengali**, **Hindi**, **Arabic**, **Tamil**, **Yoruba**.

---

## Tokenizers Evaluated

| Tokenizer | Model | Vocabulary size | Access |
|---|---|---|---|
| `tiktoken o200k_base` | GPT-4o | 200,019 | Public |
| `Qwen/Qwen2.5-7B` | Qwen 2.5 7B | 151,936 | HuggingFace (open) |
| `mistralai/Mistral-7B-v0.1` | Mistral 7B | 32,000 | HuggingFace (open) |

---

## Metrics

**Token Fertility Ratio (TFR)**
The ratio of token count in a target language to token count in English for the same item and tokenizer. A TFR of 2.0 means the language requires twice as many tokens.

**API Cost Multiplier (ACM)**
Numerically equal to TFR under a given tokenizer. Reflects the cost premium paid per API call for equivalent semantic content.

**Effective Context Window (ECW)**
`ECW = nominal_window / TFR`. For GPT-4o's 128,000-token window, a Tamil user effectively has access to only ~61,880 tokens of content capacity — the same nominal price buys 48% of the usable context.

---

## Repository Structure

```
tea-benchmark/
├── data/
│   ├── tea_corpus.csv            # Raw 120-item multilingual corpus
│   ├── tea_corpus_flagged.csv    # Corpus + Bengali quality labels (generated)
│   └── quality_report.txt        # Bengali quality audit report (generated)
├── results/                      # All computed outputs (generated)
│   ├── raw_token_counts.csv      # Token counts per item × language × tokenizer
│   ├── tfr_by_item.csv           # TFR per item × language × tokenizer
│   ├── summary_stats.csv         # Aggregated stats by language/tokenizer/tier
│   ├── acm_table.csv             # API Cost Multiplier table
│   └── ecw_table.csv             # Effective Context Window table
├── scripts/
│   ├── 01_quality_check.py       # Bengali translation quality classification
│   └── 02_tokenize.py            # Tokenization pipeline
├── requirements.txt              # Python dependencies
├── RUNBOOK.md                    # Full step-by-step operational guide
└── Notes.md                      # Development log
```

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)

# 2. Install all pinned dependencies
pip install -r requirements.txt

# 3. Run Bengali quality classification (generates tea_corpus_flagged.csv)
python scripts/01_quality_check.py

# 4. Run tokenization pipeline (generates all results/)
python scripts/02_tokenize.py
```

For full details on what each script does, expected output, and troubleshooting, see [RUNBOOK.md](RUNBOOK.md).

---

## Methodology Notes

### Bengali quality classification

Bengali translations in the corpus contain Python identifiers and error type names (e.g., `TypeError`, `NoneType`) that are lexically English. The quality check script classifies each item by the ratio of Bengali Unicode characters (U+0980–U+09FF) to total alphabetic characters:

- `clean` (≥ 0.75) — predominantly Bengali
- `mixed` (0.40–0.75) — significant code-switching
- `english_retained` (< 0.40) — predominantly English/Latin

In the current corpus, 62 of 120 Bengali items (51.7%) are classified `clean`, 50 (41.7%) are `mixed`, and 8 (6.7%) are `english_retained`. The 58 non-clean items are included in the `all` subset and excluded from the `clean_only` sensitivity subset in `summary_stats.csv`.

### TFR computation

TFR is computed per item (not over aggregated token counts) to avoid length-composition artifacts. Summary statistics (mean, median, std) are computed over per-item TFR values within each group.

### Cost model

API cost projections use GPT-4o input pricing of **$2.50 per 1 million tokens** (as of benchmark date). The cost-per-1k-requests figure represents the *additional* cost versus an equivalent English request, assuming one item of average English token length per request. It is not the total cost.

---

## Citation

If you use this benchmark or its methodology, please cite:

```
@misc{tea-benchmark-2026,
  title   = {Measuring the Tokenization Premium: A Cross-Script Cost Audit
             for Underserved Language Communities},
  author  = {},
  year    = {2026},
  url     = {}
}
```

---

*Benchmark date: 2026-05-09 | Platform: Jetstream2*
