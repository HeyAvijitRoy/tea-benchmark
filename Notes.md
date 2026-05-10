# Notes

---

## 2026-05-09 — Project started on Jetstream2

- Initialized repository on Jetstream2.

---

## 2026-05-09 — Day 3: Tokenization pipeline built and executed

### What was done

**Bengali quality check (Step 1)**

Ran `scripts/01_quality_check.py` against `data/tea_corpus.csv` (70 items, 6 language columns). The script computes the ratio of Bengali Unicode characters (U+0980–U+09FF) to total alphabetic characters and classifies each item:

- `clean` (bn_ratio ≥ 0.75): 60 items — 85.7%
- `mixed` (0.40–0.75): 9 items — 12.9%
- `english_retained` (< 0.40): 1 item — 1.4% (T1-11, a purely symbolic error with no translatable text)

Output: `data/tea_corpus_flagged.csv` (70 rows × 12 columns, adds `bn_ratio`, `bn_quality`, `tier`) and `data/quality_report.txt`.

**Tokenization pipeline (Step 2)**

Wrote and ran `scripts/02_tokenize.py`. Installed dependencies: `pandas==3.0.2`, `tiktoken==0.12.0`, `transformers==5.8.0`, `tokenizers==0.22.2`, `huggingface-hub==1.14.0`.

Three tokenizers loaded:
- `tiktoken o200k_base` — GPT-4o vocabulary (200k tokens)
- `Qwen/Qwen2.5-7B` — via HuggingFace AutoTokenizer, no model weights downloaded
- `mistralai/Mistral-7B-v0.1` — via HuggingFace AutoTokenizer, no model weights downloaded

Tokenized all 70 items × 6 languages × 3 tokenizers = 1,260 combinations. No NaN values encountered (all cells populated in the source corpus).

**Results produced:**

| File | Rows | Description |
|---|---|---|
| `results/raw_token_counts.csv` | 1,260 | Raw token counts per item × language × tokenizer |
| `results/tfr_by_item.csv` | 1,260 | TFR per item × language × tokenizer |
| `results/summary_stats.csv` | 108 | Aggregated stats (all + clean_only subsets) |
| `results/acm_table.csv` | 6 | API Cost Multiplier table (GPT-4o pricing) |
| `results/ecw_table.csv` | 6 | Effective Context Window table |

**Summary of computed TFRs (mean, all items, tiktoken GPT-4o):**

| Language | Mean TFR | ECW (of 128k) |
|---|---|---|
| English | 1.0000 | 128,000 (100%) |
| Arabic | 1.4112 | 90,702 (70.9%) |
| Hindi | 1.6984 | 75,365 (58.9%) |
| Bengali | 1.8044 | 70,937 (55.4%) |
| Tamil | 2.0685 | 61,880 (48.3%) |
| Yoruba | 2.3767 | 53,856 (42.1%) |

Notable: Tamil and Yoruba show the worst GPT-4o TFRs; Tamil reaches 6.4× under both Qwen and Mistral due to near-absent Indic vocabulary in those tokenizers.

**Documentation written:**

- `RUNBOOK.md` — full operational guide: prerequisites, environment setup, step-by-step run instructions, expected terminal output, complete output file schema reference, interpretation notes, and troubleshooting section.
- `README.md` — updated with project overview, key findings table, corpus description, tokenizer inventory, metric definitions, repository layout, quickstart, methodology notes, and citation block.


## 2026-05-09 — Dataset updated and full pipeline re-run

`tea_corpus.csv` was expanded from 70 to 120 items (T1: 20→35, T2: 30→50, T3: 20→35). All three pipeline scripts were re-run from scratch and all docs updated.

**Bengali quality (120 items):**
- `clean` (bn_ratio ≥ 0.75): 62 items — 51.7%
- `mixed` (0.40–0.75): 50 items — 41.7%
- `english_retained` (< 0.40): 8 items — 6.7%
- Mean bn_ratio: 0.746 | Median: 0.855

Note: the clean fraction dropped significantly vs the original 70-item corpus (51.7% vs 85.7%), driven by the newly added T1 items which contain many Python identifiers without Bengali equivalents.

**Updated TFR (tiktoken GPT-4o, all 120 items):**

| Language | Mean TFR | ECW (of 128k) |
|---|---|---|
| English | 1.0000 | 128,000 (100.0%) |
| Arabic | 1.4358 | 89,148 (69.6%) |
| Bengali | 1.5616 | 81,967 (64.0%) |
| Hindi | 1.7190 | 74,461 (58.2%) |
| Tamil | 2.0942 | 61,121 (47.8%) |
| Yoruba | 2.3725 | 53,951 (42.2%) |

Notable shift: Bengali TFR dropped from 1.80 to 1.56 (now below Hindi), likely because the new items include more short T1 error-type names where Bengali retains Latin characters, lowering the token count relative to English.

**Open-weight model TFRs (updated):**

| Language | Qwen2.5-7B | Mistral-7B |
|---|---|---|
| Bengali | 4.50 | 4.44 |
| Hindi | 4.86 | 5.20 |
| Arabic | 1.70 | 3.86 |
| Tamil | 6.55 | 6.57 |
| Yoruba | 3.18 | 3.33 |

All figures, README, and RUNBOOK updated to reflect 120-item corpus.

---

## 2026-05-09 — `01_quality_check.py` run
- items: 120
- clean: 62
- mixed: 50
- english_retained: 8


## 2026-05-09 — `02_tokenize.py` run
- items: 120
- languages: 6
- tokenizers: 3
- result_rows: 2160


## 2026-05-09 — `generate_tfr_figure.py` run
- figures: 4
- tfr_pdf: tfr_by_language_tokenizer.pdf
- ecw_pdf: ecw_by_language.pdf


## 2026-05-09 — `generate_tfr_figure.py` run
- figures: 4
- tfr_pdf: tfr_by_language_tokenizer.pdf
- ecw_pdf: ecw_by_language.pdf


## 2026-05-10 — `generate_tfr_figure.py` run
- figures: 4
- tfr_pdf: tfr_by_language_tokenizer.pdf
- ecw_pdf: ecw_by_language.pdf


## 2026-05-10 — `generate_tfr_figure.py` run
- figures: 4
- tfr_pdf: tfr_by_language_tokenizer.pdf
- ecw_pdf: ecw_by_language.pdf
