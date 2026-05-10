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
