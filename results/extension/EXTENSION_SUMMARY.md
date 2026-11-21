# TEA Extension Summary

This extension builds on the Tokenization Equity Audit (TEA; arXiv:2608.09046) without changing any published TEA output. It adds three analyses: current open-weight tokenizers on FLORES-200 (Task 1), a Hindi script-share sensitivity analysis on the TEA corpus (Task 2), and a technical-versus-general domain comparison that uses the same aggregation on both sides (Task 3). Commands and options are in `RUNBOOK.md`, Sections 11–13.

## Tokenizer manifest

Full manifest, including the complete commit SHAs and library versions: `results/extension/tokenizer_manifest.csv`.

| Label | Source | Revision | Vocab | Type | Status | Skip reason |
|---|---|---|---:|---|---|---|
| `r50k_base` | `r50k_base` | pinned tiktoken 0.12.0 | 50257 | BPE (tiktoken) | ran | – |
| `cl100k_base` | `cl100k_base` | pinned tiktoken 0.12.0 | 100277 | BPE (tiktoken) | ran | – |
| `o200k_base` | `o200k_base` | pinned tiktoken 0.12.0 | 200019 | BPE (tiktoken) | ran | – |
| `o200k_harmony` | `o200k_harmony` | pinned tiktoken 0.12.0 | 201088 | BPE (tiktoken) | ran | – |
| `qwen2.5_7b` | `Qwen/Qwen2.5-7B` | `d14972939875` | 151665 | Qwen2Tokenizer (BPE) | ran | – |
| `qwen3_8b` | `Qwen/Qwen3-8B` | `b968826d9c46` | 151669 | Qwen2Tokenizer (BPE) | ran | – |
| `llama3.1_8b` | `meta-llama/Llama-3.1-8B` | `d04e592bb4f6` | – | – | skipped | gated; access pending (HTTP 403 when online; absent from the local cache in the offline run) |
| `gemma3_4b` | local copy of `google/gemma-3-4b-it` | `093f9f388b31` | 262145 | GemmaTokenizer (BPE) | ran | – |
| `mistral_7b_v0.1` | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57` | 32000 | TokenizersBackend (BPE) | ran | – |
| `mistral_nemo_tekken` | `mistralai/Mistral-Nemo-Base-2407` | `a4477a2f9779` | 131072 | TokenizersBackend (BPE) | ran | – |
| `deepseek_v3` | `deepseek-ai/DeepSeek-V3` | `e815299b0bcb` | 128815 | TokenizersBackend (BPE) | ran | – |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16` | `434456c9a675` | 131072 | TokenizersBackend (BPE) | ran | – |

The Nemotron repository is the newest ungated NVIDIA Nemotron text base model on Hugging Face when the revisions were resolved (created 2026-08-05). The newer Nemotron-3.5-Super-VL base model is image-text-to-text with a multimodal processor, and Nemotron-Labs repositories are research variants, so neither was used. Gemma 3 was loaded from a git-ignored local copy of its tokenizer files at the pinned revision (`data/tokenizers/gemma3_4b/`, RUNBOOK Section 11), with every other Hugging Face tokenizer loaded from the local cache; all previously reported numbers for the other tokenizers were reproduced unchanged. Llama 3.1 is included automatically once access is granted and the scripts are re-run.

## Task 1 — Open-weight tokenizers on FLORES-200

Same-text corpus ratio (total tokens / total English tokens) on the 997 FLORES-200 dev sentences, with 95% bootstrap CI (2,000 resamples, seed 42).

| Tokenizer | Bengali | Hindi | Arabic | Tamil | Yoruba | EN tokens / sentence |
|---|---:|---:|---:|---:|---:|---:|
| `r50k_base` | 9.61× | 7.44× | 4.37× | 15.43× | 3.86× | 25.8 |
| `cl100k_base` | 5.84× | 4.79× | 3.03× | 7.61× | 2.95× | 25.9 |
| `o200k_base` | 1.71× | 1.58× | 1.38× | 1.98× | 2.18× | 25.6 |
| `o200k_harmony` | 1.71× | 1.58× | 1.38× | 1.98× | 2.18× | 25.6 |
| `qwen2.5_7b` | 5.05× | 4.45× | 1.63× | 6.10× | 2.68× | 26.3 |
| `qwen3_8b` | 5.05× | 4.45× | 1.63× | 6.10× | 2.68× | 26.3 |
| `gemma3_4b` | 1.21× | 1.32× | 1.47× | 1.47× | 2.40× | 26.0 |
| `mistral_7b_v0.1` | 4.93× | 4.58× | 3.48× | 5.84× | 2.84× | 28.6 |
| `mistral_nemo_tekken` | 2.06× | 1.79× | 1.25× | 2.17× | 2.81× | 26.6 |
| `deepseek_v3` | 2.03× | 2.82× | 1.62× | 3.07× | 2.83× | 25.8 |
| `nemotron` | 2.06× | 1.79× | 1.25× | 2.17× | 2.81× | 26.6 |

- English costs 25.6–28.6 tokens per sentence under every tokenizer, so the spread comes from the target languages.
- `gemma3_4b` gives the lowest Bengali, Hindi and Tamil ratios (1.21×, 1.32×, 1.47×); Tekken (and Nemotron) gives the lowest Arabic ratio and `o200k_base` the lowest Yoruba ratio.
- Qwen2.5/Qwen3 and Mistral-7B remain at 4.5–5× for Bengali and Hindi and about 6× for Tamil.
- Identical counts on every sentence: `o200k_harmony` = `o200k_base`, `qwen3_8b` = `qwen2.5_7b`, `nemotron` = `mistral_nemo_tekken`. `gemma3_4b` matches no other tokenizer.

Files: `results/extension/flores_open_weight/`, `figures/extension/flores_open_weight_ratio.png`.

## Task 2 — Hindi script-share sensitivity (TEA corpus)

Hindi labels use the Bengali procedure of `01_quality_check.py` with Devanagari letters and combining marks. Distribution over 120 items: clean 110 (T1 25, T2 50, T3 35), mixed 10 (all T1), english_retained 0. Bengali for comparison: 62 / 50 / 8.

| Tokenizer | All items (n = 120) | Clean-only (n = 110) | T1 all → clean-only |
|---|---:|---:|---:|
| `o200k_base` | 1.72× | 1.75× | 1.86× → 2.03× |
| `gemma3_4b` | 1.42× | 1.45× | 1.55× → 1.69× |
| `mistral_nemo_tekken` / `nemotron` | 1.96× | 2.00× | 2.11× → 2.36× |
| `deepseek_v3` | 3.02× | 3.11× | 3.10× → 3.52× |
| `qwen2.5_7b` / `qwen3_8b` | 4.87× | 5.02× | 4.79× → 5.46× |
| `mistral_7b_v0.1` | 5.20× | 5.37× | 5.08× → 5.79× |

- Retained English is a minor factor for Hindi overall (+0.03 under `o200k_base`) but matters within T1 (+0.17).
- Per-item Hindi TFRs reproduce TEA's `results/tfr_by_item.csv` for `o200k_base`, `qwen2.5_7b` and `mistral_7b_v0.1` (max difference < 1e-15).

Files: `data/extension/tea_hindi_flagged.csv`, `results/extension/hindi_script_share/`.

## Task 3 — Technical vs general text, matched aggregation

Both sides use the mean of per-item (TEA) or per-sentence (FLORES-200) ratios. Gap = TEA − FLORES with a 95% bootstrap CI.

Verification: `o200k_base` TEA means reproduce the published values (Bengali 1.5616 → 1.56, Hindi 1.7190 → 1.72, Arabic 1.4358 → 1.44, Tamil 2.0942 → 2.09, Yoruba 2.3725 → 2.37), and `o200k_base` FLORES means equal `results/flores_same_text/summary.csv` for all five languages. Both checks passed.

Domain gap, all TEA items (Bengali clean-only in brackets):

| Tokenizer | Bengali | Hindi | Arabic | Tamil | Yoruba |
|---|---:|---:|---:|---:|---:|
| `r50k_base` | −1.70 (+0.62) | +0.45 | +0.14 | +0.28 | +0.45 |
| `cl100k_base` | −0.73 (+0.60) | +0.40 | +0.12 | +0.34 | +0.53 |
| `o200k_base` / `o200k_harmony` | −0.16 (+0.12) | +0.13 | +0.05 | +0.10 | +0.18 |
| `qwen2.5_7b` / `qwen3_8b` | −0.63 (+0.50) | +0.36 | +0.05 | +0.34 | +0.47 |
| `gemma3_4b` | +0.07 (+0.23) | +0.09 | +0.04 | +0.12 | +0.34 |
| `mistral_7b_v0.1` | −0.58 (+0.55) | +0.54 | +0.34 | +0.62 | +0.45 |
| `mistral_nemo_tekken` / `nemotron` | −0.26 (+0.07) | +0.15 | +0.08 | +0.12 | +0.43 |
| `deepseek_v3` | −0.25 (+0.09) | +0.17 | +0.05 | +0.12 | +0.38 |

- Bengali is the only language with a negative gap on all items, under every tokenizer except `gemma3_4b` (+0.07); on clean-only items the gap is positive for every tokenizer (CI includes 0 only for Tekken/Nemotron).
- Hindi gaps are positive for every tokenizer, all items and clean-only (+0.09 to +0.76), with CIs excluding 0.
- Yoruba gaps are positive with CIs excluding 0 for all 11 tokenizers; Tamil for 8 of 11; Arabic for 5 of 11.

Files: `results/extension/domain_comparison/`, `figures/extension/domain_gap_bengali_hindi.png`.
