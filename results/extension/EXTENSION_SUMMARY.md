# TEA Extension Summary

This extension builds on the Tokenization Equity Audit (TEA; arXiv:2608.09046) without changing any published TEA output. It adds three analyses: current open-weight tokenizers on FLORES-200 (Task 1), a Hindi script-share sensitivity analysis on the TEA corpus (Task 2), and a technical-versus-general domain comparison that uses the same aggregation on both sides (Task 3), plus a fixed English baseline for the FLORES-200 ratios and a direct comparison of Llama 3.1 with `cl100k_base`. Commands and options are in `RUNBOOK.md`, Sections 11–15.

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
| `qwen3.8_2.4t` | local copy of `Qwen/Qwen3.8-2.4T-A95B` | `4f885e51d6bb` | 248077 | Qwen2Tokenizer (BPE) | ran | – |
| `llama3.1_8b` | local copy of `meta-llama/Llama-3.1-8B` | `d04e592bb4f6` | 128256 | TokenizersBackend (BPE) | ran | – |
| `gemma3_4b` | local copy of `google/gemma-3-4b-it` | `093f9f388b31` | 262145 | GemmaTokenizer (BPE) | ran | – |
| `gemma4_e2b` | local copy of `google/gemma-4-E2B-it` | `4adf15748de9` | 262144 | GemmaTokenizer (BPE) | ran | – |
| `mistral_7b_v0.1` | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57` | 32000 | TokenizersBackend (BPE) | ran | – |
| `mistral_nemo_tekken` | `mistralai/Mistral-Nemo-Base-2407` | `a4477a2f9779` | 131072 | TokenizersBackend (BPE) | ran | – |
| `deepseek_v3` | `deepseek-ai/DeepSeek-V3` | `e815299b0bcb` | 128815 | TokenizersBackend (BPE) | ran | – |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16` | `434456c9a675` | 131072 | TokenizersBackend (BPE) | ran | – |

The Nemotron repository is the newest ungated NVIDIA Nemotron text base model on Hugging Face when the revisions were resolved (created 2026-08-05). The newer Nemotron-3.5-Super-VL base model is image-text-to-text with a multimodal processor, and Nemotron-Labs repositories are research variants, so neither was used. Gemma 4 is `google/gemma-4-E2B-it`, the smallest Gemma 4 model (instruct variant, as for Gemma 3). Qwen3.8 is the newest Qwen LLM family at resolution time; `Qwen/Qwen3.8-2.4T-A95B` is its only text-generation model, and its `tokenizer.json` is byte-identical to those of Qwen3.8-27B and Qwen3.8-Flash-Next. Llama 3.1, Gemma 3, Gemma 4 and Qwen3.8 were loaded from git-ignored local copies of their tokenizer files at the pinned revisions (`data/tokenizers/<label>/`, RUNBOOK Section 11; each file checked against the SHA-256 or git blob id published for it), with every other Hugging Face tokenizer loaded from the local cache; each time a tokenizer was added, all previously reported numbers for the other tokenizers were reproduced unchanged.

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
| `qwen3.8_2.4t` | 3.13× | 3.12× | 1.33× | 4.06× | 2.57× | 26.2 |
| `llama3.1_8b` | 5.79× | 2.54× | 1.66× | 7.62× | 2.71× | 25.9 |
| `gemma3_4b` | 1.21× | 1.32× | 1.47× | 1.47× | 2.40× | 26.0 |
| `gemma4_e2b` | 1.21× | 1.32× | 1.47× | 1.47× | 2.40× | 26.0 |
| `mistral_7b_v0.1` | 4.93× | 4.58× | 3.48× | 5.84× | 2.84× | 28.6 |
| `mistral_nemo_tekken` | 2.06× | 1.79× | 1.25× | 2.17× | 2.81× | 26.6 |
| `deepseek_v3` | 2.03× | 2.82× | 1.62× | 3.07× | 2.83× | 25.8 |
| `nemotron` | 2.06× | 1.79× | 1.25× | 2.17× | 2.81× | 26.6 |

- English costs 25.6–28.6 tokens per sentence under every tokenizer, so the spread comes from the target languages.
- `gemma3_4b` / `gemma4_e2b` give the lowest Bengali, Hindi and Tamil ratios (1.21×, 1.32×, 1.47×); Tekken (and Nemotron) gives the lowest Arabic ratio and `o200k_base` the lowest Yoruba ratio.
- Qwen2.5/Qwen3 and Mistral-7B remain at 4.5–5× for Bengali and Hindi and about 6× for Tamil. Qwen3.8 (new 248k vocabulary) lowers Bengali and Hindi to about 3.1× and Tamil to 4.06×. Llama 3.1 is close to `cl100k_base` for Bengali (5.79×) and Tamil (7.62×) but much lower for Hindi (2.54×) and Arabic (1.66×); see the Llama 3.1 vs `cl100k_base` section.
- Identical counts on every sentence: `o200k_harmony` = `o200k_base`, `qwen3_8b` = `qwen2.5_7b`, `nemotron` = `mistral_nemo_tekken`, `gemma4_e2b` = `gemma3_4b` (also identical on every TEA item; the two `tokenizer.json` vocabularies are not byte-identical). `qwen3.8_2.4t` and `llama3.1_8b` match no other tokenizer.

Files: `results/extension/flores_open_weight/`, `figures/extension/flores_open_weight_ratio.png`.

## Task 2 — Hindi script-share sensitivity (TEA corpus)

Hindi labels use the Bengali procedure of `01_quality_check.py` with Devanagari letters and combining marks. Distribution over 120 items: clean 110 (T1 25, T2 50, T3 35), mixed 10 (all T1), english_retained 0. Bengali for comparison: 62 / 50 / 8.

| Tokenizer | All items (n = 120) | Clean-only (n = 110) | T1 all → clean-only |
|---|---:|---:|---:|
| `o200k_base` | 1.72× | 1.75× | 1.86× → 2.03× |
| `gemma3_4b` / `gemma4_e2b` | 1.42× | 1.45× | 1.55× → 1.69× |
| `mistral_nemo_tekken` / `nemotron` | 1.96× | 2.00× | 2.11× → 2.36× |
| `llama3.1_8b` | 2.77× | 2.84× | 2.78× → 3.12× |
| `deepseek_v3` | 3.02× | 3.11× | 3.10× → 3.52× |
| `qwen3.8_2.4t` | 3.45× | 3.55× | 3.50× → 3.97× |
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
| `qwen3.8_2.4t` | −0.35 (+0.27) | +0.28 | +0.08 | +0.31 | +0.54 |
| `llama3.1_8b` | −0.73 (+0.59) | +0.21 | +0.07 | +0.34 | +0.44 |
| `gemma3_4b` / `gemma4_e2b` | +0.07 (+0.23) | +0.09 | +0.04 | +0.12 | +0.34 |
| `mistral_7b_v0.1` | −0.58 (+0.55) | +0.54 | +0.34 | +0.62 | +0.45 |
| `mistral_nemo_tekken` / `nemotron` | −0.26 (+0.07) | +0.15 | +0.08 | +0.12 | +0.43 |
| `deepseek_v3` | −0.25 (+0.09) | +0.17 | +0.05 | +0.12 | +0.38 |

- Bengali is the only language with a negative gap on all items, under every tokenizer except `gemma3_4b` / `gemma4_e2b` (+0.07); on clean-only items the gap is positive for every tokenizer (CI includes 0 only for Tekken/Nemotron).
- Hindi gaps are positive for every tokenizer, all items and clean-only (+0.09 to +0.76), with CIs excluding 0.
- Yoruba gaps are positive with CIs excluding 0 for all 14 tokenizers; Tamil for 10 of 14; Arabic for 7 of 14.

Files: `results/extension/domain_comparison/`, `figures/extension/domain_gap_bengali_hindi.png`.

## Fixed English baseline

The FLORES-200 corpus ratio divides by each tokenizer's own English count. `scripts/08_fixed_baseline.py` also divides by a fixed English count, that of `o200k_base`, so every tokenizer shares one denominator. Own / fixed corpus ratio:

| Tokenizer | English vs `o200k_base` | Bengali | Hindi | Arabic | Tamil | Yoruba |
|---|---:|---:|---:|---:|---:|---:|
| `r50k_base` | 1.008 | 9.61 / 9.68 | 7.44 / 7.50 | 4.37 / 4.40 | 15.43 / 15.55 | 3.86 / 3.89 |
| `cl100k_base` | 1.010 | 5.84 / 5.90 | 4.79 / 4.84 | 3.03 / 3.06 | 7.61 / 7.69 | 2.95 / 2.98 |
| `o200k_base` / `o200k_harmony` | 1.000 | 1.71 / 1.71 | 1.58 / 1.58 | 1.38 / 1.38 | 1.98 / 1.98 | 2.18 / 2.18 |
| `qwen2.5_7b` / `qwen3_8b` | 1.028 | 5.05 / 5.20 | 4.45 / 4.57 | 1.63 / 1.68 | 6.10 / 6.27 | 2.68 / 2.75 |
| `qwen3.8_2.4t` | 1.025 | 3.13 / 3.21 | 3.12 / 3.20 | 1.33 / 1.36 | 4.06 / 4.16 | 2.57 / 2.64 |
| `llama3.1_8b` | 1.010 | 5.79 / 5.85 | 2.54 / 2.56 | 1.66 / 1.68 | 7.62 / 7.69 | 2.71 / 2.74 |
| `gemma3_4b` / `gemma4_e2b` | 1.014 | 1.21 / 1.23 | 1.32 / 1.34 | 1.47 / 1.49 | 1.47 / 1.49 | 2.40 / 2.43 |
| `mistral_7b_v0.1` | 1.116 | 4.93 / 5.50 | 4.58 / 5.11 | 3.48 / 3.88 | 5.84 / 6.51 | 2.84 / 3.17 |
| `mistral_nemo_tekken` / `nemotron` | 1.037 | 2.06 / 2.13 | 1.79 / 1.86 | 1.25 / 1.30 | 2.17 / 2.25 | 2.81 / 2.91 |
| `deepseek_v3` | 1.006 | 2.03 / 2.04 | 2.82 / 2.84 | 1.62 / 1.63 | 3.07 / 3.08 | 2.83 / 2.84 |

- Every tokenizer encodes English with at least as many tokens as `o200k_base` (1.000–1.116), so fixed ratios are equal to or higher than own ratios; the largest change is for Mistral-7B (+11.6%).
- Rankings within a language change only among close neighbours: with the fixed baseline, Mistral-7B moves above Qwen2.5/Qwen3 for Bengali and Tamil and above `cl100k_base` for Hindi and Yoruba, and for Yoruba, Tekken/Nemotron move above DeepSeek-V3 (2.91 vs 2.84, from 2.81 vs 2.83) and Qwen2.5/Qwen3 move above Llama 3.1 (2.75 vs 2.74, from 2.68 vs 2.71). Arabic is unchanged.

Files: `results/extension/fixed_baseline/`.

## Llama 3.1 vs `cl100k_base`

`scripts/09_llama_vs_cl100k.py` compares the two tokenizers directly.

Vocabulary (by byte string, special tokens excluded): all 100,256 `cl100k_base` tokens are in Llama 3.1's 128,000-token base vocabulary, every one with the same id; Llama 3.1 adds 27,744 tokens.

FLORES-200 dev token totals:

| Language | `cl100k_base` | `llama3.1_8b` | Change | Ratio vs English (`cl100k_base` → `llama3.1_8b`) |
|---|---:|---:|---:|---:|
| English | 25,801 | 25,792 | −0.0% | 1.00× → 1.00× |
| Bengali | 150,555 | 149,436 | −0.7% | 5.84× → 5.79× |
| Hindi | 123,680 | 65,407 | −47.1% | 4.79× → 2.54× |
| Arabic | 78,239 | 42,803 | −45.3% | 3.03× → 1.66× |
| Tamil | 196,415 | 196,414 | −0.0% | 7.61× → 7.62× |
| Yoruba | 76,108 | 69,934 | −8.1% | 2.95× → 2.71× |

- English: 989 of 997 sentences (99.2%) have identical counts; the per-sentence difference (Llama 3.1 − `cl100k_base`) ranges from −2 to 0, median 0.
- The added tokens roughly halve Hindi and Arabic token counts, reduce Yoruba by 8%, and leave Bengali and Tamil essentially unchanged.

Files: `results/extension/llama_vs_cl100k/`.
