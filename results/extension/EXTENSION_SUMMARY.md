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
| `llama3.1_8b` | `meta-llama/Llama-3.1-8B` | `d04e592bb4f6` | – | – | skipped | gated; access not granted to the account used (HTTP 403) |
| `gemma3_4b` | `google/gemma-3-4b-it` | `093f9f388b31` | – | – | skipped | gated; access not granted to the account used (HTTP 403) |
| `mistral_7b_v0.1` | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57` | 32000 | TokenizersBackend (BPE) | ran | – |
| `mistral_nemo_tekken` | `mistralai/Mistral-Nemo-Base-2407` | `a4477a2f9779` | 131072 | TokenizersBackend (BPE) | ran | – |
| `deepseek_v3` | `deepseek-ai/DeepSeek-V3` | `e815299b0bcb` | 128815 | TokenizersBackend (BPE) | ran | – |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16` | `434456c9a675` | 131072 | TokenizersBackend (BPE) | ran | – |

The Nemotron repository is the newest ungated NVIDIA Nemotron text base model on Hugging Face when the revisions were resolved (created 2026-08-05). The newer Nemotron-3.5-Super-VL base model is image-text-to-text with a multimodal processor, and Nemotron-Labs repositories are research variants, so neither was used. The gated tokenizers are included automatically once access is granted and the scripts are re-run.

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
| `mistral_7b_v0.1` | 4.93× | 4.58× | 3.48× | 5.84× | 2.84× | 28.6 |
| `mistral_nemo_tekken` | 2.06× | 1.79× | 1.25× | 2.17× | 2.81× | 26.6 |
| `deepseek_v3` | 2.03× | 2.82× | 1.62× | 3.07× | 2.83× | 25.8 |
| `nemotron` | 2.06× | 1.79× | 1.25× | 2.17× | 2.81× | 26.6 |

- English costs 25.6–28.6 tokens per sentence under every tokenizer, so the spread comes from the target languages.
- `o200k_base` gives the lowest Bengali, Hindi and Tamil ratios; Tekken (and Nemotron) gives the lowest Arabic ratio.
- Qwen2.5/Qwen3 and Mistral-7B remain at 4.5–5× for Bengali and Hindi and about 6× for Tamil.
- Identical counts on every sentence: `o200k_harmony` = `o200k_base`, `qwen3_8b` = `qwen2.5_7b`, `nemotron` = `mistral_nemo_tekken`.

Files: `results/extension/flores_open_weight/`, `figures/extension/flores_open_weight_ratio.png`.
