**Hindi label distribution** (TEA corpus, 120 items)

| Label | Items | Share | T1 | T2 | T3 |
|---|---:|---:|---:|---:|---:|
| clean | 110 | 91.7% | 25 | 50 | 35 |
| mixed | 10 | 8.3% | 10 | 0 | 0 |
| english_retained | 0 | 0.0% | 0 | 0 | 0 |

**Hindi TFR, all items vs clean-only** (mean / median, n)

| Tokenizer | All items | Clean-only | Clean − all (mean) |
|---|---:|---:|---:|
| r50k_base | 7.97× / 8.12× (n=120) | 8.28× / 8.24× (n=110) | +0.31 |
| cl100k_base | 5.25× / 5.25× (n=120) | 5.43× / 5.36× (n=110) | +0.18 |
| o200k_base | 1.72× / 1.68× (n=120) | 1.74× / 1.69× (n=110) | +0.03 |
| o200k_harmony | 1.72× / 1.68× (n=120) | 1.74× / 1.69× (n=110) | +0.03 |
| qwen2.5_7b | 4.86× / 4.92× (n=120) | 5.02× / 4.94× (n=110) | +0.16 |
| qwen3_8b | 4.86× / 4.92× (n=120) | 5.02× / 4.94× (n=110) | +0.16 |
| gemma3_4b | 1.42× / 1.37× (n=120) | 1.44× / 1.38× (n=110) | +0.02 |
| mistral_7b_v0.1 | 5.20× / 5.21× (n=120) | 5.37× / 5.36× (n=110) | +0.17 |
| mistral_nemo_tekken | 1.96× / 1.90× (n=120) | 2.00× / 1.93× (n=110) | +0.04 |
| deepseek_v3 | 3.02× / 2.94× (n=120) | 3.11× / 3.07× (n=110) | +0.09 |
| nemotron | 1.96× / 1.90× (n=120) | 2.00× / 1.93× (n=110) | +0.04 |

Script share = Devanagari letters and combining marks / (those + Latin letters); labels and thresholds as in `01_quality_check.py` (clean ≥ 0.75, mixed 0.40–0.75, english_retained < 0.40). Per-item TFR = Hindi tokens / English tokens, computed as in `02_tokenize.py`. By-tier values are in `summary.csv`. Counting every code point in the Devanagari ranges (including the danda) would change 0 label(s).
