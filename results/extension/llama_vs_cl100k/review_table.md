**Token totals on FLORES-200 dev (997 sentences)**

| Language | cl100k_base tokens | llama3.1_8b tokens | Change | Ratio vs English, cl100k_base | Ratio vs English, llama3.1_8b |
|---|---:|---:|---:|---:|---:|
| English | 25,801 | 25,792 | -0.0% | 1.00× | 1.00× |
| Bengali | 150,555 | 149,436 | -0.7% | 5.84× | 5.79× |
| Hindi | 123,680 | 65,407 | -47.1% | 4.79× | 2.54× |
| Arabic | 78,239 | 42,803 | -45.3% | 3.03× | 1.66× |
| Tamil | 196,415 | 196,414 | -0.0% | 7.61× | 7.62× |
| Yoruba | 76,108 | 69,934 | -8.1% | 2.95× | 2.71× |

English: 989 of 997 sentences (99.2%) have identical token counts; per-sentence difference (llama3.1_8b − cl100k_base) min -2, median +0, max +0.

**Vocabulary overlap (by byte string; special tokens excluded)**

| Measure | Tokens |
|---|---:|
| cl100k_base base tokens | 100,256 |
| Llama 3.1 base tokens | 128,000 |
| cl100k_base tokens present in Llama 3.1 | 100,256 |
| cl100k_base tokens absent from Llama 3.1 | 0 |
| Llama 3.1 tokens not in cl100k_base | 27,744 |
| shared tokens with the same id | 100,256 |
