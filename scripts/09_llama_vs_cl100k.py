"""
09_llama_vs_cl100k.py
---------------------
TEA Benchmark — Extension: Llama 3.1 vs cl100k_base

Purpose
-------
The Llama 3 tokenizer is reported to extend OpenAI's cl100k_base vocabulary
with about 28,000 additional tokens, many of them for languages other than
English. This script checks that relationship on two levels:

1. Token counts on FLORES-200 dev (from Task 1's per-sentence counts):
   - total tokens per language under cl100k_base and llama3.1_8b, and the
     percentage change;
   - for English: how many of the 997 sentences have the same count under both,
     and the distribution (min / median / max) of per-sentence differences
     (llama3.1_8b − cl100k_base);
   - each language's same-text corpus ratio under both tokenizers.

2. Vocabularies, compared by byte string:
   - how many cl100k_base tokens are present in the Llama 3.1 base vocabulary;
   - how many Llama 3.1 base tokens are not in cl100k_base;
   - how many shared tokens also have the same id (cl100k rank == Llama id).
   Llama 3.1 uses byte-level BPE, so its token strings are mapped back to raw
   bytes with the GPT-2 byte-to-unicode table before comparing. Special
   (added) tokens are excluded on both sides.

Inputs
------
    results/extension/flores_open_weight/per_sentence_counts.csv   (Task 1)
    tiktoken cl100k_base (local tiktoken cache)
    Llama 3.1 tokenizer.json from data/tokenizers/llama3.1_8b/ if present,
    otherwise from the Hugging Face cache at the pinned revision (offline)

Usage
-----
    python scripts/09_llama_vs_cl100k.py
    python scripts/09_llama_vs_cl100k.py --self-test

Outputs (results/extension/llama_vs_cl100k/)
--------------------------------------------
    summary.csv       one row per language (English first)
    vocabulary.csv    vocabulary overlap counts
    review_table.md   both tables in Markdown
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve repository root (same convention as 02_tokenize.py) ─────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


def load_script_module(name: str, filename: str):
    """Import a numbered script (e.g. 04_tokenizer_registry.py) as a module."""
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


registry = load_script_module("tokenizer_registry", "04_tokenizer_registry.py")

LANGUAGES = ["English", "Bengali", "Hindi", "Arabic", "Tamil", "Yoruba"]
CL100K = "cl100k_base"
LLAMA = "llama3.1_8b"
COUNTS_PATH = Path("results") / "extension" / "flores_open_weight" / "per_sentence_counts.csv"
OUT_DIR = Path("results") / "extension" / "llama_vs_cl100k"


# ── Count comparison ─────────────────────────────────────────────────────────

def compare_counts(counts: pd.DataFrame, a: str = CL100K, b: str = LLAMA) -> pd.DataFrame:
    """One row per language: totals, % change, same-text ratios, English detail."""
    sub = counts[counts["tokenizer"].isin([a, b])]
    wide = sub.pivot_table(index=["language", "sentence_id"], columns="tokenizer",
                           values="tokens").reset_index()
    en = wide[wide["language"] == "English"]
    en_a, en_b = float(en[a].sum()), float(en[b].sum())

    rows = []
    for lang in LANGUAGES:
        w = wide[wide["language"] == lang]
        if w.empty:
            continue
        ta, tb = float(w[a].sum()), float(w[b].sum())
        row = {
            "language":            lang,
            "n_sentences":         len(w),
            f"tokens_{a}":         int(ta),
            f"tokens_{b}":         int(tb),
            "pct_change":          100.0 * (tb - ta) / ta,
            f"ratio_{a}":          ta / en_a,
            f"ratio_{b}":          tb / en_b,
            "identical_sentences": np.nan,
            "identical_share":     np.nan,
            "diff_min":            np.nan,
            "diff_median":         np.nan,
            "diff_max":            np.nan,
        }
        if lang == "English":
            d = (w[b] - w[a]).to_numpy()
            row.update({
                "identical_sentences": int((d == 0).sum()),
                "identical_share":     float((d == 0).mean()),
                "diff_min":            float(d.min()),
                "diff_median":         float(np.median(d)),
                "diff_max":            float(d.max()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ── Vocabulary comparison ────────────────────────────────────────────────────

def bytes_to_unicode() -> dict:
    """The GPT-2 byte-level BPE mapping from bytes (0–255) to printable characters."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


UNICODE_TO_BYTE = {c: b for b, c in bytes_to_unicode().items()}


def token_str_to_bytes(token: str) -> bytes:
    return bytes(UNICODE_TO_BYTE[c] for c in token)


def llama_tokenizer_json() -> Path:
    local = registry.local_tokenizer_dir(LLAMA)
    if local is not None and (local / "tokenizer.json").is_file():
        return local / "tokenizer.json"
    from huggingface_hub import hf_hub_download
    spec = registry.SPECS[LLAMA]
    return Path(hf_hub_download(spec.source, "tokenizer.json", revision=spec.revision,
                                local_files_only=True))


def load_llama_vocab(path: Path) -> dict:
    """{token bytes: id} for the base (non-added) vocabulary."""
    j = json.loads(path.read_text(encoding="utf-8"))
    return {token_str_to_bytes(tok): idx for tok, idx in j["model"]["vocab"].items()}


def load_cl100k_vocab() -> dict:
    """{token bytes: rank} for cl100k_base's mergeable (non-special) tokens."""
    import tiktoken
    return dict(tiktoken.get_encoding(CL100K)._mergeable_ranks)


def compare_vocab(cl: dict, ll: dict) -> pd.DataFrame:
    shared = cl.keys() & ll.keys()
    same_id = sum(1 for t in shared if cl[t] == ll[t])
    rows = [
        ("cl100k_base base tokens", len(cl)),
        ("Llama 3.1 base tokens", len(ll)),
        ("cl100k_base tokens present in Llama 3.1", len(shared)),
        ("cl100k_base tokens absent from Llama 3.1", len(cl) - len(shared)),
        ("Llama 3.1 tokens not in cl100k_base", len(ll) - len(shared)),
        ("shared tokens with the same id", same_id),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# ── Output ───────────────────────────────────────────────────────────────────

def write_review_table(summary: pd.DataFrame, vocab: pd.DataFrame, path: Path) -> None:
    e = summary[summary["language"] == "English"].iloc[0]
    lines = [
        "**Token totals on FLORES-200 dev (997 sentences)**",
        "",
        "| Language | cl100k_base tokens | llama3.1_8b tokens | Change | Ratio vs English, cl100k_base | Ratio vs English, llama3.1_8b |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r['language']} | {r['tokens_cl100k_base']:,} | {r['tokens_llama3.1_8b']:,} "
                     f"| {r['pct_change']:+.1f}% | {r['ratio_cl100k_base']:.2f}× | {r['ratio_llama3.1_8b']:.2f}× |")
    lines += [
        "",
        f"English: {int(e['identical_sentences'])} of {int(e['n_sentences'])} sentences "
        f"({100 * e['identical_share']:.1f}%) have identical token counts; per-sentence difference "
        f"(llama3.1_8b − cl100k_base) min {e['diff_min']:+.0f}, median {e['diff_median']:+.0f}, "
        f"max {e['diff_max']:+.0f}.",
        "",
        "**Vocabulary overlap (by byte string; special tokens excluded)**",
        "",
        "| Measure | Tokens |",
        "|---|---:|",
    ]
    for _, r in vocab.iterrows():
        lines.append(f"| {r['metric']} | {int(r['value']):,} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_results(summary: pd.DataFrame, vocab: pd.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("LLAMA 3.1 VS CL100K_BASE ON FLORES-200 DEV")
    print("=" * 92)
    print(f"{'Language':<9}{'cl100k':>10}{'llama3.1':>10}{'Change':>9}{'Ratio cl100k':>14}{'Ratio llama':>13}")
    for _, r in summary.iterrows():
        print(f"{r['language']:<9}{r['tokens_cl100k_base']:>10,}{r['tokens_llama3.1_8b']:>10,}"
              f"{r['pct_change']:>+8.1f}%{r['ratio_cl100k_base']:>13.2f}x{r['ratio_llama3.1_8b']:>12.2f}x")
    e = summary[summary["language"] == "English"].iloc[0]
    print(f"\nEnglish: identical counts in {int(e['identical_sentences'])}/{int(e['n_sentences'])} "
          f"sentences ({100 * e['identical_share']:.1f}%); difference min {e['diff_min']:+.0f}, "
          f"median {e['diff_median']:+.0f}, max {e['diff_max']:+.0f}")
    print("\nVOCABULARY")
    for _, r in vocab.iterrows():
        print(f"  {r['metric']:<44}{int(r['value']):>9,}")
    print("=" * 92)


# ── Self-test ────────────────────────────────────────────────────────────────

def self_test():
    print("Running self-test on synthetic data ...")
    # Byte-level mapping: space -> 'Ġ', newline -> 'Ċ', printable ASCII unchanged.
    assert token_str_to_bytes("Ġhello") == b" hello"
    assert token_str_to_bytes("Ċ") == b"\n"
    assert token_str_to_bytes("abc") == b"abc"
    assert len(bytes_to_unicode()) == 256 and len(UNICODE_TO_BYTE) == 256
    # Bengali "অ" (U+0985) is 3 UTF-8 bytes, each mapped to one character.
    enc = "".join(bytes_to_unicode()[b] for b in "অ".encode("utf-8"))
    assert token_str_to_bytes(enc) == "অ".encode("utf-8")

    # Counts: 3 sentences, English and Bengali.
    #   English cl100k 10, 12, 8  (30)   llama 10, 11, 8  (29)  diffs 0, -1, 0
    #   Bengali cl100k 60, 60, 60 (180)  llama 30, 30, 30 (90)
    counts = pd.DataFrame({
        "sentence_id": [1, 2, 3] * 4,
        "language":    (["English"] * 3 + ["Bengali"] * 3) * 2,
        "tokenizer":   [CL100K] * 6 + [LLAMA] * 6,
        "tokens":      [10, 12, 8, 60, 60, 60, 10, 11, 8, 30, 30, 30],
    })
    s = compare_counts(counts).set_index("language")
    assert s.loc["English", "tokens_cl100k_base"] == 30 and s.loc["English", "tokens_llama3.1_8b"] == 29
    assert math.isclose(s.loc["English", "pct_change"], -100 / 30)          # -3.33%
    assert s.loc["English", "identical_sentences"] == 2
    assert math.isclose(s.loc["English", "identical_share"], 2 / 3)
    assert (s.loc["English", "diff_min"], s.loc["English", "diff_median"],
            s.loc["English", "diff_max"]) == (-1.0, 0.0, 0.0)
    assert math.isclose(s.loc["Bengali", "pct_change"], -50.0)
    assert math.isclose(s.loc["Bengali", "ratio_cl100k_base"], 6.0)          # 180 / 30
    assert math.isclose(s.loc["Bengali", "ratio_llama3.1_8b"], 90 / 29)
    assert math.isnan(s.loc["Bengali", "identical_sentences"])

    # Vocabulary: 3 shared (2 with the same id), 1 only in cl100k, 2 only in Llama.
    cl = {b"a": 0, b"b": 1, b" the": 2, b"x": 3}
    ll = {b"a": 0, b"b": 1, b" the": 5, "অ".encode(): 6, b"zz": 7}
    v = compare_vocab(cl, ll).set_index("metric")["value"]
    assert v["cl100k_base tokens present in Llama 3.1"] == 3
    assert v["cl100k_base tokens absent from Llama 3.1"] == 1
    assert v["Llama 3.1 tokens not in cl100k_base"] == 2
    assert v["shared tokens with the same id"] == 2
    print("Self-test passed: byte-level decoding, totals and % change (−3.33%, −50%), "
          "English identical share 2/3 and differences (−1, 0, 0), ratios, and vocabulary "
          "overlap counts all match hand-computed values.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true",
                        help="Check the computation on synthetic data and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return None

    if not COUNTS_PATH.is_file():
        sys.exit(f"\nMissing {COUNTS_PATH}. Run scripts/05_flores_open_weight.py first.\n")
    counts = pd.read_csv(COUNTS_PATH, encoding="utf-8")
    missing = {CL100K, LLAMA} - set(counts["tokenizer"])
    if missing:
        sys.exit(f"\n{sorted(missing)} not in {COUNTS_PATH}; run Task 1 with them first.\n")
    summary = compare_counts(counts)

    print("Comparing vocabularies ...")
    vocab = compare_vocab(load_cl100k_vocab(), load_llama_vocab(llama_tokenizer_json()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8")
    vocab.to_csv(OUT_DIR / "vocabulary.csv", index=False, encoding="utf-8")
    write_review_table(summary, vocab, OUT_DIR / "review_table.md")
    print_results(summary, vocab)
    print(f"\nOutput files written:")
    for name in ("summary.csv", "vocabulary.csv", "review_table.md"):
        print(f"  {OUT_DIR / name}")
    return {"languages": len(summary)}


if __name__ == "__main__":
    registry.configure_console()
    stats = main()
    if stats is not None:
        from log_run import append_run_entry
        append_run_entry(
            script="09_llama_vs_cl100k.py",
            note="Llama 3.1 vs cl100k_base: token counts and vocabulary overlap.",
            stats=stats,
        )
