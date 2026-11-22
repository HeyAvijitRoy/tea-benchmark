"""
10_verify_extension.py
----------------------
TEA Benchmark — Extension: Reproduction Check

Purpose
-------
Compares the CSVs written by scripts 03, 05, 06, 07, 08 and 09 with the values
reported in README.md (Extension section) and RUNBOOK.md (Sections 10-16), so
that anyone regenerating the extension can confirm their numbers match.

Each check reads one value from one results file and compares it with the
expected value:
    - ratios and means: absolute tolerance (default 0.005);
    - token counts, vocabulary counts and sentence counts: exact match.

Status per check
----------------
    PASS     value within tolerance
    FAIL     value outside tolerance, or a non-gated row is absent
    SKIPPED  row absent for a gated tokenizer (Gemma 3, Gemma 4, Llama 3.1)
             that was not run because licence access was not granted
    MISSING  results file not present (the script that writes it was not run)

The script uses only the Python standard library and reads results files only;
it does not load any tokenizer and needs no network access.

Usage
-----
    python scripts/10_verify_extension.py --self-test
    python scripts/10_verify_extension.py
    python scripts/10_verify_extension.py --tolerance 0.01
    python scripts/10_verify_extension.py --strict    # MISSING / SKIPPED also fail

Exit code: 0 if no check failed, 1 otherwise.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXT = Path("results") / "extension"

# Tokenizers that need licence acceptance on Hugging Face (RUNBOOK Section 11).
GATED = {"gemma3_4b", "gemma4_e2b", "llama3.1_8b"}

# ---------------------------------------------------------------------------
# Expected values
# Each entry: (description, relative path, row filter, column, expected, kind)
#   kind = "ratio" -> compared with --tolerance
#   kind = "exact" -> compared as integers
# ---------------------------------------------------------------------------
FLORES_03 = Path("results") / "flores_same_text" / "summary.csv"
FLORES_05 = EXT / "flores_open_weight" / "summary.csv"
LLAMA_SUM = EXT / "llama_vs_cl100k" / "summary.csv"
LLAMA_VOC = EXT / "llama_vs_cl100k" / "vocabulary.csv"
FIXED = EXT / "fixed_baseline" / "summary.csv"
HINDI = EXT / "hindi_script_share" / "summary.csv"
DOMAIN = EXT / "domain_comparison" / "domain_comparison.csv"


def _tl(tokenizer, language):
    return {"tokenizer": tokenizer, "language": language}


CHECKS = [
    # Section 10: FLORES-200 same-text ratios, tiktoken only (script 03)
    ("03 o200k_base Bengali corpus ratio", FLORES_03, _tl("o200k_base", "Bengali"), "corpus_ratio", 1.7102, "ratio"),
    ("03 o200k_base Bengali CI low", FLORES_03, _tl("o200k_base", "Bengali"), "corpus_ratio_ci_low", 1.6925, "ratio"),
    ("03 o200k_base Bengali CI high", FLORES_03, _tl("o200k_base", "Bengali"), "corpus_ratio_ci_high", 1.7282, "ratio"),
    ("03 o200k_base Hindi corpus ratio", FLORES_03, _tl("o200k_base", "Hindi"), "corpus_ratio", 1.5773, "ratio"),
    ("03 o200k_base Yoruba corpus ratio", FLORES_03, _tl("o200k_base", "Yoruba"), "corpus_ratio", 2.1793, "ratio"),
    ("03 sentences per language", FLORES_03, _tl("o200k_base", "Bengali"), "n_sentences", 997, "exact"),
    # Section 11: open-weight tokenizers (script 05)
    ("05 o200k_base Bengali corpus ratio", FLORES_05, _tl("o200k_base", "Bengali"), "corpus_ratio", 1.7102, "ratio"),
    ("05 cl100k_base Bengali corpus ratio", FLORES_05, _tl("cl100k_base", "Bengali"), "corpus_ratio", 5.8352, "ratio"),
    ("05 qwen3_8b Bengali corpus ratio", FLORES_05, _tl("qwen3_8b", "Bengali"), "corpus_ratio", 5.0530, "ratio"),
    ("05 qwen3.8_2.4t Bengali corpus ratio", FLORES_05, _tl("qwen3.8_2.4t", "Bengali"), "corpus_ratio", 3.1310, "ratio"),
    ("05 mistral_nemo_tekken Bengali corpus ratio", FLORES_05, _tl("mistral_nemo_tekken", "Bengali"), "corpus_ratio", 2.0569, "ratio"),
    ("05 deepseek_v3 Bengali corpus ratio", FLORES_05, _tl("deepseek_v3", "Bengali"), "corpus_ratio", 2.0255, "ratio"),
    ("05 gemma3_4b Bengali corpus ratio", FLORES_05, _tl("gemma3_4b", "Bengali"), "corpus_ratio", 1.2096, "ratio"),
    ("05 gemma3_4b Tamil corpus ratio", FLORES_05, _tl("gemma3_4b", "Tamil"), "corpus_ratio", 1.4681, "ratio"),
    ("05 gemma4_e2b Bengali corpus ratio", FLORES_05, _tl("gemma4_e2b", "Bengali"), "corpus_ratio", 1.2096, "ratio"),
    ("05 llama3.1_8b Bengali corpus ratio", FLORES_05, _tl("llama3.1_8b", "Bengali"), "corpus_ratio", 5.7939, "ratio"),
    # Section 12: Hindi script-share (script 06)
    ("06 o200k_base Hindi TEA mean TFR, all items", HINDI,
     {"tokenizer": "o200k_base", "subset": "all", "tier": "all"}, "mean_tfr", 1.7190, "ratio"),
    ("06 o200k_base Hindi TEA mean TFR, clean only", HINDI,
     {"tokenizer": "o200k_base", "subset": "clean_only", "tier": "all"}, "mean_tfr", 1.7448, "ratio"),
    ("06 o200k_base Hindi clean-only items", HINDI,
     {"tokenizer": "o200k_base", "subset": "clean_only", "tier": "all"}, "n_items", 110, "exact"),
    # Section 13: domain comparison; TEA means match the published TEA values
    ("07 o200k_base Bengali TEA mean TFR", DOMAIN,
     {"tokenizer": "o200k_base", "language": "Bengali", "tea_subset": "all"}, "tea_mean_tfr", 1.5616, "ratio"),
    ("07 o200k_base Hindi TEA mean TFR", DOMAIN,
     {"tokenizer": "o200k_base", "language": "Hindi", "tea_subset": "all"}, "tea_mean_tfr", 1.7190, "ratio"),
    ("07 o200k_base Arabic TEA mean TFR", DOMAIN,
     {"tokenizer": "o200k_base", "language": "Arabic", "tea_subset": "all"}, "tea_mean_tfr", 1.4358, "ratio"),
    ("07 o200k_base Tamil TEA mean TFR", DOMAIN,
     {"tokenizer": "o200k_base", "language": "Tamil", "tea_subset": "all"}, "tea_mean_tfr", 2.0942, "ratio"),
    ("07 o200k_base Yoruba TEA mean TFR", DOMAIN,
     {"tokenizer": "o200k_base", "language": "Yoruba", "tea_subset": "all"}, "tea_mean_tfr", 2.3725, "ratio"),
    # Section 14: fixed English baseline (script 08)
    ("08 o200k_base English total (reference)", FIXED, _tl("gemma3_4b", "English"), "tokens_en_reference", 25536, "exact"),
    ("08 gemma3_4b English relative to o200k_base", FIXED, _tl("gemma3_4b", "English"), "en_relative", 1.0135, "ratio"),
    ("08 gemma3_4b Bengali fixed-baseline ratio", FIXED, _tl("gemma3_4b", "Bengali"), "fixed_ratio", 1.2259, "ratio"),
    ("08 gemma3_4b Hindi fixed-baseline ratio", FIXED, _tl("gemma3_4b", "Hindi"), "fixed_ratio", 1.3351, "ratio"),
    # Section 15: Llama 3.1 vs cl100k_base (script 09)
    ("09 English tokens, cl100k_base", LLAMA_SUM, {"language": "English"}, "tokens_cl100k_base", 25801, "exact"),
    ("09 English tokens, llama3.1_8b", LLAMA_SUM, {"language": "English"}, "tokens_llama3.1_8b", 25792, "exact"),
    ("09 English sentences with identical counts", LLAMA_SUM, {"language": "English"}, "identical_sentences", 989, "exact"),
    ("09 Bengali tokens, llama3.1_8b", LLAMA_SUM, {"language": "Bengali"}, "tokens_llama3.1_8b", 149436, "exact"),
    ("09 Hindi tokens, cl100k_base", LLAMA_SUM, {"language": "Hindi"}, "tokens_cl100k_base", 123680, "exact"),
    ("09 Hindi tokens, llama3.1_8b", LLAMA_SUM, {"language": "Hindi"}, "tokens_llama3.1_8b", 65407, "exact"),
    ("09 cl100k_base tokens present in Llama 3.1", LLAMA_VOC,
     {"metric": "cl100k_base tokens present in Llama 3.1"}, "value", 100256, "exact"),
    ("09 Llama 3.1 tokens not in cl100k_base", LLAMA_VOC,
     {"metric": "Llama 3.1 tokens not in cl100k_base"}, "value", 27744, "exact"),
    ("09 shared tokens with the same id", LLAMA_VOC,
     {"metric": "shared tokens with the same id"}, "value", 100256, "exact"),
]

# Results files that come only from a gated tokenizer: an absent file is SKIPPED.
GATED_FILES = {LLAMA_SUM, LLAMA_VOC}


def configure_console() -> None:
    """Use UTF-8 for output on Windows (as the other scripts do)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def read_rows(path: Path, cache: dict) -> list:
    """Read a CSV once and cache its rows; return None if the file is absent."""
    if path not in cache:
        full = REPO_ROOT / path
        if not full.exists():
            cache[path] = None
        else:
            with open(full, encoding="utf-8", newline="") as f:
                cache[path] = list(csv.DictReader(f))
    return cache[path]


def find_row(rows: list, where: dict):
    """Return the first row whose columns match every key in `where`."""
    for row in rows:
        if all(row.get(k) == v for k, v in where.items()):
            return row
    return None


def compare(actual: str, expected, kind: str, tolerance: float):
    """Return (passed, actual_value) for one value."""
    if kind == "exact":
        value = int(float(actual))
        return value == int(expected), value
    value = float(actual)
    return abs(value - float(expected)) <= tolerance + 1e-12, value


def run_check(check, tolerance: float, cache: dict) -> dict:
    """Evaluate one check and return a result record."""
    desc, path, where, column, expected, kind = check
    gated = where.get("tokenizer") in GATED or path in GATED_FILES
    result = {"check": desc, "file": str(path).replace("\\", "/"),
              "expected": expected, "actual": "", "status": ""}

    rows = read_rows(path, cache)
    if rows is None:
        result["status"] = "SKIPPED" if gated else "MISSING"
        return result

    row = find_row(rows, where)
    if row is None or row.get(column, "") == "":
        result["status"] = "SKIPPED" if gated else "FAIL"
        return result

    try:
        passed, actual = compare(row[column], expected, kind, tolerance)
    except ValueError:
        result.update(actual=row[column], status="FAIL")
        return result
    result.update(actual=actual, status="PASS" if passed else "FAIL")
    return result


def print_report(results: list) -> dict:
    """Print one line per check and a summary; return the status counts."""
    width = max(len(r["check"]) for r in results)
    for r in results:
        actual = "" if r["actual"] == "" else r["actual"]
        print(f"  {r['status']:<8} {r['check']:<{width}}  expected {r['expected']!s:<9} got {actual}")
    counts = {s: sum(r["status"] == s for r in results)
              for s in ("PASS", "FAIL", "SKIPPED", "MISSING")}
    print("\n  " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    missing_files = sorted({r["file"] for r in results if r["status"] == "MISSING"})
    for f in missing_files:
        print(f"  missing file: {f}")
    return counts


def self_test() -> None:
    """Check the comparison and lookup logic on synthetic values."""
    assert compare("1.7102", 1.7102, "ratio", 0.005) == (True, 1.7102)
    assert compare("1.7160", 1.7102, "ratio", 0.005)[0] is False
    assert compare("1.7150", 1.7102, "ratio", 0.005)[0] is True
    assert compare("989.0", 989, "exact", 0.005) == (True, 989)
    assert compare("988", 989, "exact", 0.005)[0] is False
    rows = [{"tokenizer": "a", "language": "x", "v": "1"},
            {"tokenizer": "b", "language": "x", "v": "2"}]
    assert find_row(rows, {"tokenizer": "b", "language": "x"})["v"] == "2"
    assert find_row(rows, {"tokenizer": "c"}) is None

    # A gated tokenizer with no row is SKIPPED; a non-gated one is FAIL.
    fake = Path("fake.csv")
    cache = {fake: rows}
    assert run_check(("g", fake, {"tokenizer": "gemma3_4b"}, "v", 1, "exact"), 0.005, cache)["status"] == "SKIPPED"
    assert run_check(("n", fake, {"tokenizer": "zzz"}, "v", 1, "exact"), 0.005, cache)["status"] == "FAIL"
    assert run_check(("p", fake, {"tokenizer": "a"}, "v", 1, "exact"), 0.005, cache)["status"] == "PASS"
    # An absent file is MISSING (or SKIPPED for a gated tokenizer).
    absent = Path("results") / "does_not_exist.csv"
    assert run_check(("m", absent, {"tokenizer": "a"}, "v", 1, "exact"), 0.005, {})["status"] == "MISSING"
    assert run_check(("s", absent, {"tokenizer": "llama3.1_8b"}, "v", 1, "exact"), 0.005, {})["status"] == "SKIPPED"
    print("Self-test passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare regenerated extension results with the reported values.")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="absolute tolerance for ratios and means (default 0.005)")
    parser.add_argument("--strict", action="store_true",
                        help="treat MISSING and SKIPPED checks as failures")
    parser.add_argument("--self-test", action="store_true", help="run the built-in tests and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return None

    print(f"Checking {len(CHECKS)} reported values (ratio tolerance ±{args.tolerance}) ...\n")
    cache = {}
    results = [run_check(c, args.tolerance, cache) for c in CHECKS]
    counts = print_report(results)

    failed = counts["FAIL"] > 0
    if args.strict and (counts["MISSING"] or counts["SKIPPED"]):
        failed = True
    print("\nResult: " + ("FAILED" if failed else "OK"))
    counts["exit_code"] = 1 if failed else 0
    return counts


if __name__ == "__main__":
    configure_console()
    stats = main()
    if stats is not None:
        try:
            # Notes.md is a local, git-ignored run log; logging is optional.
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from log_run import append_run_entry
            append_run_entry(
                script="10_verify_extension.py",
                note="Extension reproduction check against reported values.",
                stats={k: v for k, v in stats.items() if k != "exit_code"},
            )
        except Exception:
            pass
        sys.exit(stats["exit_code"])
