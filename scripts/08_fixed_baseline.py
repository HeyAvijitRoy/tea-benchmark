"""
08_fixed_baseline.py
--------------------
TEA Benchmark — Extension: Fixed English Baseline

Purpose
-------
A same-text ratio divides a language's token count by the SAME tokenizer's
English count. Tokenizers differ in how efficiently they encode English, so
the ratio mixes two things: the cost of the language and the cost of English.
This script separates them by also dividing by a fixed reference English
count, that of o200k_base, which is identical for every tokenizer.

For every tokenizer and language, on the FLORES-200 dev counts from Task 1
(results/extension/flores_open_weight/per_sentence_counts.csv):

    own_ratio      = sum(tokens_L, tokenizer)  / sum(tokens_EN, tokenizer)
    fixed_ratio    = sum(tokens_L, tokenizer)  / sum(tokens_EN, o200k_base)
    en_relative    = sum(tokens_EN, tokenizer) / sum(tokens_EN, o200k_base)

so that fixed_ratio = own_ratio × en_relative. own_ratio is the corpus ratio
reported in Task 1. For English, own_ratio = 1 and fixed_ratio = en_relative.

Usage
-----
    python scripts/08_fixed_baseline.py
    python scripts/08_fixed_baseline.py --reference cl100k_base
    python scripts/08_fixed_baseline.py --self-test

Outputs (results/extension/fixed_baseline/)
-------------------------------------------
    summary.csv       one row per tokenizer x language
    review_table.md   compact table (own / fixed ratio per language, English relative)
"""

import argparse
import importlib.util
import math
import os
import sys
import warnings
from pathlib import Path

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
DEFAULT_REFERENCE = "o200k_base"
COUNTS_PATH = Path("results") / "extension" / "flores_open_weight" / "per_sentence_counts.csv"
OUT_DIR = Path("results") / "extension" / "fixed_baseline"

# Expected values (two decimals; English relative to three) used as a check.
EXPECTED = {
    ("gemma3_4b", "Bengali"): (1.21, 1.23),
    ("gemma3_4b", "Tamil"):   (1.47, 1.49),
}
EXPECTED_EN_RELATIVE = {"gemma3_4b": 1.014}


# ── Computation ──────────────────────────────────────────────────────────────

def fixed_baseline(counts: pd.DataFrame, reference: str = DEFAULT_REFERENCE) -> pd.DataFrame:
    """counts: long table with sentence_id, language, tokenizer, tokens."""
    if reference not in set(counts["tokenizer"]):
        sys.exit(f"\nReference tokenizer '{reference}' not found in the counts.\n")
    totals = counts.groupby(["tokenizer", "language"])["tokens"].sum()
    n_sent = counts.groupby(["tokenizer", "language"])["sentence_id"].nunique()
    ref_en = float(totals[(reference, "English")])

    rows = []
    for tok in counts["tokenizer"].unique():
        own_en = float(totals[(tok, "English")])
        for lang in LANGUAGES:
            if (tok, lang) not in totals.index:
                continue
            t = float(totals[(tok, lang)])
            rows.append({
                "tokenizer":             tok,
                "language":              lang,
                "n_sentences":           int(n_sent[(tok, lang)]),
                "tokens":                int(t),
                "tokens_en_own":         int(own_en),
                "tokens_en_reference":   int(ref_en),
                "reference":             reference,
                "own_ratio":             t / own_en,
                "fixed_ratio":           t / ref_en,
                "en_relative":           own_en / ref_en,
            })
    df = pd.DataFrame(rows)
    order = {l: i for i, l in enumerate(registry.ALL_LABELS)}
    df["_t"] = df["tokenizer"].map(order).fillna(len(order))
    df["_l"] = df["language"].map({l: i for i, l in enumerate(LANGUAGES)})
    return df.sort_values(["_t", "_l"]).drop(columns=["_t", "_l"]).reset_index(drop=True)


def check_expected(summary: pd.DataFrame) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for (tok, lang), (own, fixed) in EXPECTED.items():
        r = summary[(summary["tokenizer"] == tok) & (summary["language"] == lang)]
        if r.empty:
            ok = False
            lines.append(f"{tok} {lang}: not found")
            continue
        r = r.iloc[0]
        match = round(r["own_ratio"], 2) == own and round(r["fixed_ratio"], 2) == fixed
        ok &= match
        lines.append(f"{tok} {lang}: own {r['own_ratio']:.4f} -> {r['own_ratio']:.2f} (expected {own:.2f}), "
                     f"fixed {r['fixed_ratio']:.4f} -> {r['fixed_ratio']:.2f} (expected {fixed:.2f})  "
                     f"{'MATCH' if match else 'MISMATCH'}")
    for tok, en in EXPECTED_EN_RELATIVE.items():
        r = summary[(summary["tokenizer"] == tok) & (summary["language"] == "English")]
        val = float(r["en_relative"].iloc[0]) if not r.empty else float("nan")
        match = round(val, 3) == en
        ok &= match
        lines.append(f"{tok} English relative to reference: {val:.4f} -> {val:.3f} "
                     f"(expected {en:.3f})  {'MATCH' if match else 'MISMATCH'}")
    return ok, lines


# ── Output ───────────────────────────────────────────────────────────────────

def write_review_table(summary: pd.DataFrame, path: Path) -> None:
    ref = summary["reference"].iloc[0]
    langs = [l for l in LANGUAGES if l != "English"]
    lines = [
        "| Tokenizer | English vs " + f"`{ref}`" + " | "
        + " | ".join(f"{l} own / fixed" for l in langs) + " |",
        "|---|---:|" + "---:|" * len(langs),
    ]
    for tok in summary["tokenizer"].unique():
        t = summary[summary["tokenizer"] == tok].set_index("language")
        cells = [f"{t.loc[l, 'own_ratio']:.2f}× / {t.loc[l, 'fixed_ratio']:.2f}×" for l in langs]
        lines.append(f"| {tok} | {t.loc['English', 'en_relative']:.3f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"FLORES-200 dev, {int(summary['n_sentences'].max())} sentences. Own = tokens in the "
        "language / the same tokenizer's English tokens (Task 1 corpus ratio). Fixed = tokens in "
        f"the language / `{ref}` English tokens, the same denominator for every tokenizer. "
        f"English vs `{ref}` = the tokenizer's English tokens / `{ref}` English tokens.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: pd.DataFrame) -> None:
    langs = [l for l in LANGUAGES if l != "English"]
    print("\n" + "=" * 100)
    print(f"FIXED ENGLISH BASELINE (reference: {summary['reference'].iloc[0]}) — own / fixed corpus ratio")
    print("=" * 100)
    print(f"{'Tokenizer':<21}{'EN rel':>8}" + "".join(f"{l:>14}" for l in langs))
    for tok in summary["tokenizer"].unique():
        t = summary[summary["tokenizer"] == tok].set_index("language")
        print(f"{tok:<21}{t.loc['English', 'en_relative']:>8.3f}"
              + "".join(f"{t.loc[l, 'own_ratio']:>7.2f}/{t.loc[l, 'fixed_ratio']:<6.2f}" for l in langs))
    print("=" * 100)


# ── Self-test ────────────────────────────────────────────────────────────────

def self_test():
    """
    Two tokenizers, two sentences, English and Bengali:
        ref:  EN 10 + 10 = 20,  BN 30 + 30 = 60
        tokA: EN 12 + 13 = 25,  BN 40 + 35 = 75
    ref:  own 60/20 = 3.0, fixed 3.0, EN relative 1.0
    tokA: own 75/25 = 3.0, fixed 75/20 = 3.75, EN relative 25/20 = 1.25
    """
    print("Running self-test on synthetic data ...")
    global LANGUAGES
    saved = LANGUAGES
    LANGUAGES = ["English", "Bengali"]
    try:
        counts = pd.DataFrame({
            "sentence_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "language":    ["English", "English", "Bengali", "Bengali"] * 2,
            "tokenizer":   ["ref"] * 4 + ["tokA"] * 4,
            "tokens":      [10, 10, 30, 30, 12, 13, 40, 35],
        })
        s = fixed_baseline(counts, reference="ref").set_index(["tokenizer", "language"])
        assert math.isclose(s.loc[("ref", "Bengali"), "own_ratio"], 3.0)
        assert math.isclose(s.loc[("ref", "Bengali"), "fixed_ratio"], 3.0)
        assert math.isclose(s.loc[("ref", "English"), "en_relative"], 1.0)
        assert math.isclose(s.loc[("tokA", "Bengali"), "own_ratio"], 3.0)
        assert math.isclose(s.loc[("tokA", "Bengali"), "fixed_ratio"], 3.75)
        assert math.isclose(s.loc[("tokA", "English"), "en_relative"], 1.25)
        assert math.isclose(s.loc[("tokA", "English"), "own_ratio"], 1.0)
        assert math.isclose(s.loc[("tokA", "English"), "fixed_ratio"], 1.25)
        # fixed = own x en_relative
        assert math.isclose(s.loc[("tokA", "Bengali"), "fixed_ratio"],
                            s.loc[("tokA", "Bengali"), "own_ratio"] * s.loc[("tokA", "English"), "en_relative"])
        assert int(s.loc[("tokA", "Bengali"), "n_sentences"]) == 2
    finally:
        LANGUAGES = saved
    print("Self-test passed: own ratio 3.00, fixed ratio 3.75, English relative 1.25, and "
          "fixed = own × English relative, all as computed by hand.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="Tokenizer whose English count is the fixed denominator "
                             "(default o200k_base)")
    parser.add_argument("--self-test", action="store_true",
                        help="Check the computation on synthetic data and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return None

    if not COUNTS_PATH.is_file():
        sys.exit(f"\nMissing {COUNTS_PATH}. Run scripts/05_flores_open_weight.py first.\n")
    counts = pd.read_csv(COUNTS_PATH, encoding="utf-8")
    summary = fixed_baseline(counts, args.reference)

    if args.reference == DEFAULT_REFERENCE:
        print("\nCHECK AGAINST EXPECTED VALUES")
        ok, lines = check_expected(summary)
        for line in lines:
            print(f"  {line}")
        if not ok:
            sys.exit("\nCHECK FAILED: see above. No outputs were written.\n")
        print("  All expected values reproduced.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8")
    write_review_table(summary, OUT_DIR / "review_table.md")
    print_summary(summary)
    print(f"\nOutput files written:\n  {OUT_DIR / 'summary.csv'}\n  {OUT_DIR / 'review_table.md'}")
    return {"tokenizers": summary["tokenizer"].nunique(), "reference": args.reference}


if __name__ == "__main__":
    registry.configure_console()
    stats = main()
    if stats is not None:
        from log_run import append_run_entry
        append_run_entry(
            script="08_fixed_baseline.py",
            note="FLORES-200 corpus ratios against a fixed English baseline.",
            stats=stats,
        )
