"""
01_quality_check.py
-------------------
TEA Benchmark — Step 1: Bengali Translation Quality Classification

Computes the ratio of Bengali Unicode characters to total alphabetic characters
for each item in the Bengali translation column, then assigns a quality label:

    clean            : Bengali ratio >= 0.75  (predominantly Bengali)
    mixed            : 0.40 <= ratio < 0.75   (significant code-switching)
    english_retained : ratio < 0.40           (predominantly English/Latin)

Output
------
Writes two files to data/:
    tea_corpus_flagged.csv   — original corpus with bn_quality and bn_ratio columns added
    quality_report.txt       — summary statistics for the methodology section

Usage
-----
    python scripts/01_quality_check.py

Requires: pandas (no other dependencies)
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path

# ── Unicode range for Bengali script ──────────────────────────────────────────
# U+0980–U+09FF covers the entire Bengali block: vowels, consonants,
# matras (vowel signs), conjuncts, digits, and punctuation marks.
BENGALI_START = 0x0980
BENGALI_END   = 0x09FF

# ── Thresholds ─────────────────────────────────────────────────────────────────
CLEAN_THRESHOLD  = 0.75
MIXED_THRESHOLD  = 0.40


def configure_console() -> None:
    """Use UTF-8 for multilingual output on Windows and compatible terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def is_bengali_char(char: str) -> bool:
    """Return True if the character falls within the Bengali Unicode block."""
    return BENGALI_START <= ord(char) <= BENGALI_END


def is_latin_alpha(char: str) -> bool:
    """Return True if the character is a Latin alphabetic character (A-Z, a-z)."""
    return char.isalpha() and ord(char) < 128


def compute_bengali_ratio(text: str) -> float:
    """
    Compute the ratio of Bengali characters to total alphabetic characters.

    Only alphabetic characters (Bengali or Latin) are counted. Digits,
    punctuation, spaces, and special characters are excluded from both
    the numerator and denominator. This ensures that retained Python
    identifiers (e.g. 'ZeroDivisionError') count as Latin but that
    colons, brackets, and whitespace do not dilute the denominator.

    Returns 0.0 for strings with no alphabetic content.
    """
    bengali_count = sum(1 for c in text if is_bengali_char(c))
    latin_count   = sum(1 for c in text if is_latin_alpha(c))
    total_alpha   = bengali_count + latin_count

    if total_alpha == 0:
        return 0.0
    return bengali_count / total_alpha


def assign_quality_label(ratio: float) -> str:
    """Map a Bengali character ratio to a quality label."""
    if ratio >= CLEAN_THRESHOLD:
        return "clean"
    elif ratio >= MIXED_THRESHOLD:
        return "mixed"
    else:
        return "english_retained"


def extract_tier(key: str) -> str:
    """Extract tier label (T1, T2, T3) from item key."""
    match = re.match(r'^(T[123])', key)
    return match.group(1) if match else "unknown"


def run_quality_check(
    corpus_path: str = "data/tea_corpus.csv",
    output_path: str = "data/tea_corpus_flagged.csv",
    report_path: str = "data/quality_report.txt"
) -> pd.DataFrame:
    """
    Main function. Reads the corpus, computes Bengali ratios,
    assigns quality labels, writes flagged corpus and report.
    """
    # ── Load corpus ────────────────────────────────────────────────────────────
    print(f"Loading corpus from {corpus_path} ...")
    df = pd.read_csv(corpus_path, encoding="utf-8")

    required_cols = {"Key", "English", "Bengali"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in corpus: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # ── Compute ratio and label ────────────────────────────────────────────────
    print("Computing Bengali character ratios ...")
    df["bn_ratio"]   = df["Bengali"].apply(
        lambda x: compute_bengali_ratio(str(x)) if pd.notna(x) else 0.0
    )
    df["bn_quality"] = df["bn_ratio"].apply(assign_quality_label)
    df["tier"]       = df["Key"].apply(extract_tier)

    # ── Save flagged corpus ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Flagged corpus saved to {output_path}")

    # ── Build report ───────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("TEA BENCHMARK — BENGALI TRANSLATION QUALITY REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"\nThresholds applied:")
    report_lines.append(f"  clean            : bn_ratio >= {CLEAN_THRESHOLD}")
    report_lines.append(f"  mixed            : {MIXED_THRESHOLD} <= bn_ratio < {CLEAN_THRESHOLD}")
    report_lines.append(f"  english_retained : bn_ratio < {MIXED_THRESHOLD}")

    # Overall distribution
    report_lines.append("\n── Overall Distribution ──────────────────────────────")
    overall = df["bn_quality"].value_counts()
    for label in ["clean", "mixed", "english_retained"]:
        count = overall.get(label, 0)
        pct   = 100 * count / len(df)
        report_lines.append(f"  {label:<20}: {count:>3} items  ({pct:.1f}%)")
    report_lines.append(f"  {'TOTAL':<20}: {len(df):>3} items")

    # Per-tier breakdown
    report_lines.append("\n── Per-Tier Breakdown ────────────────────────────────")
    for tier in ["T1", "T2", "T3"]:
        tier_df = df[df["tier"] == tier]
        report_lines.append(f"\n  {tier} ({len(tier_df)} items):")
        tier_dist = tier_df["bn_quality"].value_counts()
        for label in ["clean", "mixed", "english_retained"]:
            count = tier_dist.get(label, 0)
            pct   = 100 * count / len(tier_df) if len(tier_df) > 0 else 0
            report_lines.append(f"    {label:<20}: {count:>3}  ({pct:.1f}%)")

    # Items flagged for sensitivity analysis
    flagged = df[df["bn_quality"] != "clean"]
    report_lines.append("\n── Items Excluded from Clean-Only Analysis ──────────")
    report_lines.append(f"  Total flagged (mixed + english_retained): {len(flagged)}")

    if len(flagged) > 0:
        report_lines.append(
            f"\n  {'Key':<10} {'Tier':<6} {'Quality':<20} {'Ratio':>6}  "
            f"Bengali text (first 60 chars)"
        )
        report_lines.append(f"  {'-'*9} {'-'*5} {'-'*19} {'-'*6}  {'-'*40}")
        for _, row in flagged.sort_values("bn_ratio").iterrows():
            preview = str(row["Bengali"])[:60].replace("\n", " ")
            report_lines.append(
                f"  {row['Key']:<10} {row['tier']:<6} {row['bn_quality']:<20} "
                f"{row['bn_ratio']:>6.3f}  {preview}"
            )

    # Ratio statistics
    report_lines.append("\n── Ratio Statistics ─────────────────────────────────")
    report_lines.append(f"  Mean bn_ratio  : {df['bn_ratio'].mean():.3f}")
    report_lines.append(f"  Median bn_ratio: {df['bn_ratio'].median():.3f}")
    report_lines.append(
        f"  Min bn_ratio   : {df['bn_ratio'].min():.3f}"
        f"  ({df.loc[df['bn_ratio'].idxmin(), 'Key']})"
    )
    report_lines.append(
        f"  Max bn_ratio   : {df['bn_ratio'].max():.3f}"
        f"  ({df.loc[df['bn_ratio'].idxmax(), 'Key']})"
    )

    report_lines.append("\n" + "=" * 60)
    report_lines.append("End of report")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")

    return df


if __name__ == "__main__":
    configure_console()
    # Resolve paths relative to the repository root,
    # regardless of where the script is called from.
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    result_df = run_quality_check(
        corpus_path="data/tea_corpus.csv",
        output_path="data/tea_corpus_flagged.csv",
        report_path="data/quality_report.txt"
    )

    from log_run import append_run_entry
    counts = result_df["bn_quality"].value_counts().to_dict()
    append_run_entry(
        script="01_quality_check.py",
        stats={
            "items":            len(result_df),
            "clean":            counts.get("clean", 0),
            "mixed":            counts.get("mixed", 0),
            "english_retained": counts.get("english_retained", 0),
        },
    )
