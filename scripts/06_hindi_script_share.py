"""
06_hindi_script_share.py
------------------------
TEA Benchmark — Extension Task 2: Hindi Script-Share Sensitivity Analysis

Purpose
-------
TEA reports a code-switching sensitivity analysis for Bengali only (clean-only
items, 01_quality_check.py). This script applies the same procedure to Hindi:

    hi_script_share = Devanagari characters / (Devanagari + Latin alphabetic characters)

with the thresholds and labels of 01_quality_check.py (imported, not copied):

    clean            : share >= 0.75
    mixed            : 0.40 <= share < 0.75
    english_retained : share < 0.40

Devanagari characters are those in U+0900–U+097F and U+A8E0–U+A8FF whose
Unicode category is a letter (L*) or a combining mark (M*, the vowel signs and
virama that are part of written syllables). Digits and punctuation in those
ranges (in the TEA corpus, only the danda U+0964) are excluded, as are all
non-alphabetic Latin characters, so only alphabetic text is counted. The
script also reports how many labels would change if every code point in the
ranges were counted, which is how 01_quality_check.py treats the Bengali block.

It then computes per-item Hindi TFR (Hindi tokens / English tokens for the
same item) exactly as 02_tokenize.py does, for every tokenizer that ran in
Task 1, and compares all items with clean-only items, overall and by tier.

Usage
-----
    python scripts/06_hindi_script_share.py
    python scripts/06_hindi_script_share.py --self-test

Outputs
-------
    data/extension/tea_hindi_flagged.csv                 Key, tier, hi_script_share, hi_quality
    results/extension/hindi_script_share/summary.csv
    results/extension/hindi_script_share/item_level.csv
    results/extension/hindi_script_share/label_distribution.csv
    results/extension/hindi_script_share/review_table.md
"""

import argparse
import importlib.util
import math
import os
import sys
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve repository root (same convention as 02_tokenize.py) ─────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


def load_script_module(name: str, filename: str):
    """Import a numbered script (e.g. 01_quality_check.py) as a module."""
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quality = load_script_module("quality_check", "01_quality_check.py")
tea_tokenize = load_script_module("tea_tokenize", "02_tokenize.py")
registry = load_script_module("tokenizer_registry", "04_tokenizer_registry.py")

DEVANAGARI_RANGES = ((0x0900, 0x097F), (0xA8E0, 0xA8FF))
LABELS = ["clean", "mixed", "english_retained"]
TIERS = ["T1", "T2", "T3"]

CORPUS_PATH = Path("data") / "tea_corpus_flagged.csv"
FLAGGED_PATH = Path("data") / "extension" / "tea_hindi_flagged.csv"
OUT_DIR = Path("results") / "extension" / "hindi_script_share"

# TEA's published per-item TFRs use these labels for the same tokenizers.
TEA_LABEL = {"o200k_base": "tiktoken_o200k", "qwen2.5_7b": "qwen2.5_7b",
             "mistral_7b_v0.1": "mistral_7b_v0.1"}


# ── Script share ─────────────────────────────────────────────────────────────

def in_devanagari_ranges(char: str) -> bool:
    o = ord(char)
    return any(lo <= o <= hi for lo, hi in DEVANAGARI_RANGES)


def is_devanagari_alpha(char: str) -> bool:
    """Devanagari letter or combining mark (vowel sign, virama, nukta, ...)."""
    return in_devanagari_ranges(char) and unicodedata.category(char)[0] in ("L", "M")


def compute_hindi_share(text: str, whole_block: bool = False) -> float:
    """
    Devanagari share of alphabetic characters, following
    01_quality_check.compute_bengali_ratio: Latin letters are counted with the
    same is_latin_alpha() test, everything else is ignored, and a string with no
    alphabetic content returns 0.0. With whole_block=True every code point in
    the Devanagari ranges counts (the 01_quality_check treatment of Bengali).
    """
    is_script = in_devanagari_ranges if whole_block else is_devanagari_alpha
    script_count = sum(1 for c in text if is_script(c))
    latin_count = sum(1 for c in text if quality.is_latin_alpha(c))
    total = script_count + latin_count
    return 0.0 if total == 0 else script_count / total


def flag_hindi(df: pd.DataFrame) -> pd.DataFrame:
    text = df["Hindi"].apply(lambda x: str(x) if pd.notna(x) else "")
    out = pd.DataFrame({
        "Key":  df["Key"],
        "tier": df["tier"],
        "hi_script_share": text.apply(compute_hindi_share),
    })
    out["hi_quality"] = out["hi_script_share"].apply(quality.assign_quality_label)
    block_share = text.apply(lambda t: compute_hindi_share(t, whole_block=True))
    out.attrs["labels_changed_whole_block"] = int(
        (block_share.apply(quality.assign_quality_label) != out["hi_quality"]).sum())
    return out


def label_distribution(flagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub in [("all", flagged)] + [(t, flagged[flagged["tier"] == t]) for t in TIERS]:
        counts = sub["hi_quality"].value_counts()
        for label in LABELS:
            n = int(counts.get(label, 0))
            rows.append({"scope": scope, "hi_quality": label, "n_items": n,
                         "pct": round(100 * n / len(sub), 1) if len(sub) else 0.0})
    return pd.DataFrame(rows)


# ── Per-item TFR, exactly as 02_tokenize.py ─────────────────────────────────

def compute_item_tfr(df: pd.DataFrame, counters: dict, languages: list[str]) -> pd.DataFrame:
    """
    For each item, tokenizer and language: token counts via 02_tokenize.safe_count
    (NaN for missing or empty text) and TFR = tokens_L / tokens_EN, NaN when
    either count is NaN or English has 0 tokens. Shared with 07_domain_comparison.py.
    """
    records = []
    for tok_name, count_fn in counters.items():
        print(f"  Tokenizing TEA corpus with {tok_name} ...")
        for _, row in df.iterrows():
            en = tea_tokenize.safe_count(count_fn, row.get("English"))
            for lang in languages:
                cnt = tea_tokenize.safe_count(count_fn, row.get(lang))
                if math.isnan(en) or math.isnan(cnt) or en == 0:
                    tfr = float("nan")
                else:
                    tfr = cnt / en
                records.append({
                    "key": row["Key"], "tier": row["tier"], "language": lang,
                    "tokenizer": tok_name, "token_count": cnt,
                    "en_token_count": en, "tfr": tfr,
                })
    return pd.DataFrame(records)


def summarize_subsets(items: pd.DataFrame) -> pd.DataFrame:
    """mean / median / SD / n of Hindi TFR, all items vs clean-only, overall and by tier."""
    rows = []
    for tok, tdf in items.groupby("tokenizer", sort=False):
        for subset, sdf in (("all", tdf), ("clean_only", tdf[tdf["hi_quality"] == "clean"])):
            for tier in ["all"] + TIERS:
                vals = (sdf if tier == "all" else sdf[sdf["tier"] == tier])["tfr"].dropna()
                rows.append({
                    "tokenizer": tok, "subset": subset, "tier": tier,
                    "mean_tfr": vals.mean(), "median_tfr": vals.median(),
                    "std_tfr": vals.std(), "n_items": len(vals),
                })
    return pd.DataFrame(rows)


def check_against_published(items: pd.DataFrame) -> list[str]:
    """Compare per-item Hindi TFRs with TEA's results/tfr_by_item.csv where labels overlap."""
    path = Path("results") / "tfr_by_item.csv"
    if not path.is_file():
        return ["results/tfr_by_item.csv not found; skipped comparison"]
    pub = pd.read_csv(path, encoding="utf-8")
    pub = pub[pub["language"] == "Hindi"]
    lines = []
    for ours, theirs in TEA_LABEL.items():
        a = items[items["tokenizer"] == ours].set_index("key")["tfr"]
        b = pub[pub["tokenizer"] == theirs].set_index("key")["tfr"]
        if a.empty or b.empty:
            continue
        j = pd.concat([a, b], axis=1, keys=["ours", "tea"], join="inner")
        diff = (j["ours"] - j["tea"]).abs().max()
        lines.append(f"{ours} vs TEA {theirs}: {len(j)} items, max |difference| = {diff:.2e}")
    return lines


# ── Output ───────────────────────────────────────────────────────────────────

def write_review_table(summary: pd.DataFrame, dist: pd.DataFrame,
                       changed: int, path: Path) -> None:
    overall = dist[dist["scope"] == "all"].set_index("hi_quality")
    n_total = int(overall["n_items"].sum())
    lines = [
        "**Hindi label distribution** (TEA corpus, " f"{n_total} items)",
        "",
        "| Label | Items | Share | T1 | T2 | T3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        tiers = [int(dist[(dist["scope"] == t) & (dist["hi_quality"] == label)]["n_items"].iloc[0])
                 for t in TIERS]
        lines.append(f"| {label} | {int(overall.loc[label, 'n_items'])} "
                     f"| {overall.loc[label, 'pct']:.1f}% | {tiers[0]} | {tiers[1]} | {tiers[2]} |")
    lines += [
        "",
        "**Hindi TFR, all items vs clean-only** (mean / median, n)",
        "",
        "| Tokenizer | All items | Clean-only | Clean − all (mean) |",
        "|---|---:|---:|---:|",
    ]
    ov = summary[summary["tier"] == "all"]
    for tok in ov["tokenizer"].unique():
        a = ov[(ov["tokenizer"] == tok) & (ov["subset"] == "all")].iloc[0]
        c = ov[(ov["tokenizer"] == tok) & (ov["subset"] == "clean_only")].iloc[0]
        lines.append(
            f"| {tok} | {a['mean_tfr']:.2f}× / {a['median_tfr']:.2f}× (n={a['n_items']}) "
            f"| {c['mean_tfr']:.2f}× / {c['median_tfr']:.2f}× (n={c['n_items']}) "
            f"| {c['mean_tfr'] - a['mean_tfr']:+.2f} |")
    lines += [
        "",
        "Script share = Devanagari letters and combining marks / (those + Latin letters); "
        "labels and thresholds as in `01_quality_check.py` (clean ≥ 0.75, mixed 0.40–0.75, "
        "english_retained < 0.40). Per-item TFR = Hindi tokens / English tokens, computed as "
        "in `02_tokenize.py`. By-tier values are in `summary.csv`. Counting every code point "
        f"in the Devanagari ranges (including the danda) would change {changed} label(s).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_results(dist: pd.DataFrame, summary: pd.DataFrame, changed: int) -> None:
    print("\nHINDI LABEL DISTRIBUTION")
    for scope in ["all"] + TIERS:
        sub = dist[dist["scope"] == scope]
        parts = ", ".join(f"{r.hi_quality} {r.n_items} ({r.pct:.1f}%)" for r in sub.itertuples())
        print(f"  {scope:<4}: {parts}")
    print(f"  Labels that would change if the whole Devanagari block were counted: {changed}")

    print("\n" + "=" * 78)
    print("HINDI TFR: ALL ITEMS VS CLEAN-ONLY (overall)")
    print("=" * 78)
    print(f"{'Tokenizer':<21}{'All mean':>10}{'median':>8}{'n':>5}"
          f"{'Clean mean':>12}{'median':>8}{'n':>5}{'Diff':>8}")
    ov = summary[summary["tier"] == "all"]
    for tok in ov["tokenizer"].unique():
        a = ov[(ov["tokenizer"] == tok) & (ov["subset"] == "all")].iloc[0]
        c = ov[(ov["tokenizer"] == tok) & (ov["subset"] == "clean_only")].iloc[0]
        print(f"{tok:<21}{a['mean_tfr']:>10.3f}{a['median_tfr']:>8.3f}{a['n_items']:>5}"
              f"{c['mean_tfr']:>12.3f}{c['median_tfr']:>8.3f}{c['n_items']:>5}"
              f"{c['mean_tfr'] - a['mean_tfr']:>+8.3f}")
    print("=" * 78)


# ── Self-test (no network, no tokenizers) ────────────────────────────────────

def self_test():
    print("Running self-test on synthetic data ...")
    # "नमस्ते" = न म स ् त े : 4 letters + virama (Mn) + vowel sign (Mc) = 6 counted
    assert math.isclose(compute_hindi_share("नमस्ते"), 1.0)
    assert math.isclose(compute_hindi_share("नमस्ते abc"), 6 / 9)           # 0.667 -> mixed
    assert math.isclose(compute_hindi_share("नमस्ते।"), 1.0)                 # danda ignored
    assert math.isclose(compute_hindi_share("। abc"), 0.0)                   # danda ignored
    assert math.isclose(compute_hindi_share("। abc", whole_block=True), 0.25)  # 1 / (1 + 3)
    assert math.isclose(compute_hindi_share("१२३ abc"), 0.0)                 # Devanagari digits ignored
    assert compute_hindi_share("123 ()") == 0.0                              # nothing alphabetic
    assert math.isclose(compute_hindi_share("ꣲ a"), 0.5)                   # U+A8F2 (Devanagari Extended, Lo)
    # 01_quality_check thresholds
    assert quality.assign_quality_label(0.75) == "clean"
    assert quality.assign_quality_label(6 / 9) == "mixed"
    assert quality.assign_quality_label(0.40) == "mixed"
    assert quality.assign_quality_label(0.39) == "english_retained"

    # TFR and subset summary with a character-count "tokenizer":
    #   item  tier  EN       HI            tokens EN/HI  TFR   label
    #   A     T1    "ab"     "कखगघ"        2 / 4         2.0   clean
    #   B     T1    "abcd"   "कख abcd"     4 / 7         1.75  english_retained (2/6 = 0.33)
    #   C     T2    "abcd"   "कखगघङच"      4 / 6         1.5   clean
    #   D     T2    ""       "कख"          NaN           NaN   clean
    df = pd.DataFrame({
        "Key": ["A", "B", "C", "D"], "tier": ["T1", "T1", "T2", "T2"],
        "English": ["ab", "abcd", "abcd", ""],
        "Hindi": ["कखगघ", "कख abcd", "कखगघङच", "कख"],
    })
    flagged = flag_hindi(df)
    assert list(flagged["hi_quality"]) == ["clean", "english_retained", "clean", "clean"]
    items = compute_item_tfr(df, {"chars": len}, ["Hindi"])
    items = items.merge(flagged[["Key", "hi_quality"]], left_on="key", right_on="Key")
    assert list(items["token_count"]) == [4, 7, 6, 2]
    assert math.isclose(items["tfr"].iloc[1], 7 / 4) and math.isnan(items["tfr"].iloc[3])
    s = summarize_subsets(items)
    get = lambda sub, tier: s[(s["subset"] == sub) & (s["tier"] == tier)].iloc[0]
    assert get("all", "all")["n_items"] == 3                                 # NaN dropped
    assert math.isclose(get("all", "all")["mean_tfr"], (2.0 + 1.75 + 1.5) / 3)  # 1.75
    assert math.isclose(get("clean_only", "all")["mean_tfr"], 1.75)          # (2.0 + 1.5) / 2
    assert get("clean_only", "all")["n_items"] == 2
    assert math.isclose(get("clean_only", "T1")["mean_tfr"], 2.0)
    assert math.isclose(get("all", "T1")["std_tfr"], math.sqrt(0.03125))    # SD of (2.0, 1.75)
    print("Self-test passed: script share (letters, marks, danda, digits, Extended block), "
          "01_quality_check thresholds, per-item TFR with NaN handling, and subset means, "
          "medians, SDs and counts all match hand-computed values.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", default=None, choices=registry.ALL_LABELS,
                        help="Tokenizers to run (default: those marked 'ran' in the manifest)")
    parser.add_argument("--self-test", action="store_true",
                        help="Check the logic on synthetic data and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return None

    print(f"\nLoading corpus from {CORPUS_PATH} ...")
    df = pd.read_csv(CORPUS_PATH, encoding="utf-8")
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
    print(f"  {len(df)} items loaded")

    print("\nComputing Hindi script share ...")
    flagged = flag_hindi(df)
    changed = flagged.attrs["labels_changed_whole_block"]
    FLAGGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(FLAGGED_PATH, index=False, encoding="utf-8")
    dist = label_distribution(flagged)

    labels = args.labels or registry.ran_labels(registry.MANIFEST_PATH)
    print(f"\nLoading tokenizers that ran in Task 1: {', '.join(labels)}")
    counters = registry.load_counters(labels)
    missing = [l for l in labels if l not in counters]
    if missing:
        print(f"  WARNING: could not load {missing}; continuing without them")

    print("\nComputing per-item Hindi TFR ...")
    items = compute_item_tfr(df, counters, ["Hindi"])
    items = items.merge(flagged[["Key", "hi_script_share", "hi_quality"]],
                        left_on="key", right_on="Key").drop(columns=["Key", "language"])
    items = items.rename(columns={"token_count": "hi_token_count"})[
        ["key", "tier", "hi_script_share", "hi_quality", "tokenizer",
         "hi_token_count", "en_token_count", "tfr"]]
    summary = summarize_subsets(items)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items.to_csv(OUT_DIR / "item_level.csv", index=False, encoding="utf-8")
    summary.round(4).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8")
    dist.to_csv(OUT_DIR / "label_distribution.csv", index=False, encoding="utf-8")
    write_review_table(summary, dist, changed, OUT_DIR / "review_table.md")

    print_results(dist, summary, changed)
    print("\nCheck against TEA's published per-item Hindi TFRs:")
    for line in check_against_published(items):
        print(f"  {line}")

    print("\nOutput files written:")
    for p in [FLAGGED_PATH, OUT_DIR / "summary.csv", OUT_DIR / "item_level.csv",
              OUT_DIR / "label_distribution.csv", OUT_DIR / "review_table.md"]:
        print(f"  {p}")

    counts = flagged["hi_quality"].value_counts()
    return {
        "items":            len(df),
        "clean":            int(counts.get("clean", 0)),
        "mixed":            int(counts.get("mixed", 0)),
        "english_retained": int(counts.get("english_retained", 0)),
        "tokenizers":       len(counters),
    }


if __name__ == "__main__":
    registry.configure_console()
    stats = main()
    if stats is not None:
        from log_run import append_run_entry
        append_run_entry(
            script="06_hindi_script_share.py",
            note="Hindi script-share labels and all-items vs clean-only TFR on the TEA corpus.",
            stats=stats,
        )
