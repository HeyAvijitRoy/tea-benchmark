"""
Generate per-language tokenizer segmentation examples using GPT-4o o200k_base.
Prints token splits, writes a CSV to results/, and prints a LaTeX table.
"""

import csv
import os
import tiktoken

ENC = tiktoken.get_encoding("o200k_base")

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tea_corpus.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LANGUAGES = ["English", "Bengali", "Hindi", "Arabic", "Tamil", "Yoruba"]


def tokenize(text: str) -> list[str]:
    return [ENC.decode([t]) for t in ENC.encode(text)]


def fmt_tokens(pieces: list[str], max_show: int = 6) -> str:
    shown = pieces[:max_show]
    parts = ", ".join(f'"{p}"' for p in shown)
    suffix = ", ..." if len(pieces) > max_show else ""
    return f"[{parts}{suffix}]"


def load_all_keys(csv_path: str) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [row["Key"] for row in csv.DictReader(f) if row.get("Key")]


def load_row(csv_path: str, key: str) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Key"] == key:
                return row
    raise ValueError(f"Key {key!r} not found in {csv_path}")


def pick_key(csv_path: str) -> str:
    keys = load_all_keys(csv_path)
    print(f"Available row keys ({len(keys)} total): {', '.join(keys[:10])}", end="")
    print(" ..." if len(keys) > 10 else "")
    while True:
        choice = input("Enter row key (e.g. T1-10): ").strip()
        if choice in keys:
            return choice
        print(f"  '{choice}' not found. Try again.")


def escape_latex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def build_latex_table(rows: list[dict]) -> str:
    header = r"""\begin{table*}[t]
\centering
\caption{Illustrative tokenizer segmentation differences under GPT-4o \texttt{o200k\_base}.}
\label{tab:token_split}
\small
\begin{tabular}{p{1.5cm} p{4.5cm} p{6.5cm}}
\toprule
\textbf{Language} & \textbf{Input text} & \textbf{Example tokenizer segmentation} \\
\midrule"""

    lines = [header]
    for i, row in enumerate(rows):
        if i > 0:
            lines.append(r"\midrule")
        seg = fmt_tokens(row["pieces"])
        lines.append(
            f"{row['lang']} &\n"
            f"\\texttt{{{escape_latex(row['text'])}}} &\n"
            f"\\texttt{{{escape_latex(seg)}}} \\\\"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}")
    return "\n".join(lines)


def write_csv(key: str, table_rows: list[dict]) -> str:
    out_path = os.path.join(RESULTS_DIR, f"token_split_{key}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "language", "text", "token_count", "token_ids", "segments"])
        for row in table_rows:
            ids = ENC.encode(row["text"])
            writer.writerow([
                key,
                row["lang"],
                row["text"],
                len(row["pieces"]),
                ids,
                row["pieces"],
            ])
    return out_path


def main():
    key = pick_key(CORPUS_PATH)
    corpus_row = load_row(CORPUS_PATH, key)

    print(f"\nRow key: {key}\n")

    table_rows = []
    for lang in LANGUAGES:
        text = corpus_row.get(lang, "").strip()
        if not text:
            continue
        pieces = tokenize(text)
        table_rows.append({"lang": lang, "text": text, "pieces": pieces})

        print(f"Language : {lang}")
        print(f"Text     : {text}")
        print(f"Token IDs: {ENC.encode(text)}")
        print(f"Segments : {pieces}")
        print(f"Count    : {len(pieces)} tokens")
        print()

    out_path = write_csv(key, table_rows)
    print(f"CSV written to: {out_path}\n")

    print("=" * 70)
    print("LaTeX table:\n")
    print(build_latex_table(table_rows))


if __name__ == "__main__":
    main()
