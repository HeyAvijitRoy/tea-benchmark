"""
Generate per-language tokenizer segmentation examples using GPT-4o o200k_base.
Prints token splits, writes a CSV to results/, and prints a LaTeX table.
"""

import csv
import os
import sys
import tiktoken

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENC = tiktoken.get_encoding("o200k_base")

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tea_corpus.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
LANGUAGES = ["English", "Bengali", "Hindi", "Arabic", "Tamil", "Yoruba"]


def configure_console() -> None:
    """Use UTF-8 for multilingual output on Windows and compatible terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


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


def write_figure(key: str, table_rows: list[dict]) -> str:
    """Write an auditable token-count figure from the computed rows."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "tokenizer_fragmentation_example.png")

    languages = [row["lang"] for row in table_rows]
    counts = [len(row["pieces"]) for row in table_rows]
    colors = ["#0072B2"] + ["#E69F00"] * (len(table_rows) - 1)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    bars = ax.barh(languages, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Token count")
    ax.set_title(f"GPT-4o o200k_base token counts for example {key}")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.bar_label(bars, labels=[str(value) for value in counts], padding=3, fontsize=8.5)
    ax.set_xlim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
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

    figure_path = write_figure(key, table_rows)
    print(f"Figure written to: {figure_path}\n")

    print("=" * 70)
    print("LaTeX table:\n")
    print(build_latex_table(table_rows))


if __name__ == "__main__":
    configure_console()
    main()
