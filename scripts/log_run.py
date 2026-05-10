"""
log_run.py
----------
TEA Benchmark — Run Logger

Appends a dated entry to Notes.md whenever a pipeline script completes.
Called automatically at the end of each pipeline script, or manually:

    python scripts/log_run.py --script 02_tokenize.py --note "re-ran after corpus update"

Arguments (all optional)
------------------------
    --script   Name of the script that was run (default: "pipeline")
    --note     Free-text note to append (default: none)
    --stats    Key=value pairs to log, e.g. items=70 rows=1260
"""

import argparse
import re
from datetime import date
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
NOTES_FILE = REPO_ROOT / "Notes.md"


def append_run_entry(script: str = "pipeline", note: str = "", stats: dict = None):
    today = date.today().isoformat()

    lines = [f"\n## {today} — `{script}` run\n"]
    if note:
        lines.append(f"{note}\n")
    if stats:
        for k, v in stats.items():
            lines.append(f"- {k}: {v}\n")

    entry = "\n" + "".join(lines)

    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"Notes.md updated ({today} — {script})")


def _parse_stats(raw: list[str]) -> dict:
    out = {}
    for item in raw:
        if "=" in item:
            k, _, v = item.partition("=")
            out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append a run entry to Notes.md")
    parser.add_argument("--script", default="pipeline",  help="Script name")
    parser.add_argument("--note",   default="",           help="Free-text note")
    parser.add_argument("--stats",  nargs="*", default=[], help="key=value pairs")
    args = parser.parse_args()

    append_run_entry(
        script=args.script,
        note=args.note,
        stats=_parse_stats(args.stats),
    )
