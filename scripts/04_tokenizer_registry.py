"""
04_tokenizer_registry.py
------------------------
TEA Benchmark — Extension: Shared Tokenizer Registry

Purpose
-------
One place that defines every tokenizer used by the TEA extension scripts
(05_flores_open_weight.py, 06_hindi_script_share.py, 07_domain_comparison.py),
with a pinned revision for each, and a single way to load and count with them.

    load_counters(labels) -> {label: count_fn}
    write_manifest(path)  -> results/extension/tokenizer_manifest.csv

Counting follows the existing TEA scripts:
    tiktoken      len(enc.encode_ordinary(text))
    Hugging Face  len(tok.encode(text, add_special_tokens=False))

Every Hugging Face tokenizer is pinned to a commit SHA, resolved once with
huggingface_hub.HfApi().model_info(repo_id).sha and hard-coded below. tiktoken
encodings are fixed by the pinned tiktoken version in requirements.txt.

Network handling
----------------
Each download is retried with exponential backoff (5 attempts; waits of 5, 10,
20 and 40 seconds). A tokenizer that still fails, or a gated repository whose
access is refused, is recorded as "skipped" with the reason and the run
continues. Gated repositories (Llama 3.1, Gemma 3) need `huggingface-cli login`
(or `hf auth login`) and acceptance of the model licence on its Hugging Face page.

Local copies
------------
If data/tokenizers/<label>/ exists, that Hugging Face tokenizer is loaded from
the folder instead (AutoTokenizer.from_pretrained(<folder>)), and the manifest
source reads "local copy of <repo>@<revision>". The folder must hold the
tokenizer files downloaded at the pinned revision. data/tokenizers/ is
git-ignored and must never be committed.

Usage
-----
    # Load every registered tokenizer and write the manifest
    python scripts/04_tokenizer_registry.py

    # Offline check of the retry, skip and manifest logic
    python scripts/04_tokenizer_registry.py --self-test

From another script:

    registry = importlib.util ... (see load_registry() in 05_flores_open_weight.py)
    counters = registry.load_counters(registry.ALL_LABELS)
    registry.write_manifest(registry.MANIFEST_PATH)
"""

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve repository root (same convention as 02_tokenize.py) ─────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

MANIFEST_PATH = Path("results") / "extension" / "tokenizer_manifest.csv"

# Optional manually downloaded tokenizer files, one folder per label, e.g.
# data/tokenizers/gemma3_4b/. Used instead of a download when present.
# Git-ignored: model licences may not allow redistribution.
LOCAL_TOKENIZER_DIR = Path("data") / "tokenizers"

RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 5.0   # seconds; doubles after each failed attempt


# ── Registry ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TokenizerSpec:
    label: str
    kind: str          # "tiktoken" or "hf"
    source: str        # tiktoken encoding name or Hugging Face repo id
    revision: str      # commit SHA for HF; "pinned tiktoken package" for tiktoken
    note: str = ""


REGISTRY: list[TokenizerSpec] = [
    # OpenAI public encodings. o200k_base is the encoding TEA labels tiktoken_o200k.
    TokenizerSpec("r50k_base",     "tiktoken", "r50k_base",     "pinned tiktoken package",
                  "GPT-3 era encoding"),
    TokenizerSpec("cl100k_base",   "tiktoken", "cl100k_base",   "pinned tiktoken package",
                  "GPT-4 / GPT-3.5 encoding"),
    TokenizerSpec("o200k_base",    "tiktoken", "o200k_base",    "pinned tiktoken package",
                  "GPT-4o era encoding; same as TEA's tiktoken_o200k"),
    TokenizerSpec("o200k_harmony", "tiktoken", "o200k_harmony", "pinned tiktoken package",
                  "GPT-OSS encoding (o200k_base vocabulary plus harmony special tokens)"),

    # Open-weight base-model tokenizers on Hugging Face.
    TokenizerSpec("qwen2.5_7b", "hf", "Qwen/Qwen2.5-7B",
                  "d149729398750b98c0af14eb82c78cfe92750796",
                  "existing TEA pin (02_tokenize.py)"),
    TokenizerSpec("qwen3_8b", "hf", "Qwen/Qwen3-8B",
                  "b968826d9c46dd6066d109eabc6255188de91218"),
    TokenizerSpec("qwen3.8_2.4t", "hf", "Qwen/Qwen3.8-2.4T-A95B",
                  "4f885e51d6bbfee5866446d56d3f96f336c39723",
                  "Newest Qwen LLM family (Qwen3.8) at resolution time; the only text-generation "
                  "model of the family. Its tokenizer.json is byte-identical to those of "
                  "Qwen3.8-27B and Qwen3.8-Flash-Next. Loaded from data/tokenizers/."),
    TokenizerSpec("llama3.1_8b", "hf", "meta-llama/Llama-3.1-8B",
                  "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                  "gated: needs Hugging Face login and licence acceptance. "
                  "Loaded from data/tokenizers/."),
    TokenizerSpec("gemma3_4b", "hf", "google/gemma-3-4b-it",
                  "093f9f388b31de276ce2de164bdc2081324b9767",
                  "gated: needs Hugging Face login and licence acceptance"),
    TokenizerSpec("gemma4_e2b", "hf", "google/gemma-4-E2B-it",
                  "4adf15748de9e1a0097b6274cee9c15836c41c37",
                  "Smallest Gemma 4 model (E2B), instruct variant as for gemma3_4b. "
                  "Loaded from data/tokenizers/."),
    TokenizerSpec("mistral_7b_v0.1", "hf", "mistralai/Mistral-7B-v0.1",
                  "27d67f1b5f57dc0953326b2601d68371d40ea8da",
                  "existing TEA pin (02_tokenize.py)"),
    TokenizerSpec("mistral_nemo_tekken", "hf", "mistralai/Mistral-Nemo-Base-2407",
                  "a4477a2f977929a969745b69bbd62e03043551a5",
                  "Tekken tokenizer"),
    TokenizerSpec("deepseek_v3", "hf", "deepseek-ai/DeepSeek-V3",
                  "e815299b0bcbac849fa540c768ef21845365c9eb"),
    TokenizerSpec("nemotron", "hf", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16",
                  "434456c9a6753f29d24e23c95d622aaf17111b3b",
                  "Newest ungated NVIDIA Nemotron text base model on Hugging Face at "
                  "resolution time (created 2026-08-05). The newer "
                  "Nemotron-3.5-Super-VL-120B-A12B-Base is image-text-to-text and ships "
                  "a multimodal processor, and Nemotron-Labs repos are research variants, "
                  "so neither was used."),
]

SPECS = {s.label: s for s in REGISTRY}
ALL_LABELS = [s.label for s in REGISTRY]

# Manifest rows, filled in by load_counters(); one per attempted label.
_STATUS: dict[str, dict] = {}


# ── Console ──────────────────────────────────────────────────────────────────

def configure_console() -> None:
    """Use UTF-8 for multilingual output on Windows (as 01_quality_check.py does)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# ── Retry with exponential backoff ───────────────────────────────────────────

class NonRetryableError(Exception):
    """Access refused (gated or missing repository): retrying will not help."""


def _is_access_refused(exc: BaseException) -> bool:
    """True if the error, or anything in its cause chain, is a refusal of access."""
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
        access_types = (GatedRepoError, RepositoryNotFoundError)
    except Exception:
        access_types = ()
    markers = ("gated repo", "gated", "401", "403", "access to model",
               "awaiting a decision", "not in the authorized list")
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if access_types and isinstance(exc, access_types):
            return True
        if isinstance(exc, NonRetryableError):
            return True
        msg = str(exc).lower()
        if any(m in msg for m in markers):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def retry_with_backoff(fn, attempts: int = RETRY_ATTEMPTS,
                       base_delay: float = RETRY_BASE_DELAY,
                       sleep=time.sleep, log=print):
    """
    Call fn() until it succeeds, up to `attempts` times, waiting base_delay,
    2*base_delay, 4*base_delay, ... between attempts. Access refusals are not
    retried. Re-raises the last error when every attempt fails.
    """
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any download error is retried
            if _is_access_refused(exc):
                raise NonRetryableError(_short_reason(exc)) from exc
            if attempt == attempts:
                raise
            log(f"    attempt {attempt}/{attempts} failed ({_short_reason(exc)}); "
                f"retrying in {delay:.0f}s")
            sleep(delay)
            delay *= 2


def _short_reason(exc: BaseException, limit: int = 200) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_tiktoken(spec: TokenizerSpec):
    import tiktoken
    if spec.source not in tiktoken.list_encoding_names():
        raise NonRetryableError(
            f"tiktoken {tiktoken.__version__} does not provide '{spec.source}'")
    enc = retry_with_backoff(lambda: tiktoken.get_encoding(spec.source))
    count_fn = (lambda text, e=enc: len(e.encode_ordinary(text)))
    info = {
        "vocab_size":      enc.n_vocab,
        "tokenizer_type":  "BPE (tiktoken)",
        "library_version": f"tiktoken {tiktoken.__version__}",
    }
    return count_fn, info


def local_tokenizer_dir(label: str):
    """data/tokenizers/<label>/ if it exists (manually downloaded files; git-ignored)."""
    d = LOCAL_TOKENIZER_DIR / label
    return d if d.is_dir() else None


def _load_hf(spec: TokenizerSpec):
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    local = local_tokenizer_dir(spec.label)
    if local is not None:
        # Files downloaded by hand at spec.revision; nothing is fetched.
        tok = AutoTokenizer.from_pretrained(str(local), use_fast=True)
    else:
        tok = retry_with_backoff(lambda: AutoTokenizer.from_pretrained(
            spec.source, revision=spec.revision, use_fast=True, trust_remote_code=True))
    count_fn = (lambda text, t=tok: len(t.encode(text, add_special_tokens=False)))
    model_type = ""
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is not None:
        model_type = type(backend.model).__name__
    info = {
        "vocab_size":      len(tok),
        "tokenizer_type":  f"{type(tok).__name__}" + (f" ({model_type})" if model_type else ""),
        "library_version": f"transformers {transformers.__version__}; "
                           f"tokenizers {tokenizers.__version__}",
    }
    if local is not None:
        info["source"] = f"local copy of {spec.source}@{spec.revision}"
    return count_fn, info


def load_counters(labels=None) -> dict:
    """
    Load the requested tokenizers and return {label: count_fn} for those that
    loaded. Failures are recorded as skipped (see write_manifest) and never
    stop the run.
    """
    labels = ALL_LABELS if labels is None else list(labels)
    counters = {}
    for label in labels:
        spec = SPECS[label]
        print(f"  Loading {label} ({spec.source}) ...")
        row = {
            "label": label, "source": spec.source, "revision": spec.revision,
            "vocab_size": "", "tokenizer_type": "", "library_version": "",
            "status": "skipped", "skip_reason": "", "note": spec.note,
        }
        try:
            loader = _load_tiktoken if spec.kind == "tiktoken" else _load_hf
            count_fn, info = loader(spec)
            # Smoke test: a short English string must produce at least one token.
            if count_fn("Hello world") < 1:
                raise RuntimeError("tokenizer returned no tokens for 'Hello world'")
            counters[label] = count_fn
            row.update(info, status="ran")
            print(f"    ok: vocab {info['vocab_size']}, {info['tokenizer_type']}")
        except NonRetryableError as exc:
            row["skip_reason"] = f"access refused or unavailable: {exc}"
            row["library_version"] = _library_version(spec.kind)
            print(f"    SKIPPED: {row['skip_reason']}")
        except Exception as exc:  # noqa: BLE001 - never stop the whole run
            row["skip_reason"] = (f"failed after {RETRY_ATTEMPTS} attempts: "
                                  f"{_short_reason(exc)}")
            row["library_version"] = _library_version(spec.kind)
            print(f"    SKIPPED: {row['skip_reason']}")
        _STATUS[label] = row
    return counters


def _library_version(kind: str) -> str:
    try:
        if kind == "tiktoken":
            import tiktoken
            return f"tiktoken {tiktoken.__version__}"
        import tokenizers
        import transformers
        return f"transformers {transformers.__version__}; tokenizers {tokenizers.__version__}"
    except Exception:
        return ""


MANIFEST_COLUMNS = ["label", "source", "revision", "vocab_size", "tokenizer_type",
                    "library_version", "status", "skip_reason", "note"]


def manifest_frame() -> pd.DataFrame:
    """Manifest rows in registry order, for every label load_counters() attempted."""
    rows = [_STATUS[l] for l in ALL_LABELS if l in _STATUS]
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = manifest_frame()
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def ran_labels(path: Path = MANIFEST_PATH) -> list[str]:
    """Labels marked 'ran' in a written manifest (used by Tasks 2 and 3)."""
    path = Path(path)
    if not path.is_file():
        sys.exit(f"\nTokenizer manifest not found at {path}. "
                 "Run scripts/05_flores_open_weight.py first.\n")
    m = pd.read_csv(path, encoding="utf-8")
    return [l for l in m.loc[m["status"] == "ran", "label"] if l in SPECS]


def print_manifest(df: pd.DataFrame) -> None:
    print("\nTOKENIZER MANIFEST")
    for _, r in df.iterrows():
        extra = f"  ({r['skip_reason']})" if r["status"] != "ran" else ""
        vocab = r["vocab_size"] if r["vocab_size"] != "" else "-"
        print(f"  {r['label']:<20} {r['status']:<8} vocab={vocab!s:<8} {r['source']}{extra}")


# ── Self-test (no network) ───────────────────────────────────────────────────

def self_test():
    print("Running self-test (no network) ...")
    waits = []

    # 1. Fails twice, then succeeds: three calls, waits of 5 s then 10 s.
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionResetError("connection reset by peer")
        return "ok"

    assert retry_with_backoff(flaky, sleep=waits.append, log=lambda *_: None) == "ok"
    assert calls["n"] == 3, calls
    assert waits == [5.0, 10.0], waits

    # 2. Always fails: 5 attempts, waits 5, 10, 20, 40 s, then the error is raised.
    waits.clear()
    calls["n"] = 0

    def broken():
        calls["n"] += 1
        raise TimeoutError("read timed out")

    try:
        retry_with_backoff(broken, sleep=waits.append, log=lambda *_: None)
        raise AssertionError("expected TimeoutError")
    except TimeoutError:
        pass
    assert calls["n"] == 5, calls
    assert waits == [5.0, 10.0, 20.0, 40.0], waits

    # 3. Gated repository: refused once, not retried.
    waits.clear()
    calls["n"] = 0

    def gated():
        calls["n"] += 1
        raise OSError("You are trying to access a gated repo. 401 Client Error")

    try:
        retry_with_backoff(gated, sleep=waits.append, log=lambda *_: None)
        raise AssertionError("expected NonRetryableError")
    except NonRetryableError:
        pass
    assert calls["n"] == 1 and waits == [], (calls, waits)

    # 4. Manifest: one ran, one skipped, written and read back.
    _STATUS.clear()
    _STATUS["o200k_base"] = {"label": "o200k_base", "source": "o200k_base",
                             "revision": "pinned tiktoken package", "vocab_size": 200019,
                             "tokenizer_type": "BPE (tiktoken)", "library_version": "x",
                             "status": "ran", "skip_reason": "", "note": ""}
    _STATUS["llama3.1_8b"] = {"label": "llama3.1_8b", "source": "meta-llama/Llama-3.1-8B",
                              "revision": SPECS["llama3.1_8b"].revision, "vocab_size": "",
                              "tokenizer_type": "", "library_version": "x",
                              "status": "skipped", "skip_reason": "gated", "note": ""}
    tmp = Path("results") / "extension" / "_selftest_manifest.csv"
    try:
        write_manifest(tmp)
        assert ran_labels(tmp) == ["o200k_base"], ran_labels(tmp)
        back = pd.read_csv(tmp, encoding="utf-8")
        assert list(back.columns) == MANIFEST_COLUMNS
        assert list(back["status"]) == ["ran", "skipped"]
    finally:
        tmp.unlink(missing_ok=True)
        _STATUS.clear()

    # 5. Every HF entry is pinned to a 40-character commit SHA.
    for s in REGISTRY:
        if s.kind == "hf":
            assert len(s.revision) == 40 and all(c in "0123456789abcdef" for c in s.revision), s
    print("Self-test passed: retry backoff (5, 10, 20, 40 s), no retry on gated access, "
          "manifest round-trip, and SHA pins all behave as expected.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", default=ALL_LABELS, choices=ALL_LABELS,
                        help="Tokenizers to load (default: all registered)")
    parser.add_argument("--self-test", action="store_true",
                        help="Check the retry and manifest logic offline and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return None

    print("\nLoading registered tokenizers ...")
    counters = load_counters(args.labels)
    manifest = write_manifest(MANIFEST_PATH)
    print_manifest(manifest)
    print(f"\nManifest written to {MANIFEST_PATH}")
    return {"registered": len(args.labels), "ran": len(counters),
            "skipped": len(args.labels) - len(counters)}


if __name__ == "__main__":
    configure_console()
    stats = main()
    if stats is not None:
        from log_run import append_run_entry
        append_run_entry(
            script="04_tokenizer_registry.py",
            note="Loaded registered extension tokenizers and wrote the manifest.",
            stats=stats,
        )
