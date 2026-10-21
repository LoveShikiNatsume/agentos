"""Prepare real SWE-bench Lite tasks as a multi-agent scenario corpus.

We use SWE-bench Lite (300 consensus, real-world GitHub issues across
astropy/django/sympy/...) in its *oracle* variant, whose ``text`` field already
bundles the issue statement + the relevant (gold) source files. That gives a
realistic, *large shared repo context* (median ~10.5k tokens) without needing
Docker or running any test suite.

IMPORTANT: We are NOT scoring correctness (% resolved). We only borrow these
tasks as a realistic, recognized *workload* to measure our Runtime's systems
efficiency (token / cache / cost / fault-recovery). No patch is validated.

Output: runs/swe_bench/tasks.json — a small filtered+sampled subset whose shared
context fits comfortably in DeepSeek's window and keeps API cost modest.

Usage:
    .venv/bin/python scripts/swe_bench/prepare_swe_tasks.py \
        --n 10 --min-tok 3000 --max-tok 12000 --out runs/swe_bench/tasks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHARS_PER_TOKEN = 4.0  # rough estimate, only used for filtering/sampling


def _load_oracle_rows(cache_dir: Path) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "princeton-nlp/SWE-bench_Lite_oracle",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return pq.read_table(path).to_pylist()


def _est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def prepare(n: int, min_tok: int, max_tok: int, out: Path, cache_dir: Path) -> None:
    rows = _load_oracle_rows(cache_dir)
    print(f"[load] {len(rows)} SWE-bench Lite (oracle) tasks")

    # Filter to a manageable shared-context size window.
    eligible = []
    for r in rows:
        text = r.get("text", "") or ""
        tok = _est_tokens(text)
        if min_tok <= tok <= max_tok:
            eligible.append((tok, r))
    eligible.sort(key=lambda x: x[0])
    print(f"[filter] {len(eligible)} tasks with shared context in "
          f"[{min_tok}, {max_tok}] est tokens")

    if not eligible:
        raise SystemExit("no eligible tasks in token window; widen --min/--max-tok")

    # Even spread across the size window (not all tiny / all huge), deterministic.
    step = max(1, len(eligible) // n)
    picked = [eligible[i] for i in range(0, len(eligible), step)][:n]

    tasks = []
    for tok, r in picked:
        tasks.append({
            "instance_id": r["instance_id"],
            "repo": r["repo"],
            "base_commit": r.get("base_commit", ""),
            "problem_statement": r.get("problem_statement", "") or "",
            "shared_context": r.get("text", "") or "",   # issue + oracle code
            "est_context_tokens": tok,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "princeton-nlp/SWE-bench_Lite_oracle (test split)",
        "note": "Real consensus coding tasks used as a workload; NOT scored for correctness.",
        "n_tasks": len(tasks),
        "token_window": [min_tok, max_tok],
        "tasks": tasks,
    }, indent=2, ensure_ascii=False))

    print(f"[saved] {out}  ({len(tasks)} tasks)")
    print("  picked:")
    for t in tasks:
        print(f"    {t['instance_id']:35s} {t['repo']:24s} ~{t['est_context_tokens']:>6} tok")


def main() -> None:
    ap = argparse.ArgumentParser(prog="prepare-swe-tasks")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-tok", type=int, default=3000)
    ap.add_argument("--max-tok", type=int, default=12000)
    ap.add_argument("--out", default="runs/swe_bench/tasks.json")
    ap.add_argument("--cache-dir", default="runs/swe_bench/_hf_cache")
    args = ap.parse_args()
    prepare(args.n, args.min_tok, args.max_tok, Path(args.out), Path(args.cache_dir))


if __name__ == "__main__":
    main()
