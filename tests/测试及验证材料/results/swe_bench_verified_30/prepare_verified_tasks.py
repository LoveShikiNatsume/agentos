#!/usr/bin/env python3
"""Task 10: prepare 30 SWE-bench VERIFIED tasks as the honest_bench workload.

Mirrors scripts/swe_bench/prepare_swe_tasks.py (which is read-only and hardcodes
SWE-bench_Lite_oracle) but sources from SWE-bench Verified:
  - instance list + repo/base_commit/problem_statement/difficulty: princeton-nlp/SWE-bench_Verified
  - oracle 'text' (issue + gold source files) : princeton-nlp/SWE-bench_oracle (test split),
    filtered to the Verified instance_ids (Verified is a 500-subset of full test).

Sampling standard (written into output, decided independently of results):
  - token window [3000, 12000] est tokens (len//4), same as the Lite batch, so the
    shared-context size is comparable and API cost stays modest.
  - PROPORTIONAL stratified across repos (largest-remainder) so the project mix
    matches Verified's own distribution — no cherry-picking easy/small tasks.
  - within each repo, evenly spaced by est_token size (deterministic).
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

N = 30
MIN_TOK, MAX_TOK = 3000, 12000
OUT = Path("runs/swe_bench/tasks_verified_30.json")


def main() -> None:
    vp = glob.glob("runs/swe_bench/_hf_cache_verified/**/test-00000-of-00001.parquet", recursive=True)[0]
    ver = pd.read_parquet(vp)[["instance_id", "repo", "base_commit", "problem_statement", "difficulty"]]
    vids = set(ver["instance_id"])

    op = hf_hub_download("princeton-nlp/SWE-bench_oracle", "data/test-00000-of-00001.parquet",
                         repo_type="dataset", cache_dir="runs/swe_bench/_hf_cache_oracle")
    orc = pd.read_parquet(op)[["instance_id", "text"]]

    m = ver.merge(orc, on="instance_id", how="inner")
    m["est_tok"] = (m["text"].astype(str).str.len() / 4).astype(int)
    elig = m[(m["est_tok"] >= MIN_TOK) & (m["est_tok"] <= MAX_TOK)].copy()
    print(f"[coverage] Verified={len(ver)} merged_with_oracle={len(m)} eligible_in_window={len(elig)}")

    # proportional allocation across repos (largest remainder), deterministic
    counts = elig["repo"].value_counts()
    total = counts.sum()
    raw = {r: N * c / total for r, c in counts.items()}
    alloc = {r: int(v) for r, v in raw.items()}
    rem = N - sum(alloc.values())
    for r in sorted(raw, key=lambda r: (raw[r] - alloc[r], counts[r]), reverse=True)[:rem]:
        alloc[r] += 1

    picked = []
    for repo in sorted(counts.index):
        k = alloc.get(repo, 0)
        if k <= 0:
            continue
        g = elig[elig["repo"] == repo].sort_values("est_tok").reset_index(drop=True)
        step = max(1, len(g) // k)
        idxs = [i for i in range(0, len(g), step)][:k]
        for i in idxs:
            picked.append(g.iloc[i])

    tasks = [{
        "instance_id": r["instance_id"], "repo": r["repo"], "base_commit": r["base_commit"],
        "problem_statement": r["problem_statement"] or "", "shared_context": r["text"] or "",
        "est_context_tokens": int(r["est_tok"]), "difficulty": r["difficulty"],
    } for r in picked]
    tasks.sort(key=lambda t: (t["repo"], t["instance_id"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "princeton-nlp/SWE-bench_Verified (500) + princeton-nlp/SWE-bench_oracle text field",
        "note": "Real OpenAI-verified consensus coding tasks used as a workload; NOT scored for correctness.",
        "sampling_standard": (f"token window [{MIN_TOK},{MAX_TOK}] est tokens; proportional stratified "
                              "across repos (largest-remainder) to match Verified project distribution; "
                              "evenly spaced by token size within each repo; deterministic, results-independent."),
        "n_tasks": len(tasks), "token_window": [MIN_TOK, MAX_TOK],
        "repo_allocation": {r: alloc[r] for r in sorted(alloc) if alloc[r] > 0},
        "tasks": tasks,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[saved] {OUT}  ({len(tasks)} tasks)")
    print("[allocation]", {r: alloc[r] for r in sorted(alloc) if alloc[r] > 0})
    for t in tasks:
        print(f"    {t['instance_id']:34s} {t['repo']:24s} ~{t['est_context_tokens']:>6} tok  {t['difficulty']}")


if __name__ == "__main__":
    main()
