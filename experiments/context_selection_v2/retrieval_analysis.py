"""Network-free recall curves: v1 lexical distractors vs neutral, plus v2."""
from __future__ import annotations
import argparse
from pathlib import Path
import random
from statistics import mean
from experiments.context_selection.generate import collect_documents
from experiments.context_selection.policies import bm25 as original_bm25
from .common import HERE, digest, load, questions_for, write_new
from .generate import neutral_candidates
from .policies import bm25


def goals_for(q, bundle):
    if bundle is None:
        return [q["prompt"]]
    plan = bundle["questions"].get(q["id"], {})
    if plan.get("status") != "completed":
        return []
    return [r["goal"] for r in plan["plan"]["readers"]]


def curves(questions, documents, bundle=None):
    """Macro question recall, mean reader recall within question; also union.

    No goal-based gold allocation: every query is scored against all required
    docs, consistently across candidate/query variants. Missing plans excluded.
    """
    report = {}
    for index, ranker in (("v2_name_symbols_text", bm25), ("v1_name_text", original_bm25)):
        report[index] = {}
        for mode in ("goal", "question_plus_goal"):
            values, unions = [[] for _ in range(8)], [[] for _ in range(8)]
            for q in questions:
                goals = goals_for(q, bundle)
                if not goals:
                    continue
                rankings = [ranker(q, documents, goal if mode == "goal" else q["prompt"] + "\n" + goal, 8) for goal in goals]
                required = set(q["required_refs"])
                for k in range(1, 9):
                    values[k-1].append(mean(len(required & set(r[:k])) / len(required) for r in rankings))
                    unions[k-1].append(len(required & set().union(*(set(r[:k]) for r in rankings))) / len(required))
            report[index][mode] = {"questions": len(values[0]),
                "recall_at_k": [mean(v) if v else None for v in values],
                "reader_union_recall_at_k": [mean(v) if v else None for v in unions]}
    return report


def analyze_retrieval(v1, v2, corpus, *, seed=20261019, split="all", plans_v1=None, plans_v2=None):
    for manifest, plans in ((v1, plans_v1), (v2, plans_v2)):
        if plans is not None and plans["manifest_sha256"] != digest(manifest):
            raise ValueError("plans do not match retrieval manifest")
    # Enforce unchanged source identity for neutral resampling.
    for manifest in (v1, v2):
        for ref, doc in manifest["documents"].items():
            if ref not in corpus or corpus[ref] != doc:
                raise ValueError("local stdlib differs from frozen corpus")
    originals = questions_for(v1, split)
    neutral = []
    for q in originals:
        rng = random.Random(int(digest([seed, q["id"]]), 16))
        neutral.append({**q, "candidate_refs": neutral_candidates(corpus, q["required_refs"], rng)})
    v2_questions = questions_for(v2, split)
    return {"seed": seed, "split": split, "manifest_sha256": {"v1": digest(v1), "v2": digest(v2)},
        "protocol": {"k": list(range(1, 9)), "metric": "macro question mean of per-reader required-doc recall; reader union also reported",
                     "neutral": "one deterministic tiered uniform resample per v1 question; original required docs unchanged",
                     "query_v1": "frozen reader goals" if plans_v1 else "question fallback; goal and question+goal identical because query terms form a set",
                     "query_v2": "frozen reader goals" if plans_v2 else "question fallback; goal and question+goal identical because query terms form a set",
                     "plans_sha256": {"v1": digest(plans_v1) if plans_v1 else None, "v2": digest(plans_v2) if plans_v2 else None},
                     "no_model": True, "vectors": "not evaluated"},
        "v1_original": curves(originals, corpus, plans_v1),
        "v1_neutral": curves(neutral, corpus, plans_v1),
        "v2": curves(v2_questions, corpus, plans_v2),
        "v2_families": {family: curves([q for q in v2_questions if q["family"] == family], corpus, plans_v2)
                        for family in ("hop", "describe")}}


def render(report):
    lines = ["# Offline BM25 recall", "", "Question fallback is used when frozen plans are not supplied. No model calls.", "",
             "| Dataset | Index | Query | N | @1 | @2 | @3 | @4 | @5 | @6 | @7 | @8 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in ("v1_original", "v1_neutral", "v2"):
        for index, modes in report[label].items():
            for query, row in modes.items():
                scores = " | ".join(f"{v:.4f}" if v is not None else "NA" for v in row["recall_at_k"])
                lines.append(f"| {label} | {index} | {query} | {row['questions']} | {scores} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-manifest", type=Path, default=HERE.parent / "context_selection/manifest.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--source-root", type=Path, default=Path("/usr/lib/python3.11"))
    parser.add_argument("--plans-v1", type=Path)
    parser.add_argument("--plans-v2", type=Path)
    parser.add_argument("--split", choices=("probe", "formal", "all"), default="all")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus, _ = collect_documents(args.source_root)
    report = analyze_retrieval(load(args.v1_manifest), load(args.manifest), corpus,
        seed=args.seed, split=args.split, plans_v1=load(args.plans_v1) if args.plans_v1 else None,
        plans_v2=load(args.plans_v2) if args.plans_v2 else None)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "retrieval.json", report)
    (args.output / "retrieval.md").write_text(render(report))
    print(render(report))

if __name__ == "__main__":
    main()
