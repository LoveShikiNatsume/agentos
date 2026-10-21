"""Offline nominal and bounded-need budgets; never contacts a backend."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from .common import ARMS, HERE, APPEND_LIMITS, digest, load, questions_for, write_new
from .policies import SelectionContextManager, retrieval_budget, selected_documents
from .run import verify_bundle
from .workflow import CONTROL, answerer_messages, reader_messages, source_document, request_feedback


def estimate(manifest, bundle, *, split="formal", repeats=3, reader_output_tokens=256,
             need_output_tokens=128, append_summary=None):
    if repeats < 1 or reader_output_tokens < 0 or need_output_tokens < 0:
        raise ValueError("invalid budget arguments")
    verify_bundle(manifest, bundle, bundle["config"], split)
    budget, info = retrieval_budget(manifest, bundle, split, repeats=repeats)
    secondary, secondary_info = ({}, None)
    if append_summary is not None:
        secondary, secondary_info = retrieval_budget(manifest, bundle, split, repeats=repeats,
                                                     append_summary=append_summary)
    report = {"schema_version": 2, "manifest_sha256": digest(manifest), "plans_sha256": digest(bundle),
              "split": split, "repeats": repeats, "reader_output_tokens": reader_output_tokens,
              "need_output_tokens": need_output_tokens,
              "method": "ceil(message content characters / 4) + 8 per message, not model tokens",
              "assumptions": ["nominal: no need responses", "bounded_need: all readers use 2 needs/3 appended refs if allowed, otherwise 1 denied need",
                              "appended docs: longest remaining candidate texts for a conservative source-size scenario",
                              "need text and result facts lengths are assumptions, not measured; bounded_need is not a hard tokenizer bound",
                              "bm25_append omitted unless matching append summary supplies its mean k"],
              "bm25_budget": info, "bm25_append_budget": secondary_info, "arms": {}}
    qs = questions_for(manifest, split)
    plans = [bundle["questions"][q["id"]] for q in qs]
    report["planning"] = {"questions": len(qs), "request_range_before_freezing": [len(qs), 2*len(qs)],
                          "frozen_calls": sum(p["usage"]["model_calls"] for p in plans),
                          "frozen_prompt_tokens": sum(p["usage"]["prompt_tokens"] for p in plans)
                          if all(p["usage"]["prompt_tokens"] is not None for p in plans) else None}
    for arm in ARMS:
        if arm == "bm25_append" and append_summary is None:
            continue
        scenarios = {}
        for scenario in ("nominal", "bounded_need"):
            calls = tokens = skipped = 0
            for q in qs:
                frozen = bundle["questions"][q["id"]]
                if frozen["status"] != "completed":
                    skipped += 1
                    continue
                for rep in range(repeats):
                    cm = SelectionContextManager(prefix_order="name")
                    cm.add_shared_ref("control", CONTROL)
                    for ref in sorted(q["candidate_refs"]):
                        cm.add_shared_ref(ref, source_document(ref, manifest["documents"][ref]))
                    def count(messages, refs):
                        prepared = cm.prepare_context("estimate", messages, context_ref_names=["control", *refs])
                        return math.ceil(sum(len(m.get("content") or "") for m in prepared.messages)/4) + 8*len(prepared.messages)
                    for row in frozen["plan"]["readers"]:
                        key = (q["id"], row["id"], rep)
                        refs = selected_documents(arm, q, manifest["documents"], row,
                            k=(secondary if arm == "bm25_append" else budget)[key])
                        messages = reader_messages(q, row, manifest["documents"])
                        tokens += count(messages, refs)
                        calls += 1
                        if scenario == "bounded_need":
                            rounds = 2 if arm == "declared_append" else 1
                            pool = sorted(set(q["candidate_refs"]) - set(refs),
                                          key=lambda r: (-len(manifest["documents"][r]["text"]), r))
                            for turn in range(rounds):
                                need = pool[:2 if turn == 0 else 1]
                                granted = need if arm == "declared_append" else []
                                pool = [r for r in pool if r not in granted]
                                refs += granted
                                result = {"requested": need, "granted": granted,
                                          "denied": [] if arm == "declared_append" else [{"ref": r, "reason": "policy_disabled"} for r in need],
                                          "reason": "budget estimate", "attempt": 1,
                                          "remaining_requests": 1-turn if arm == "declared_append" else 0,
                                          "remaining_refs": max(0, 1-turn) if arm == "declared_append" else 0,
                                          "requests_used": turn+1, "refs_granted": 2+turn if arm == "declared_append" else 0}
                                messages += [{"role": "assistant", "content": "x" * (need_output_tokens*4)},
                                             request_feedback(result, final=turn == rounds-1)]
                                tokens += count(messages, refs)
                                calls += 1
                        cm.add_shared_ref("result/"+row["id"], "x"*(reader_output_tokens*4))
                    tokens += count(answerer_messages(q), ["result/"+r["id"] for r in frozen["plan"]["readers"]])
                    calls += 1
            scenarios[scenario] = {"execution_requests": calls, "estimated_prompt_tokens": tokens,
                "completion_cap_tokens": calls*bundle["config"]["max_tokens"], "failed_plan_questions": skipped}
        report["arms"][arm] = scenarios
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE/"manifest.json")
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--append-summary", type=Path)
    parser.add_argument("--split", choices=("probe", "formal", "all"), default="formal")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reader-output-tokens", type=int, default=256)
    parser.add_argument("--need-output-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = estimate(load(args.manifest), load(args.plans), split=args.split, repeats=args.repeats,
                      reader_output_tokens=args.reader_output_tokens, need_output_tokens=args.need_output_tokens,
                      append_summary=load(args.append_summary) if args.append_summary else None)
    write_new(args.output, result)
    print(result)

if __name__ == "__main__":
    main()
