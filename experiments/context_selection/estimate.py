"""Offline prompt-size estimate using the same selectors, templates and assembly."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from .common import ARMS, HERE, digest, load, questions_for, write_new
from .policies import SelectionContextManager, retrieval_budget, selected_documents
from .run import verify_bundle
from .workflow import CONTROL, answerer_messages, reader_messages, source_document

def estimate(manifest, bundle, *, split="formal", repeats=3, reader_output_tokens=256):
    if repeats < 1 or reader_output_tokens < 0:
        raise ValueError("invalid repeats/reader output budget")
    verify_bundle(manifest, bundle, bundle["config"], split)
    budget, budget_info = retrieval_budget(manifest, bundle, split)
    result = {"schema_version": 1, "manifest_sha256": digest(manifest), "plans_sha256": digest(bundle),
              "split": split, "repeats": repeats, "reader_output_tokens": reader_output_tokens,
              "method": "ceil(unicode message content characters/4) + 8 per message; not measured model tokens",
              "assumptions": ["frozen plans remain unchanged", "no transport or task retries",
                              "reader result length is a budget assumption, not a measured completion"],
              "bm25_budget": budget_info, "arms": {}}
    questions = questions_for(manifest, split)
    planning = [bundle["questions"][q["id"]] for q in questions]
    result["planning"] = {"questions": len(questions),
                           "frozen_calls": sum(p["usage"]["model_calls"] for p in planning),
                           "request_range_before_freezing": [len(questions), len(questions)*2]}
    for arm in ARMS:
        calls = tokens = source_tokens = skipped = 0
        for q in questions:
            frozen = bundle["questions"][q["id"]]
            if frozen["status"] != "completed":
                skipped += 1
                continue
            plan = frozen["plan"]
            manager = SelectionContextManager(prefix_order="name")
            manager.add_shared_ref("control", CONTROL)
            for ref in sorted(q["candidate_refs"]):
                manager.add_shared_ref(ref, source_document(ref, manifest["documents"][ref]))
            def count(messages, refs):
                prepared = manager.prepare_context("estimate", messages,
                                                   context_ref_names=["control"] + list(refs))
                return math.ceil(sum(len(m.get("content") or "") for m in prepared.messages)/4) + 8*len(prepared.messages)
            for row in plan["readers"]:
                chosen = selected_documents(arm, q, manifest["documents"], row,
                                             k=budget[(q["id"],row["id"])])
                tokens += count(reader_messages(q, row), chosen)
                source_tokens += sum(manifest["documents"][r]["estimated_tokens"] for r in chosen)
                manager.add_shared_ref("result/"+row["id"], "x"*(reader_output_tokens*4))
                calls += 1
            chosen = selected_documents(arm, q, manifest["documents"])
            tokens += count(answerer_messages(q), chosen + ["result/"+r["id"] for r in plan["readers"]])
            source_tokens += sum(manifest["documents"][r]["estimated_tokens"] for r in chosen)
            calls += 1
        result["arms"][arm] = {"execution_requests": calls*repeats,
              "estimated_prompt_tokens": tokens*repeats, "source_document_tokens": source_tokens*repeats,
              "completion_cap_tokens": calls*repeats*bundle["config"]["max_tokens"],
              "questions_skipped_due_to_failed_plan": skipped}
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=HERE/"manifest.json")
    parser.add_argument("--plans",type=Path,required=True)
    parser.add_argument("--split",choices=("probe","formal","all"),default="formal")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--reader-output-tokens",type=int,default=256)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=estimate(load(args.manifest),load(args.plans),split=args.split,repeats=args.repeats,
                    reader_output_tokens=args.reader_output_tokens)
    write_new(args.output,result)
    print(result)

if __name__ == "__main__":
    main()
