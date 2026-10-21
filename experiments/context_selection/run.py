"""Run one arm using a mandatory, already frozen plan bundle."""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
from .backends import LimitedBackend, make_backend
from .common import (ARMS, HERE, PROTOCOL_VERSION, add_model_arguments, config_from_args,
                     digest, load, questions_for, write_new)
from .freeze import sum_usage
from .policies import retrieval_budget
from .scoring import normalize_answer, score_answer
from .workflow import execute_one, prompt_fingerprint, validate_plan

def verify_bundle(manifest, bundle, config, split):
    if bundle["manifest_sha256"] != digest(manifest):
        raise ValueError("frozen plans belong to a different manifest")
    if (bundle["protocol_version"] != PROTOCOL_VERSION
            or bundle["prompt_sha256"] != prompt_fingerprint()):
        raise ValueError("planner protocol changed; create a new reviewed plan bundle")
    for key in ("backend", "model", "base_url", "seed", "max_tokens", "temperature"):
        if bundle["config"][key] != config[key]:
            raise ValueError(f"frozen planner configuration mismatch: {key}")
    for q in questions_for(manifest, split):
        if q["id"] not in bundle["questions"]:
            raise ValueError(f"missing frozen plan: {q['id']}")
        frozen = bundle["questions"][q["id"]]
        if frozen["status"] == "completed":
            validate_plan(frozen["plan"], q)
            if digest(frozen["plan"]) != frozen["plan_sha256"]:
                raise ValueError("frozen plan hash mismatch")

async def run_arm(manifest, bundle, arm, output, config, *, split="probe", concurrency=4,
                  repeats=3, backend_factory=None):
    if arm not in ARMS or concurrency < 1 or repeats < 1:
        raise ValueError("invalid arm/concurrency/repeats")
    verify_bundle(manifest, bundle, config, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    questions = questions_for(manifest, split)
    budget, budget_info = retrieval_budget(manifest, bundle, split)
    gate, jobs = asyncio.Semaphore(concurrency), asyncio.Semaphore(concurrency)
    async def one(q, replicate):
        frozen = bundle["questions"][q["id"]]
        planner_truncated = frozen["usage"]["truncated_calls"]
        if frozen["status"] != "completed" or planner_truncated:
            row = dict(answer=None, valid=False, failures=[frozen.get("error") or "planner truncated"],
                       model_calls=0, prompt_tokens=0, completion_tokens=0, cached_tokens=0,
                       truncated_calls=0, required_recall=None, complete_required=None,
                       plan_sha256=frozen["plan_sha256"])
        else:
            async with jobs:
                replicate_config = {**config, "seed": config["seed"] + replicate}
                backend = backend_factory(replicate_config) if backend_factory else make_backend(replicate_config)
                row = await execute_one(q, manifest["documents"], frozen, arm,
                    output / q["id"] / f"repeat-{replicate}", replicate_config, budget,
                    backend=LimitedBackend(backend, gate))
        row["question_id"], row["replicate"] = q["id"], replicate
        row["execution_truncated_calls"] = row["truncated_calls"]
        row["planner_truncated_calls"] = planner_truncated
        row["truncated_calls"] += planner_truncated
        row["correct"] = bool(row["valid"] and score_answer(row["answer"], q["answer"]))
        if row["valid"]:
            try:
                normalize_answer(row["answer"])
            except ValueError as exc:
                row["failures"].append({"kind": "answer_format", "reason": str(exc)})
        write_new(output / "results" / f"{q['id']}-r{replicate}.json", row)
        return row
    rows = await asyncio.gather(*(one(q, r) for q in questions for r in range(repeats)))
    planning = sum_usage([bundle["questions"][q["id"]] for q in questions])
    summary = {"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "arm": arm, "split": split,
               "manifest_sha256": digest(manifest), "plans_sha256": digest(bundle),
               "config": {**config, "repeats": repeats, "concurrency": concurrency},
               "token_measurement": "stub_chars_div_4" if config["backend"] == "stub" else "provider_usage",
               "planning": planning, "bm25_budget": budget_info, "questions": rows}
    write_new(output / "summary.json", summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--plans", type=Path, required=True, help="never silently invoke a planner")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--split", choices=("probe", "formal", "all"), default="probe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4, help="hard cap on simultaneous model calls")
    parser.add_argument("--repeats", type=int, default=3)
    add_model_arguments(parser)
    args = parser.parse_args()
    result = asyncio.run(run_arm(load(args.manifest), load(args.plans), args.arm, args.output,
                                config_from_args(args), split=args.split, concurrency=args.concurrency,
                                repeats=args.repeats))
    print({"arm": args.arm, "questions": len(result["questions"]),
           "valid": sum(r["valid"] for r in result["questions"]),
           "calls": sum(r["model_calls"] for r in result["questions"]),
           "summary": str(args.output / "summary.json")})

if __name__ == "__main__":
    main()
