"""Freeze fault assignments against experiment A's existing AST-scored corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from .common import HERE, PROTOCOL_VERSION, digest, load, write_new


def build_manifest(source, *, source_manifest="../context_selection/manifest.json", seed=20261019):
    rng = random.Random(seed)
    assignments = []
    for split in sorted({q["split"] for q in source["questions"]}):
        questions = sorted((q for q in source["questions"] if q["split"] == split), key=lambda q: q["id"])
        rng.shuffle(questions)
        for index, q in enumerate(questions):
            wrong_answer = {**q["answer"], "parameter": "__injected_wrong_parameter__"}
            wrong_fact = {
                "symbol": q["answer"]["winner"], "parameter_count": 1000000,
                "first_default_parameter": wrong_answer["parameter"],
                "first_default": wrong_answer["default"],
            }
            assignments.append({
                "question_id": q["id"], "type": "execution" if index % 2 == 0 else "semantic",
                "attempt": 1, "after_private_segments": rng.randint(1, len(q["required_refs"])),
                "wrong_fact": wrong_fact, "wrong_answer": wrong_answer,
            })
    return {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "seed": seed,
        "source_manifest": source_manifest, "source_manifest_sha256": digest(source),
        "protocol": {
            "task": "one Runtime agent per question; extract each required document into private segments, then answer",
            "arms": ["dirty", "snapshot", "no_retry"], "max_attempts": {"dirty": 2, "snapshot": 2, "no_retry": 1},
            "single_factor": "SemanticRecoveryPolicy.restore_context; all other dirty/snapshot configuration identical",
            "injection": "first attempt only; execution after K extractions; semantic adds a false private fact, calls answer, then overrides returned answer",
            "source_order": "source question required_refs order; retries restart full extraction",
            "scoring": "strict experiment A JSON and AST expression normalization",
        },
        "injections": sorted(assignments, key=lambda row: row["question_id"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=HERE.parent / "context_selection" / "manifest.json")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output", type=Path, default=HERE / "manifest.json")
    args = parser.parse_args()
    import os
    relative = os.path.relpath(args.source_manifest.resolve(), args.output.resolve().parent)
    manifest = build_manifest(load(args.source_manifest), source_manifest=relative, seed=args.seed)
    write_new(args.output, manifest)
    print(json.dumps({"output": str(args.output), "questions": len(manifest["injections"]),
                      "seed": args.seed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
