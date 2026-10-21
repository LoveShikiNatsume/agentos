"""Run one preassigned experiment C arm. --stub performs no network requests."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .backends import LimitedBackend, make_backend
from .common import (ARMS, HERE, PROTOCOL_VERSION, add_model_arguments, config_from_args,
                     load_manifest, questions_for, write_new)
from .workflow import execute_one, prompt_fingerprint


async def run_arm(manifest, arm, output, config, *, split="probe", concurrency=4, backend_factory=None):
    if arm not in ARMS or concurrency < 1 or split not in ("probe", "formal", "all"):
        raise ValueError("invalid arm/concurrency/split")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    gate, jobs = asyncio.Semaphore(concurrency), asyncio.Semaphore(concurrency)

    async def one(question):
        async with jobs:
            backend = backend_factory(config) if backend_factory else make_backend(config)
            row = await execute_one(question, manifest["documents"], arm, output / question["id"], config,
                                    backend=LimitedBackend(backend, gate))
            write_new(output / "results" / (question["id"] + ".json"), row)
            return row

    rows = await asyncio.gather(*(one(question) for question in questions_for(manifest, split)))
    summary = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "arm": arm, "split": split,
        "manifest_sha256": manifest["manifest_sha256"], "prompt_sha256": prompt_fingerprint(),
        "config": {**config, "concurrency": concurrency, "max_attempts": 1 if arm == "no_retry" else 2,
                   "restore_context": arm == "snapshot"},
        "token_measurement": "stub_chars_div_4_not_model_tokens" if config["backend"] == "stub" else "provider_usage",
        "questions": rows,
    }
    write_new(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--split", choices=("probe", "formal", "all"), default="probe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4, help="hard cap on simultaneous model requests")
    add_model_arguments(parser)
    args = parser.parse_args()
    summary = asyncio.run(run_arm(load_manifest(args.manifest), args.arm, args.output,
                                 config_from_args(args), split=args.split, concurrency=args.concurrency))
    print(json.dumps({"arm": args.arm, "questions": len(summary["questions"]),
        "completed": sum(row["completed"] for row in summary["questions"]),
        "correct": sum(row["correct"] for row in summary["questions"]),
        "calls": sum(row["model_calls"] for row in summary["questions"]),
        "summary": str(args.output / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
