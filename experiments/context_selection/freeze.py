"""Plan once, validate once plus one correction, then persist immutable inputs."""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
from .common import (HERE, PROTOCOL_VERSION, add_model_arguments, config_from_args,
                     digest, load, questions_for, write_new)
from .backends import LimitedBackend, make_backend
from .workflow import freeze_one, prompt_fingerprint

def sum_usage(rows):
    result = {}
    for key in ("model_calls", "truncated_calls", "prompt_tokens", "completion_tokens", "cached_tokens"):
        values = [row["usage"][key] for row in rows]
        result[key] = sum(values) if all(v is not None for v in values) else None
    return result

async def freeze(manifest, output, config, *, split="all", concurrency=4, backend_factory=None):
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    gate, jobs = asyncio.Semaphore(concurrency), asyncio.Semaphore(concurrency)
    async def one(q):
        async with jobs:
            backend = backend_factory(config) if backend_factory else make_backend(config)
            return await freeze_one(q, manifest["documents"], output / q["id"], config,
                                    backend=LimitedBackend(backend, gate))
    rows = await asyncio.gather(*(one(q) for q in questions_for(manifest, split)))
    bundle = {"schema_version": 1, "protocol_version": PROTOCOL_VERSION,
              "manifest_sha256": digest(manifest), "split": split, "config": config,
              "prompt_sha256": prompt_fingerprint(),
              "questions": {row["question_id"]: row for row in rows}, "usage": sum_usage(rows)}
    write_new(output / "plans.json", bundle)
    return bundle

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--split", choices=("probe", "formal", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    add_model_arguments(parser)
    args = parser.parse_args()
    bundle = asyncio.run(freeze(load(args.manifest), args.output, config_from_args(args),
                              split=args.split, concurrency=args.concurrency))
    print({"plans": str(args.output / "plans.json"), "questions": len(bundle["questions"]),
           "failed": sum(q["status"] != "completed" for q in bundle["questions"].values()),
           "usage": bundle["usage"]})

if __name__ == "__main__":
    main()
