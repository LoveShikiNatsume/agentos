"""Offline scheduled-request budget; char/4 input estimates are not model token usage."""
from __future__ import annotations

import argparse
import json
import math

from .common import ARMS, HERE, load_manifest, questions_for, write_new
from .workflow import CONTROL, answer_messages, extraction_messages


def estimate(manifest, split="formal", *, max_tokens=4096, scratch_tokens_per_document=512):
    if max_tokens < 2048:
        raise ValueError("max_tokens must be >=2048")
    result = {"split": split, "manifest_sha256": manifest["manifest_sha256"], "max_tokens": max_tokens,
              "input_measurement": "ceil(unicode_chars/4) prompt proxy; assumed 512 tokens per private extraction",
              "scratch_tokens_per_document": scratch_tokens_per_document,
              "assumptions": "fault reached; extraction/answer calls finish normally; retry completes full extraction; no hidden transport retries",
              "early_stop": "request errors or non-stop finish reasons may end an attempt earlier; these lower actual calls",
              "arms": {}}
    for arm in ARMS:
        count = first = prompt = 0
        for q in questions_for(manifest, split):
            n = len(q["required_refs"])
            k = q["injection"]["after_private_segments"]
            semantic = q["injection"]["type"] == "semantic"
            first_calls = k + int(semantic)
            retry_calls = n + 1 if arm != "no_retry" else 0
            count += first_calls + retry_calls
            first += first_calls
            # Source payload and common control; private facts grow with each step.
            def extraction_cost(index):
                raw = extraction_messages(q, q["required_refs"][index], manifest["documents"][q["required_refs"][index]])[0]["content"]
                return math.ceil((len(CONTROL) + len(raw)) / 4)
            prompt += sum(extraction_cost(i) + i * scratch_tokens_per_document for i in range(k))
            if semantic:
                prompt += math.ceil((len(CONTROL) + len(answer_messages(q)[0]["content"])) / 4)
                prompt += (k + 1) * scratch_tokens_per_document
            if arm != "no_retry":
                residue = (k + int(semantic)) * scratch_tokens_per_document if arm == "dirty" else 0
                prompt += sum(extraction_cost(i) + residue + i * scratch_tokens_per_document + 64 for i in range(n))
                prompt += math.ceil((len(CONTROL) + len(answer_messages(q)[0]["content"])) / 4)
                prompt += residue + n * scratch_tokens_per_document + 64
        result["arms"][arm] = {
            "questions": len(questions_for(manifest, split)), "scheduled_calls": count,
            "request_count_range_with_early_stops": [len(questions_for(manifest, split)), count],
            "first_attempt_calls_if_injection_reached": first,
            "estimated_input_tokens": prompt, "output_tokens_hard_cap": count * max_tokens,
            "estimated_output_tokens_512_per_call": count * 512,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default=str(HERE / "manifest.json"))
    parser.add_argument("--split", choices=("probe", "formal", "all"), default="formal")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    result = estimate(load_manifest(args.manifest), args.split, max_tokens=args.max_tokens)
    if args.output:
        write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
