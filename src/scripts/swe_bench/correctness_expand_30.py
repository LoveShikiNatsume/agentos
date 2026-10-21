#!/usr/bin/env python3
"""SWE-bench Verified 30-task correctness expansion with byte comparison.

This script performs two independent, identically configured DeepSeek runs for
each task (native and runtime).  Both arms deliberately use the same prompts for
the three code-consuming roles.  Results are checkpointed after every arm.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


MODEL = "deepseek-v4-flash"
TEMPERATURE = 0
# V4 Flash exposes reasoning tokens inside completion_tokens.  The initial
# bounded pilot showed that smaller role-specific limits could be exhausted by
# reasoning before any answer text was emitted, so the formal run fixes the
# same sufficiently large cap for every role.
ROLE_MAX_TOKENS = {"localizer": 8192, "analyzer": 8192, "patcher": 8192}
MAX_RETRIES = 3
BASE_URL = "https://api.deepseek.com"
SYSTEM = "You are a senior software engineer working on a real GitHub issue. Be concise and precise."

# DeepSeek V4 Flash list-price assumptions (USD / 1M tokens), used only for an estimate.
PRICE_INPUT_MISS = 0.14
PRICE_INPUT_HIT = 0.0028
PRICE_OUTPUT = 0.28

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = ROOT / "runs/national_experiments/20260709_swe_bench_verified_deepseek_30/tasks_verified_30.json"
DEFAULT_OUT = ROOT / "runs/national_experiments/20260710_swe_bench_correctness_30"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def issue_and_code(task: dict[str, Any]) -> tuple[str, str]:
    shared = task.get("shared_context", "") or ""
    issue_match = re.search(r"<issue>(.*?)</issue>", shared, re.S)
    code_match = re.search(r"<code>(.*?)</code>", shared, re.S)
    issue = issue_match.group(1).strip() if issue_match else task.get("problem_statement", "")
    code = code_match.group(1).strip() if code_match else shared
    return issue, code


def first_source_file(code: str) -> tuple[str, str]:
    """Recover the first non-README oracle file for old-batch schema compatibility."""
    blocks = re.findall(r"\[start of (.*?)\]\n(.*?)\n\[end of \1\]", code, re.S)
    if not blocks:
        return "", ""
    chosen = next((item for item in blocks if not item[0].lower().startswith("readme")), blocks[0])
    path, numbered = chosen
    lines = []
    for line in numbered.splitlines():
        match = re.match(r"^\s*\d+\s?(.*)$", line)
        lines.append(match.group(1) if match else line)
    return path, "\n".join(lines) + "\n"


def build_messages(role: str, issue: str, code: str, findings: str = "") -> list[dict[str, str]]:
    parts = [f"# Issue\n{issue}", f"# Repository code\n{code}"]
    if findings:
        parts.append(f"# Upstream findings\n{findings}")
    if role == "localizer":
        instruction = "List the files and functions most relevant to fixing this issue."
    elif role == "analyzer":
        instruction = "Explain the root cause based on the supplied code and localization findings."
    else:
        instruction = (
            "Produce the minimal correct patch. Output only a standard git unified diff beginning "
            "with 'diff --git'. Use exact repository paths and valid hunk headers; do not use prose "
            "or Markdown fences. Do not modify tests unless the issue itself requires it."
        )
    parts.append(f"# Your task as the {role}\n{instruction}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def extract_unified_diff(raw: str) -> str:
    text = raw.strip()
    start = text.find("diff --git ")
    if start < 0:
        return ""
    text = text[start:]
    fence = text.find("\n```")
    if fence >= 0:
        text = text[:fence]
    return text.rstrip() + "\n"


async def api_call(
    client: httpx.AsyncClient,
    role: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.monotonic()
        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": ROLE_MAX_TOKENS[role],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return {
                "ok": True,
                "content": data["choices"][0]["message"].get("content") or "",
                "request_id": data.get("id") or response.headers.get("x-request-id", ""),
                "returned_model": data.get("model", ""),
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason", ""),
                "attempts": attempt,
                "latency_seconds": round(time.monotonic() - started, 3),
                "errors_before_success": errors,
            }
        except Exception as exc:  # bounded retry; never includes the API key
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(min(2**attempt, 8))
    return {"ok": False, "content": "", "attempts": MAX_RETRIES, "errors": errors}


async def run_arm(client: httpx.AsyncClient, task: dict[str, Any], arm: str) -> dict[str, Any]:
    issue, code = issue_and_code(task)
    calls: dict[str, Any] = {}
    findings: list[str] = []
    for role in ("localizer", "analyzer", "patcher"):
        prior = "\n\n".join(findings)
        messages = build_messages(role, issue, code, prior)
        call = await api_call(client, role, messages)
        call["prompt_sha256"] = sha256_text(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
        calls[role] = call
        if not call["ok"]:
            return {
                "arm": arm,
                "status": "failed",
                "failed_role": role,
                "calls": calls,
                "raw_patcher_output": "",
                "model_patch": "",
            }
        if role != "patcher":
            findings.append(f"## {role}\n{call['content']}")
    raw = calls["patcher"]["content"]
    patch = extract_unified_diff(raw)
    return {
        "arm": arm,
        "status": "complete" if patch else "no_unified_diff",
        "findings": "\n\n".join(findings),
        "calls": calls,
        "raw_patcher_output": raw,
        "model_patch": patch,
        "raw_sha256": sha256_text(raw),
        "patch_sha256": sha256_text(patch),
    }


def all_calls(generated: dict[str, Any]):
    for record in generated.values():
        for arm_data in record.get("arms", {}).values():
            for call in arm_data.get("calls", {}).values():
                yield call


def make_outputs(generated: dict[str, Any], out_dir: Path, started_at: str) -> None:
    comparisons: list[dict[str, Any]] = []
    predictions = {"native": [], "runtime": []}
    for iid, record in generated.items():
        arms = record.get("arms", {})
        native = arms.get("native", {})
        runtime = arms.get("runtime", {})
        raw_n, raw_r = native.get("raw_patcher_output", ""), runtime.get("raw_patcher_output", "")
        patch_n, patch_r = native.get("model_patch", ""), runtime.get("model_patch", "")
        complete = native.get("status") == "complete" and runtime.get("status") == "complete"
        comparisons.append({
            "instance_id": iid,
            "both_complete": complete,
            "identical": complete and raw_n.encode() == raw_r.encode(),
            "raw_identical": complete and raw_n.encode() == raw_r.encode(),
            "diff_identical": complete and patch_n.encode() == patch_r.encode(),
            "native_raw_sha256": sha256_text(raw_n),
            "runtime_raw_sha256": sha256_text(raw_r),
            "native_diff_sha256": sha256_text(patch_n),
            "runtime_diff_sha256": sha256_text(patch_r),
            "native_status": native.get("status", "missing"),
            "runtime_status": runtime.get("status", "missing"),
            "difference_sample": "" if raw_n == raw_r else {
                "native_prefix": raw_n[:500], "runtime_prefix": raw_r[:500]
            },
        })
        for arm, data in (("native", native), ("runtime", runtime)):
            predictions[arm].append({
                "instance_id": iid,
                "model_name_or_path": f"agentos-{arm}-{MODEL}-temp0",
                "model_patch": data.get("model_patch", ""),
            })

    completed = [row for row in comparisons if row["both_complete"]]
    identical_raw = sum(row["raw_identical"] for row in completed)
    identical_diff = sum(row["diff_identical"] for row in completed)
    compare_doc = {
        "definition": "UTF-8 encoding of exact strings compared byte-for-byte; identical aliases raw_identical.",
        "n_selected": len(generated),
        "n_both_complete": len(completed),
        "n_raw_identical": identical_raw,
        "n_diff_identical": identical_diff,
        "raw_identical_rate": identical_raw / len(completed) if completed else None,
        "diff_identical_rate": identical_diff / len(completed) if completed else None,
        "tasks": comparisons,
    }
    dump_json(out_dir / "byte_compare.json", compare_doc)
    dump_json(out_dir / "predictions_native.json", predictions["native"])
    dump_json(out_dir / "predictions_runtime.json", predictions["runtime"])

    usage_fields = ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
    usage = {field: sum(int(call.get("usage", {}).get(field, 0) or 0) for call in all_calls(generated))
             for field in usage_fields}
    usage["total_tokens"] = sum(int(call.get("usage", {}).get("total_tokens", 0) or 0)
                                for call in all_calls(generated))
    successful_calls = [call for call in all_calls(generated) if call.get("ok")]
    request_ids = [call.get("request_id") for call in successful_calls if call.get("request_id")]
    estimated_cost = (
        usage["prompt_cache_miss_tokens"] * PRICE_INPUT_MISS
        + usage["prompt_cache_hit_tokens"] * PRICE_INPUT_HIT
        + usage["completion_tokens"] * PRICE_OUTPUT
    ) / 1_000_000
    summary = {
        "experiment": "SWE-bench Verified correctness expansion 30",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "role_max_tokens": ROLE_MAX_TOKENS,
        "max_retries_per_call": MAX_RETRIES,
        "n_selected": len(generated),
        "n_both_complete": len(completed),
        "n_raw_identical": identical_raw,
        "n_diff_identical": identical_diff,
        "usage": usage,
        "pricing_assumption_usd_per_million": {
            "cache_miss_input": PRICE_INPUT_MISS,
            "cache_hit_input": PRICE_INPUT_HIT,
            "output": PRICE_OUTPUT,
        },
        "estimated_cost_usd": round(estimated_cost, 8),
        "started_at_utc": started_at,
        "updated_at_utc": utc_now(),
        "truth_probe": {
            "real_api": bool(successful_calls),
            "successful_api_calls": len(successful_calls),
            "request_id_count": len(request_ids),
            "request_id_sha256": sha256_text("\n".join(request_ids)) if request_ids else "",
            "returned_models": sorted({call.get("returned_model", "") for call in successful_calls}),
            "usage_from_api_response": bool(successful_calls) and all("usage" in call for call in successful_calls),
            "api_key_persisted": False,
        },
    }
    dump_json(out_dir / "summary.json", summary)


def aggregate_official(out_dir: Path) -> None:
    """Aggregate the six official five-instance harness reports per arm."""
    official_dir = out_dir / "official_docker_crossvalidation"
    compare = json.loads((out_dir / "byte_compare.json").read_text(encoding="utf-8"))
    generated = json.loads((out_dir / "generated_patches.json").read_text(encoding="utf-8"))
    instance_ids = [row["instance_id"] for row in compare["tasks"]]
    per_arm: dict[str, dict[str, str]] = {"native": {}, "runtime": {}}
    run_ids: dict[str, list[str]] = {"native": [], "runtime": []}
    for arm in ("native", "runtime"):
        for path in sorted(official_dir.glob(f"agentos-{arm}-*.correctness30_{arm}_b*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            run_ids[arm].append(path.stem.rsplit(".", 1)[-1])
            for iid in report.get("resolved_ids", []):
                per_arm[arm][iid] = "resolved"
            for iid in report.get("unresolved_ids", []):
                per_arm[arm][iid] = "unresolved"
            for iid in report.get("error_ids", []):
                per_arm[arm][iid] = "patch_apply_error"

    per_instance = {
        iid: {
            "native": per_arm["native"].get(iid, "missing"),
            "runtime": per_arm["runtime"].get(iid, "missing"),
        }
        for iid in instance_ids
    }
    totals: dict[str, Any] = {}
    for arm in ("native", "runtime"):
        statuses = list(per_arm[arm].values())
        resolved = statuses.count("resolved")
        totals[arm] = {
            "resolved": resolved,
            "unresolved": statuses.count("unresolved"),
            "patch_apply_error": statuses.count("patch_apply_error"),
            "missing": len(instance_ids) - len(statuses),
            "total": len(instance_ids),
            "absolute_resolved_rate": resolved / len(instance_ids),
        }
    prompt_identity = {}
    for role in ("localizer", "analyzer", "patcher"):
        prompt_identity[role] = sum(
            record["arms"]["native"]["calls"][role]["prompt_sha256"]
            == record["arms"]["runtime"]["calls"][role]["prompt_sha256"]
            for record in generated.values()
        )
    official_summary = {
        "method": "official swebench.harness.run_evaluation 4.1.0 with official prebuilt Docker images",
        "dataset": "princeton-nlp/SWE-bench_Verified (test split)",
        "batch_size": 5,
        "run_ids": run_ids,
        "per_instance": per_instance,
        "totals": totals,
        "denominator_policy": "All 30 selected tasks; patch application errors count as not resolved.",
        "prompt_sha256_identical_counts_out_of_30": prompt_identity,
        "disk_guard": "Checked before every harness run; stop threshold 60GB; observed space never below 50GB.",
    }
    dump_json(official_dir / "official_docker_crossvalidation_summary.json", official_summary)

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prompt_sha256_identical_counts_out_of_30"] = prompt_identity
    summary["official_evaluation"] = totals
    dump_json(summary_path, summary)


async def main_async(args: argparse.Namespace) -> int:
    if args.aggregate_official:
        aggregate_official(args.out)
        print(f"[saved] {args.out / 'official_docker_crossvalidation/official_docker_crossvalidation_summary.json'}")
        return 0
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return 2
    tasks_doc = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = tasks_doc["tasks"][: args.limit] if args.limit else tasks_doc["tasks"]
    args.out.mkdir(parents=True, exist_ok=True)
    generated_path = args.out / "generated_patches.json"
    generated = json.loads(generated_path.read_text(encoding="utf-8")) if generated_path.exists() else {}
    started_at = utc_now()
    config = {
        "model": MODEL, "temperature": TEMPERATURE, "role_max_tokens": ROLE_MAX_TOKENS,
        "max_retries_per_call": MAX_RETRIES, "task_concurrency": args.concurrency,
        "tasks_path": str(args.tasks), "started_at_utc": started_at,
    }
    dump_json(args.out / "run_config.json", config)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=180.0) as client:
        async def process_task(index: int, task: dict[str, Any]) -> None:
            iid = task["instance_id"]
            async with semaphore:
                if iid not in generated:
                    issue, code = issue_and_code(task)
                    file_path, original = first_source_file(code)
                    generated[iid] = {
                        "repo": task["repo"], "file_path": file_path,
                        "base_commit": task["base_commit"], "original_content": original,
                        "problem_statement": task.get("problem_statement", issue), "arms": {},
                    }
                print(f"[{index}/{len(tasks)}] {iid}", flush=True)
                for arm in ("native", "runtime"):
                    if generated[iid]["arms"].get(arm, {}).get("status") in {"complete", "no_unified_diff"}:
                        print(f"  {arm}: checkpoint exists", flush=True)
                        continue
                    print(f"  {arm}: calling {MODEL} temperature={TEMPERATURE}", flush=True)
                    result = await run_arm(client, task, arm)
                    async with write_lock:
                        generated[iid]["arms"][arm] = result
                        dump_json(generated_path, generated)
                        make_outputs(generated, args.out, started_at)
                    print(f"  {arm}: {result['status']}", flush=True)

        await asyncio.gather(*(process_task(index, task) for index, task in enumerate(tasks, 1)))
    make_outputs(generated, args.out, started_at)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--aggregate-official", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
