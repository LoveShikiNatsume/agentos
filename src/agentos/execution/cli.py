"""Argparse wiring kept separate from the legacy inspection commands."""
from __future__ import annotations

import asyncio
import argparse
import json
import sys
from pathlib import Path

from agentos.execution.config import FIELDS


def options(parser, *, with_output=True):
    for name in ("model", "base_url", "harness", "profile", "api_key_env", "thinking", "reasoning_effort", "response_format", "command_sandbox", "context_optimization"):
        parser.add_argument("--" + name.replace("_", "-"))
    for name in ("max_steps", "max_tasks", "max_parallel", "retries"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--max-tokens", type=lambda value: "provider_default" if value == "provider_default" else int(value),
                        help="Optional generation limit; omitted uses provider default (not context capacity)")
    for name in ("temperature", "timeout", "command_timeout", "run_timeout"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--command-read-only-path", dest="command_read_only_paths", action="append",
                        help="Explicit absolute dependency path exposed read-only to sandboxed commands; repeatable")
    parser.add_argument("--command-network", action=argparse.BooleanOptionalAction, default=None,
                        help="Allow sandboxed commands to share host networking (default: disabled)")
    if with_output:
        parser.add_argument("--output", help="Root directory for unique run bundles")
    parser.add_argument("--json", action="store_true", help="Stream JSONL events followed by a summary object")


def overrides(args):
    return {k: getattr(args, k) for k in FIELDS if getattr(args, k, None) is not None}


def listener(json_mode):
    def emit(event):
        if json_mode:
            print(json.dumps({"type": "event", **event}, ensure_ascii=False), flush=True)
        elif event["event_type"] in {"harness.plan_created", "execution.tool_started", "execution.model_retry", "execution.failed"}:
            data = event["data"]
            detail = f"tasks={len(data['tasks'])}" if "tasks" in data else data.get("tool") or data.get("error", "")
            print(f"[{event['task_id'] or '-'}] {event['event_type']} {detail}", file=sys.stderr, flush=True)
    return emit


def exec_command(args):
    from agentos.execution.service import execute
    result = asyncio.run(execute(args.task, cwd=args.cwd, overrides=overrides(args), listener=listener(args.json)))
    if args.json:
        print(json.dumps({"type": "summary", **result}, ensure_ascii=False))
    else:
        print(f"{result['run_id']}  workflow={result['workflow_status']}  score={result['score']['status']}")
        for tid, answer in result["answers"].items():
            print(f"{tid}: {answer}")
        usage = result["usage"]
        print(f"elapsed={result['execution_seconds']:.2f}s input_tokens={usage['prompt_tokens']} output_tokens={usage['completion_tokens']} usage_complete={usage['usage_complete']}")
        print(f"records: {result['output']}")
        if result["error"]:
            print(result["error"], file=sys.stderr)
    return 0 if result["workflow_status"] == "completed" else 1


def bench_command(args):
    from agentos.execution.bench import run_suite
    config = overrides(args)
    config.pop("output", None)
    result = asyncio.run(run_suite(args.suite, compare=args.compare, output=args.output,
                        overrides=config, listener=listener(args.json)))
    print(json.dumps({"type": "batch", **result}, ensure_ascii=False))
    failed = False
    for row in result["runs"]:
        summary = json.loads((Path(row["output"]) / "summary.json").read_text())
        score = summary["score"]
        if row.get("evaluation"):
            score = json.loads((Path(result["output"]) / row["evaluation"] / "evaluation.json").read_text())["score"]
        failed |= summary["workflow_status"] != "completed" or score.get("passed") is False or score["status"] in {"error", "not_evaluated"}
    return int(failed)


def evaluate_command(args):
    from agentos.execution.swebench import evaluate_run
    path = Path(args.spec).expanduser().resolve()
    spec = json.loads(path.read_text())
    result = asyncio.run(evaluate_run(Path(args.run), spec, output=args.output, relative_to=path.parent))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["score"]["outcome"] == "resolved" else 1


def report_command(args):
    from agentos.execution.bench import report
    print(report(args.batch, root=args.root), end="")
    return 0


def add_commands(subparsers):
    execution = subparsers.add_parser("exec", help="Execute a natural-language repository task")
    execution.add_argument("task")
    execution.add_argument("--cwd", default=".")
    options(execution)
    execution.set_defaults(func=exec_command)
    bench = subparsers.add_parser("bench", help="Execute an isolated manifest suite")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True)
    run = bench_commands.add_parser("run")
    run.add_argument("--suite", required=True)
    run.add_argument("--compare", default="native,runtime")
    run.add_argument("--output", default=".agentos/batches")
    options(run, with_output=False)
    run.set_defaults(func=bench_command)
    evaluate = bench_commands.add_parser("evaluate", help="Officially evaluate an archived run without calling a model")
    evaluate.add_argument("--run", required=True, help="Existing source run bundle")
    evaluate.add_argument("--spec", required=True, help="Trusted SWE-bench scoring JSON")
    evaluate.add_argument("--output", default=".agentos/evaluations", help="Separate append-only evaluation root")
    evaluate.set_defaults(func=evaluate_command)
    report = subparsers.add_parser("report", help="Export CSV from a batch or linked evaluation bundle")
    report.add_argument("batch")
    report.add_argument("--format", choices=["csv"], default="csv")
    report.add_argument("--root", default=".agentos/batches")
    report.set_defaults(func=report_command)
