from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from agentos.dashboard import build_dashboard
from agentos.events.log import EventLog
from agentos.events.types import RuntimeEvent
from agentos.fault.injector import FaultType, inject_fault


def _format_data(event: RuntimeEvent) -> str:
    if event.event_type == "agent.state_changed":
        old_state = event.data.get("old_state", "?")
        new_state = event.data.get("new_state", "?")
        return f"{old_state} -> {new_state}"
    if event.event_type == "fault.timeout":
        timeout = event.data.get("timeout_seconds", "?")
        return f"timeout={timeout}s"
    if event.event_type == "fault.memory_pressure":
        parts = []
        for key in ("memory_mb", "chunk_mb", "hold_seconds", "execution"):
            if key in event.data and event.data[key] not in (None, ""):
                parts.append(f"{key}={event.data[key]}")
        return " ".join(parts)
    if event.event_type.startswith("worker."):
        parts = []
        for key in ("pid", "worker_type", "exit_code", "timed_out", "killed", "error"):
            if key in event.data and event.data[key] not in (None, ""):
                parts.append(f"{key}={event.data[key]}")
        return " ".join(parts)
    if event.event_type.startswith("resource."):
        parts = []
        for key in ("path", "pid", "reason"):
            if key in event.data and event.data[key] not in (None, ""):
                parts.append(f"{key}={event.data[key]}")
        usage = event.data.get("usage")
        if isinstance(usage, dict):
            if "memory_rss_mb" in usage:
                parts.append(f"mem_mb={usage['memory_rss_mb']}")
            if "memory_peak_mb" in usage:
                parts.append(f"peak_mb={usage['memory_peak_mb']}")
            if "pids_current" in usage:
                parts.append(f"pids={usage['pids_current']}")
            if "cpu_usage_usec" in usage:
                parts.append(f"cpu_usec={usage['cpu_usage_usec']}")
        limits = event.data.get("limits")
        if isinstance(limits, dict):
            compact_limits = {
                key: value
                for key, value in limits.items()
                if value not in (None, False, [], "")
            }
            if compact_limits:
                parts.append(f"limits={compact_limits}")
        return " ".join(parts)
    if event.event_type.startswith("tool."):
        parts = []
        for key in ("tool", "duration_secs", "timed_out", "error"):
            if key in event.data and event.data[key] not in (None, ""):
                parts.append(f"{key}={event.data[key]}")
        return " ".join(parts)
    if event.data:
        return " ".join(f"{key}={value}" for key, value in event.data.items())
    return ""


def _print_trace(args: argparse.Namespace) -> int:
    log = EventLog(Path(args.log))
    events = list(
        log.read(
            agent_id=args.agent,
            task_id=args.task,
            event_type=args.type,
            category=args.category,
        )
    )
    if not events:
        print("No events found")
        return 0

    for event in events:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))
        agent = event.agent_id or "-"
        task = event.task_id or "-"
        data = _format_data(event)
        suffix = f" {data}" if data else ""
        print(f"{ts} {event.event_type:<24} agent={agent:<16} task={task:<16}{suffix}")
    return 0


def _print_ps(args: argparse.Namespace) -> int:
    log = EventLog(Path(args.log))
    agents: dict[str, dict[str, object]] = {}
    for event in log.read(task_id=args.task):
        if event.agent_id is None:
            continue
        row = agents.setdefault(
            event.agent_id,
            {
                "agent_id": event.agent_id,
                "task_id": event.task_id or "-",
                "state": "-",
                "role": "-",
                "model": "-",
                "tokens": 0,
                "failures": 0,
                "pid": "-",
                "worker": "-",
                "exit": "-",
                "cgroup": "-",
                "mem_mb": "-",
                "pids": "-",
            },
        )
        if event.task_id:
            row["task_id"] = event.task_id
        if event.event_type == "agent.created":
            row["role"] = event.data.get("role", "-")
        elif event.event_type == "agent.state_changed":
            row["state"] = event.data.get("new_state", "-")
            if event.data.get("new_state") == "failed":
                row["failures"] = int(row["failures"]) + 1
        elif event.event_type == "llm.response_received":
            row["model"] = event.data.get("model", "-")
            row["tokens"] = int(row["tokens"]) + int(event.data.get("tokens_used", 0))
        elif event.event_type in {
            "fault.triggered",
            "fault.timeout",
            "fault.memory_pressure",
        }:
            row["failures"] = int(row["failures"]) + 1
        elif event.event_type == "worker.started":
            row["pid"] = event.data.get("pid", "-")
            row["worker"] = event.data.get("worker_type", "-")
        elif event.event_type in {"worker.finished", "worker.failed", "worker.timeout"}:
            row["pid"] = event.data.get("pid", row["pid"])
            row["worker"] = event.data.get("worker_type", row["worker"])
            row["exit"] = event.data.get("exit_code", "-")
            if event.event_type in {"worker.failed", "worker.timeout"}:
                row["failures"] = int(row["failures"]) + 1
        elif event.event_type == "resource.cgroup_created":
            row["cgroup"] = event.data.get("path", "-")
        elif event.event_type == "resource.cgroup_unavailable":
            row["cgroup"] = event.data.get("path", row["cgroup"])
        elif event.event_type == "resource.cgroup_pid_added":
            row["cgroup"] = event.data.get("path", row["cgroup"])
            row["pid"] = event.data.get("pid", row["pid"])
        elif event.event_type == "resource.usage_sampled":
            row["cgroup"] = event.data.get("path", row["cgroup"])
            usage = event.data.get("usage", {})
            if isinstance(usage, dict):
                row["mem_mb"] = usage.get("memory_rss_mb", row["mem_mb"])
                row["pids"] = usage.get("pids_current", row["pids"])

    if not agents:
        print("No agents found")
        return 0

    print(
        "AGENT_ID         TASK_ID          STATE        ROLE         MODEL        "
        "TOKENS  FAILURES PID      WORKER       EXIT  MEM_MB   PIDS CGROUP"
    )
    for row in agents.values():
        print(
            f"{row['agent_id']:<16} {row['task_id']:<16} {row['state']:<12} "
            f"{row['role']:<12} {row['model']:<12} {row['tokens']:<7} "
            f"{row['failures']:<8} {row['pid']:<8} {row['worker']:<12} "
            f"{row['exit']:<5} {row['mem_mb']:<8} {row['pids']:<4} {row['cgroup']}"
        )
    return 0


def _print_context(args: argparse.Namespace) -> int:
    log = EventLog(Path(args.log))
    stats_event: RuntimeEvent | None = None
    for event in log.read(event_type="context.stats"):
        stats_event = event

    if stats_event is None:
        print("No context stats found")
        return 0

    stats = stats_event.data.get("stats", {})
    pool = stats.get("pool", {}) if isinstance(stats, dict) else {}
    print("Context Stats")
    print(f"total_refs={stats.get('total_refs', 0)}")
    for key in sorted(pool):
        print(f"{key}={pool[key]}")
    return 0


def _inject_fault(args: argparse.Namespace) -> int:
    metadata = {"source": "agentctl"}
    for key in ("memory_mb", "chunk_mb", "hold_seconds"):
        value = getattr(args, key, None)
        if value is not None:
            metadata[key] = value
    event = inject_fault(
        args.type,
        task_id=args.task,
        agent_id=args.agent,
        metadata=metadata,
    )
    EventLog(Path(args.log)).append(event)
    print(
        f"fault.injected type={event.data['fault_type']} "
        f"agent={event.agent_id or '-'} task={event.task_id or '-'}"
    )
    return 0


def _build_dashboard(args: argparse.Namespace) -> int:
    output = build_dashboard(
        log_paths=args.log or [],
        benchmark_summary_path=args.benchmark_summary,
        benchmark_report_path=args.benchmark_report,
        output_path=args.output,
    )
    print(f"dashboard written: {output}")
    return 0


def _run_live(args: argparse.Namespace) -> int:
    import asyncio as _aio
    from agentos.dashboard.broadcast import EventBroadcaster
    from agentos.dashboard.server import run_dashboard_server

    async def _main() -> int:
        broadcaster = EventBroadcaster()
        log = EventLog(Path(args.log))

        server_task = _aio.create_task(
            run_dashboard_server(broadcaster, host=args.host, port=args.port)
        )

        if args.scenario != "none":
            await _aio.sleep(0.5)
            await _run_scenario(args.scenario, broadcaster, log, backend=args.backend, delay=args.delay, inject_fault=args.fault)
            print("\n  Scenario complete. Dashboard stays live. Press Ctrl+C to exit.")

        try:
            await server_task
        except _aio.CancelledError:
            pass
        return 0

    try:
        return _aio.run(_main())
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        return 0


async def _run_scenario(
    name: str,
    broadcaster: "EventBroadcaster",
    event_log: "EventLog",
    backend: str = "mock",
    delay: float = 0.0,
    inject_fault: bool = True,
) -> None:
    from agentos.llm.factory import create_backend
    from agentos.runtime.engine import Runtime

    if name == "devops":
        from agentos.scenarios.devops.incident import (
            build_devops_dag,
            DiagnoseHandler,
            FixHandler,
            MonitorHandler,
            PostmortemHandler,
            VerifyHandler,
            INCIDENT_CONTEXT,
            INFRA_SUMMARY,
        )
        from agentos.context.manager import ContextManager
        from agentos.fault.injector import FaultInjectionPlan
        from agentos.fault.recovery import ContextRetryPolicy

        ctx_mgr = ContextManager()
        ctx_mgr.add_shared("incident-context", INCIDENT_CONTEXT)
        ctx_mgr.add_shared("infra-summary", INFRA_SUMMARY)
        dag = build_devops_dag()
        rt = Runtime(
            llm_backend=create_backend(backend, **({"latency": delay} if delay > 0 else {})),
            handlers={
                "monitor": MonitorHandler(),
                "diagnoser": DiagnoseHandler(),
                "fixer": FixHandler(),
                "verifier": VerifyHandler(),
                "postmortem": PostmortemHandler(),
            },
            context_manager=ctx_mgr,
            event_log=event_log,
            fault_injection=FaultInjectionPlan({"diagnose": "kill"}),
            recovery_policy=ContextRetryPolicy(max_retries=1),
            broadcaster=broadcaster,
        )
        await rt.run(dag)
    elif name == "software_eng":
        from agentos.scenarios.software_eng.pipeline import run_software_eng

        await run_software_eng(backend=backend, log_path=event_log.path, broadcaster=broadcaster, delay=delay, inject_fault=inject_fault)
    elif name == "research":
        from agentos.scenarios.research.pipeline import run_research

        await run_research(backend=backend, log_path=event_log.path, broadcaster=broadcaster, delay=delay, inject_fault=inject_fault)
    elif name == "e2e":
        from scripts.demo_e2e_fault import run_demo

        log_path = Path(event_log.path)
        await run_demo(log_path)
        for event in EventLog(log_path).read():
            broadcaster.publish(event)
    elif name == "benchmark":
        from scripts.run_benchmark_comparison import (
            CacheSimulatingMockBackend,
            run_runtime,
        )

        backend = CacheSimulatingMockBackend()
        await run_runtime(backend, log_path=event_log.path)
        for event in EventLog(event_log.path).read():
            broadcaster.publish(event)


def _run_finals_demo(args: argparse.Namespace) -> int:
    import asyncio
    from agentos.scenarios.finals import run_finals_demo

    summary = asyncio.run(run_finals_demo(
        Path(args.output), backend_name=args.backend,
        model=args.model, max_tokens=args.max_tokens,
    ))
    print(f"finals demo: {'PASS' if summary['passed'] else 'FAIL'}; report: {args.output}/report.md")
    return 0 if summary["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentctl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="Read and print RuntimeEvent JSONL")
    trace.add_argument("--log", required=True, help="Path to events.jsonl")
    trace.add_argument("--agent", help="Filter by agent_id")
    trace.add_argument("--task", help="Filter by task_id")
    trace.add_argument("--type", help="Filter by event_type")
    trace.add_argument("--category", help="Filter by event category")
    trace.set_defaults(func=_print_trace)

    ps = subparsers.add_parser("ps", help="Aggregate latest agent state from EventLog")
    ps.add_argument("--log", required=True, help="Path to events.jsonl")
    ps.add_argument("--task", help="Filter by task_id")
    ps.set_defaults(func=_print_ps)

    context = subparsers.add_parser("context", help="Print latest context stats")
    context.add_argument("--log", required=True, help="Path to events.jsonl")
    context.set_defaults(func=_print_context)

    inject = subparsers.add_parser("inject-fault", help="Append a fault injection event")
    inject.add_argument("--log", required=True, help="Path to events.jsonl")
    inject.add_argument(
        "--type",
        required=True,
        choices=[fault_type.value for fault_type in FaultType],
    )
    inject.add_argument("--task", required=True, help="Target task_id")
    inject.add_argument("--agent", help="Target agent_id")
    inject.add_argument("--memory-mb", type=int, help="Memory pressure target in MiB")
    inject.add_argument("--chunk-mb", type=int, help="Memory pressure chunk size in MiB")
    inject.add_argument(
        "--hold-seconds",
        type=float,
        help="Memory pressure hold duration before failure",
    )
    inject.set_defaults(func=_inject_fault)

    dashboard = subparsers.add_parser(
        "dashboard",
        help="Generate a static HTML dashboard from EventLog and benchmark data",
    )
    dashboard.add_argument(
        "--log",
        action="append",
        help="Path to RuntimeEvent JSONL. Can be repeated.",
    )
    dashboard.add_argument(
        "--benchmark-summary",
        help="Path to benchmark_summary.json",
    )
    dashboard.add_argument(
        "--benchmark-report",
        help="Path to benchmark_report.md",
    )
    dashboard.add_argument(
        "--output",
        default="runs/dashboard/index.html",
        help="Output HTML path",
    )
    dashboard.set_defaults(func=_build_dashboard)

    live = subparsers.add_parser(
        "live",
        help="Start a live K9s-style dashboard with real-time event streaming",
    )
    live.add_argument(
        "--scenario",
        choices=["e2e", "devops", "benchmark", "software_eng", "research", "none"],
        default="devops",
        help="Scenario to run while dashboard is live (default: devops)",
    )
    live.add_argument("--host", default="127.0.0.1", help="Bind address")
    live.add_argument("--port", type=int, default=9400, help="Server port")
    live.add_argument(
        "--log",
        default="runs/live/events.jsonl",
        help="Path to persist events",
    )
    live.add_argument(
        "--backend",
        choices=["mock", "cache-sim", "vllm-sim", "deepseek-sim", "deepseek", "vllm"],
        default="mock",
        help="LLM backend to use (default: mock)",
    )
    live.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="LLM call latency in seconds for demo/visualization (default: 0)",
    )
    live.add_argument(
        "--fault",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fault injection for demo (default: on, use --no-fault to disable)",
    )
    live.set_defaults(func=_run_live)

    finals = subparsers.add_parser("finals-demo", help="Verify context selection, fault isolation and retry with an inspectable demo")
    finals.add_argument("--output", required=True, help="New directory for the result bundle")
    finals.add_argument("--backend", choices=["mock", "deepseek"], default="mock")
    finals.add_argument("--model", default="deepseek-chat")
    finals.add_argument("--max-tokens", type=int, default=128, help="Output cap per model call (32-512)")
    finals.set_defaults(func=_run_finals_demo)

    from agentos.execution.cli import add_commands
    add_commands(subparsers)
    from agentos.workbench.cli import add_command
    add_command(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from agentos.worker.sandbox import SandboxUnavailable
    parser = build_parser()
    values = list(sys.argv[1:] if argv is None else argv)
    commands = {"trace", "ps", "context", "inject-fault", "dashboard", "live", "finals-demo", "exec", "bench", "report", "app"}
    if values and values[0] not in commands and not values[0].startswith("-"):
        values.insert(0, "exec")
    args = parser.parse_args(values)
    try:
        return int(args.func(args))
    except (ValueError, OSError, ImportError, SandboxUnavailable) as exc:
        if getattr(args, "json", False):
            import json
            print(json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"agentctl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
