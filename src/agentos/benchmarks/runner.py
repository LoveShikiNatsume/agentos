from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from agentos.context.manager import ContextManager
from agentos.core.task import Task, TaskDAG
from agentos.llm.base import LLMBackend, LLMRequest, LLMResponse
from agentos.llm.mock import MockLLMBackend
from agentos.llm.vllm_backend import VLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


@dataclass
class BenchmarkResult:
    mode: str  # "baseline" or "runtime"
    scenario: str
    total_latency_secs: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cache_hit_tokens: int = 0
    total_cache_miss_tokens: int = 0
    cache_hit_ratio: float = 0.0
    llm_call_count: int = 0
    recovery_time_secs: float = 0.0
    fault_count: int = 0
    event_count: int = 0
    task_results: dict[str, str] = field(default_factory=dict)
    per_call_metrics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


async def check_vllm_health(
    base_url: str,
    model: str | None = None,
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Validate local vLLM endpoints and return resolved model metadata."""
    base_url = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            models_resp = await client.get(f"{base_url}/v1/models")
            models_resp.raise_for_status()
            models_data = models_resp.json()
            model_ids = [str(item["id"]) for item in models_data.get("data", [])]
            resolved_model = model or (model_ids[0] if model_ids else None)
            if resolved_model is None:
                raise RuntimeError("vLLM /v1/models returned no models")

            metrics_resp = await client.get(f"{base_url}/metrics")
            metrics_resp.raise_for_status()
            metrics_text = metrics_resp.text
    except Exception as exc:
        raise RuntimeError(f"vLLM health check failed for {base_url}: {exc}") from exc

    return {
        "base_url": base_url,
        "model": resolved_model,
        "models": model_ids,
        "metrics_available": True,
        "prefix_cache_metrics": "prefix_cache" in metrics_text,
    }


class _MetricsCollector:
    """Wraps an LLMBackend to collect per-call metrics."""

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        resp = await self._backend.generate(request)
        elapsed = time.monotonic() - start

        entry: dict[str, Any] = {
            "agent_id": request.agent_id,
            "latency_secs": elapsed,
            "tokens_used": resp.tokens_used,
        }
        if resp.metrics:
            entry.update({
                "prompt_tokens": resp.metrics.prompt_tokens,
                "completion_tokens": resp.metrics.completion_tokens,
                "cache_hit_tokens": resp.metrics.cache_hit_tokens,
                "cache_miss_tokens": resp.metrics.cache_miss_tokens,
                "cache_hit_ratio": resp.metrics.cache_hit_ratio,
            })
        self.calls.append(entry)
        return resp


class BenchmarkRunner:
    """Run baseline vs. runtime comparison for a coding-style scenario."""

    def __init__(
        self,
        backend: LLMBackend,
        handlers: dict[str, AgentHandler],
        scenario_name: str = "coding_pipeline",
    ) -> None:
        self._backend = backend
        self._handlers = handlers
        self._scenario_name = scenario_name

    def _build_dag(self, tasks: list[Task]) -> TaskDAG:
        dag = TaskDAG()
        for t in tasks:
            dag.add_task(t)
        return dag

    async def run_baseline(
        self,
        tasks: list[Task],
        shared_context: dict[str, str] | None = None,
    ) -> BenchmarkResult:
        """Baseline: each agent builds its own prompt, no shared ContextRef."""
        collector = _MetricsCollector(self._backend)
        dag = self._build_dag(tasks)
        rt = Runtime(
            llm_backend=collector,
            handlers=self._handlers,
        )
        start = time.monotonic()
        results = await rt.run(dag)
        elapsed = time.monotonic() - start

        return self._aggregate(collector, results, elapsed, "baseline", rt.events)

    async def run_runtime(
        self,
        tasks: list[Task],
        shared_context: dict[str, str] | None = None,
    ) -> BenchmarkResult:
        """Runtime: shared ContextRef + ModelGateway + PrefixCanonicalizer."""
        collector = _MetricsCollector(self._backend)
        ctx_mgr = ContextManager()
        if shared_context:
            for key, content in shared_context.items():
                ctx_mgr.add_shared_ref(key, content)

        dag = self._build_dag(tasks)
        rt = Runtime(
            llm_backend=collector,
            handlers=self._handlers,
            context_manager=ctx_mgr,
        )
        start = time.monotonic()
        results = await rt.run(dag)
        elapsed = time.monotonic() - start

        return self._aggregate(collector, results, elapsed, "runtime", rt.events)

    async def run_comparison(
        self,
        tasks_baseline: list[Task],
        tasks_runtime: list[Task],
        shared_context: dict[str, str] | None = None,
    ) -> dict[str, BenchmarkResult]:
        baseline = await self.run_baseline(tasks_baseline)
        runtime = await self.run_runtime(tasks_runtime, shared_context)
        return {"baseline": baseline, "runtime": runtime}

    def _aggregate(
        self,
        collector: _MetricsCollector,
        results: dict[str, Any],
        total_time: float,
        mode: str,
        events: list[Any] | None = None,
    ) -> BenchmarkResult:
        total_prompt = sum(c.get("prompt_tokens", 0) for c in collector.calls)
        total_completion = sum(c.get("completion_tokens", 0) for c in collector.calls)
        total_hit = sum(c.get("cache_hit_tokens", 0) for c in collector.calls)
        total_miss = sum(c.get("cache_miss_tokens", 0) for c in collector.calls)
        hit_plus_miss = total_hit + total_miss
        ratio = total_hit / hit_plus_miss if hit_plus_miss > 0 else 0.0
        events = events or []
        fault_events = [e for e in events if getattr(e, "category", None).value == "fault"]
        recovery_time = sum(
            float(getattr(e, "data", {}).get("recovery_time_secs", 0.0))
            for e in fault_events
        )

        return BenchmarkResult(
            mode=mode,
            scenario=self._scenario_name,
            total_latency_secs=total_time,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cache_hit_tokens=total_hit,
            total_cache_miss_tokens=total_miss,
            cache_hit_ratio=ratio,
            llm_call_count=len(collector.calls),
            recovery_time_secs=recovery_time,
            fault_count=len(fault_events),
            event_count=len(events),
            task_results={k: str(v) for k, v in results.items()},
            per_call_metrics=collector.calls,
        )

    @staticmethod
    def save_results(
        results: dict[str, BenchmarkResult],
        output_dir: str | Path,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "benchmark_results.json"
        data = {k: v.to_dict() for k, v in results.items()}
        output_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output_file

    @staticmethod
    def save_csv(
        results: dict[str, BenchmarkResult],
        output_dir: str | Path,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "benchmark_results.csv"
        rows = [result.to_dict() for result in results.values()]
        fields = [
            "mode",
            "scenario",
            "total_latency_secs",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_cache_hit_tokens",
            "total_cache_miss_tokens",
            "cache_hit_ratio",
            "llm_call_count",
            "recovery_time_secs",
            "fault_count",
            "event_count",
        ]
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        return output_file

    @staticmethod
    def _result_fields() -> list[str]:
        return [
            "total_latency_secs",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_cache_hit_tokens",
            "total_cache_miss_tokens",
            "cache_hit_ratio",
            "llm_call_count",
            "recovery_time_secs",
            "fault_count",
            "event_count",
        ]

    @staticmethod
    def summarize_runs(
        runs: list[dict[str, BenchmarkResult]],
    ) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"mean": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }

        modes = sorted({mode for run in runs for mode in run})
        summary: dict[str, Any] = {
            "run_count": len(runs),
            "modes": {},
            "comparison": {},
        }
        for mode in modes:
            mode_results = [run[mode] for run in runs if mode in run]
            summary["modes"][mode] = {
                field: stats([float(getattr(result, field)) for result in mode_results])
                for field in BenchmarkRunner._result_fields()
            }

        if "baseline" in summary["modes"] and "runtime" in summary["modes"]:
            baseline = summary["modes"]["baseline"]
            runtime = summary["modes"]["runtime"]
            baseline_latency = baseline["total_latency_secs"]["mean"]
            runtime_latency = runtime["total_latency_secs"]["mean"]
            baseline_prompt = baseline["total_prompt_tokens"]["mean"]
            runtime_prompt = runtime["total_prompt_tokens"]["mean"]
            prompt_savings = baseline_prompt - runtime_prompt
            summary["comparison"] = {
                "latency_delta_secs": runtime_latency - baseline_latency,
                "latency_delta_pct": (
                    (runtime_latency - baseline_latency) / baseline_latency
                    if baseline_latency
                    else 0.0
                ),
                "prompt_token_savings": prompt_savings,
                "prompt_token_savings_pct": (
                    prompt_savings / baseline_prompt if baseline_prompt else 0.0
                ),
                "cache_hit_ratio_delta": (
                    runtime["cache_hit_ratio"]["mean"]
                    - baseline["cache_hit_ratio"]["mean"]
                ),
                "event_count_delta": (
                    runtime["event_count"]["mean"] - baseline["event_count"]["mean"]
                ),
                "fault_count_delta": (
                    runtime["fault_count"]["mean"] - baseline["fault_count"]["mean"]
                ),
            }
        return summary

    @staticmethod
    def collect_environment(
        *,
        backend_name: str,
        base_url: str,
        model: str | None,
        health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        env: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "backend": backend_name,
            "base_url": base_url,
            "model": model,
            "vllm_health": health or {},
        }
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "name": parts[0],
                                "memory_total_mb": parts[1],
                                "driver_version": parts[2],
                            }
                        )
                env["gpu"] = gpus
        except (OSError, subprocess.TimeoutExpired):
            env["gpu"] = []
        return env

    @staticmethod
    def save_formal_artifacts(
        runs: list[dict[str, BenchmarkResult]],
        output_dir: str | Path,
        *,
        environment: dict[str, Any],
        command: list[str] | None = None,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = BenchmarkRunner.summarize_runs(runs)
        run_payloads = [
            {
                "run": index + 1,
                "results": {mode: result.to_dict() for mode, result in run.items()},
            }
            for index, run in enumerate(runs)
        ]

        latest = runs[-1] if runs else {}
        results_payload: dict[str, Any] = {
            **{mode: result.to_dict() for mode, result in latest.items()},
            "runs": run_payloads,
            "environment": environment,
        }
        results_path = output_dir / "benchmark_results.json"
        results_path.write_text(
            json.dumps(results_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        csv_path = output_dir / "benchmark_results.csv"
        fields = ["run", "mode", "scenario", *BenchmarkRunner._result_fields()]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for index, run in enumerate(runs, start=1):
                for mode, result in run.items():
                    row = result.to_dict()
                    writer.writerow(
                        {
                            "run": index,
                            "mode": mode,
                            "scenario": result.scenario,
                            **{
                                field: row.get(field, "")
                                for field in BenchmarkRunner._result_fields()
                            },
                        }
                    )

        summary_path = output_dir / "benchmark_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        report_path = output_dir / "benchmark_report.md"
        report_path.write_text(
            BenchmarkRunner.render_markdown_report(
                summary,
                environment,
                command=command,
            ),
            encoding="utf-8",
        )
        return {
            "results": results_path,
            "csv": csv_path,
            "summary": summary_path,
            "report": report_path,
        }

    @staticmethod
    def render_markdown_report(
        summary: dict[str, Any],
        environment: dict[str, Any],
        *,
        command: list[str] | None = None,
    ) -> str:
        def fmt(value: Any, digits: int = 3) -> str:
            return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else str(value)

        lines = [
            "# AgentOS Runtime vLLM Benchmark Report",
            "",
            "## Environment",
            "",
            f"- Timestamp: `{environment.get('timestamp', '-')}`",
            f"- Backend: `{environment.get('backend', '-')}`",
            f"- Base URL: `{environment.get('base_url', '-')}`",
            f"- Model: `{environment.get('model', '-')}`",
            f"- Python: `{environment.get('python', '-')}`",
            f"- Platform: `{environment.get('platform', '-')}`",
        ]
        if command:
            lines.append(f"- Command: `{' '.join(command)}`")
        for gpu in environment.get("gpu", []) or []:
            lines.append(
                "- GPU: "
                f"`{gpu.get('name', '-')}`, "
                f"{gpu.get('memory_total_mb', '-')} MB, "
                f"driver `{gpu.get('driver_version', '-')}`"
            )

        lines.extend([
            "",
            "## Summary",
            "",
            f"- Runs: `{summary.get('run_count', 0)}`",
            "",
            "| Mode | Latency Mean(s) | Prompt Tokens | Completion Tokens | Cache Hit Ratio | Faults | Recovery Mean(s) | Events |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for mode, values in summary.get("modes", {}).items():
            lines.append(
                f"| {mode} "
                f"| {fmt(values['total_latency_secs']['mean'])} "
                f"| {fmt(values['total_prompt_tokens']['mean'], 1)} "
                f"| {fmt(values['total_completion_tokens']['mean'], 1)} "
                f"| {fmt(values['cache_hit_ratio']['mean'] * 100, 2)}% "
                f"| {fmt(values['fault_count']['mean'], 1)} "
                f"| {fmt(values['recovery_time_secs']['mean'])} "
                f"| {fmt(values['event_count']['mean'], 1)} |"
            )

        comparison = summary.get("comparison", {})
        lines.extend([
            "",
            "## Runtime vs Baseline",
            "",
            f"- Latency delta: `{fmt(comparison.get('latency_delta_secs', 0.0))}s` "
            f"(`{fmt(comparison.get('latency_delta_pct', 0.0) * 100, 2)}%`)",
            f"- Prompt token savings: `{fmt(comparison.get('prompt_token_savings', 0.0), 1)}` "
            f"(`{fmt(comparison.get('prompt_token_savings_pct', 0.0) * 100, 2)}%`)",
            f"- Cache hit ratio delta: "
            f"`{fmt(comparison.get('cache_hit_ratio_delta', 0.0) * 100, 2)}%`",
            "",
        ])
        return "\n".join(lines)

    @staticmethod
    def print_comparison(results: dict[str, BenchmarkResult]) -> str:
        lines = ["=== Benchmark Comparison ===", ""]
        for mode, r in results.items():
            lines.append(f"--- {mode.upper()} ---")
            lines.append(f"  Latency:          {r.total_latency_secs:.3f}s")
            lines.append(f"  LLM calls:        {r.llm_call_count}")
            lines.append(f"  Prompt tokens:    {r.total_prompt_tokens}")
            lines.append(f"  Completion tokens:{r.total_completion_tokens}")
            lines.append(f"  Cache hit tokens: {r.total_cache_hit_tokens}")
            lines.append(f"  Cache miss tokens:{r.total_cache_miss_tokens}")
            lines.append(f"  Cache hit ratio:  {r.cache_hit_ratio:.2%}")
            lines.append(f"  Faults:           {r.fault_count}")
            lines.append(f"  Events:           {r.event_count}")
            lines.append("")
        output = "\n".join(lines)
        print(output)
        return output


class _BenchmarkHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp = await ctx.llm_call(
            [{"role": "user", "content": f"Complete benchmark task: {ctx.task.name}"}]
        )
        return resp.content


def _benchmark_tasks() -> list[Task]:
    return [
        Task(task_id="plan", name="plan fibonacci implementation", agent_role="worker"),
        Task(
            task_id="code",
            name="write fibonacci code",
            agent_role="worker",
            dependencies=["plan"],
        ),
        Task(
            task_id="test",
            name="write fibonacci tests",
            agent_role="worker",
            dependencies=["code"],
        ),
    ]


def _build_backend(name: str, base_url: str, model: str | None) -> LLMBackend:
    if name == "mock":
        return MockLLMBackend(responses={"default": "benchmark response"})
    if name == "vllm":
        return VLLMBackend(base_url=base_url, model=model)
    raise ValueError(f"unsupported backend: {name}")


async def run_benchmark_cli(
    *,
    backend_name: str,
    output: str | Path,
    base_url: str = "http://localhost:8000",
    model: str | None = None,
    runs: int = 1,
    cooldown: float = 0.0,
    report: bool = False,
    command: list[str] | None = None,
) -> dict[str, BenchmarkResult]:
    if runs < 1:
        raise ValueError("runs must be >= 1")
    health: dict[str, Any] | None = None
    resolved_model = model
    if backend_name == "vllm":
        health = await check_vllm_health(base_url, model)
        resolved_model = str(health["model"])

    backend = _build_backend(backend_name, base_url, resolved_model)
    runner = BenchmarkRunner(
        backend=backend,
        handlers={"worker": _BenchmarkHandler()},
        scenario_name="coding_pipeline",
    )
    environment = BenchmarkRunner.collect_environment(
        backend_name=backend_name,
        base_url=base_url,
        model=resolved_model or backend_name,
        health=health,
    )

    all_runs: list[dict[str, BenchmarkResult]] = []
    for index in range(runs):
        if index > 0 and cooldown > 0:
            await asyncio.sleep(cooldown)
        results = await runner.run_comparison(
            tasks_baseline=_benchmark_tasks(),
            tasks_runtime=_benchmark_tasks(),
            shared_context={
                "project": (
                    "AgentOS Runtime benchmark: multiple coding agents share a "
                    "stable project context to improve prefix cache locality."
                )
            },
        )
        all_runs.append(results)
        run_dir = Path(output) / f"run_{index + 1:03d}"
        BenchmarkRunner.save_results(results, run_dir)
        BenchmarkRunner.save_csv(results, run_dir)
        BenchmarkRunner.print_comparison(results)

    latest = all_runs[-1]
    BenchmarkRunner.save_results(latest, output)
    BenchmarkRunner.save_csv(latest, output)
    if report or runs > 1:
        BenchmarkRunner.save_formal_artifacts(
            all_runs,
            output,
            environment=environment,
            command=command,
        )
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentos.benchmarks.runner")
    parser.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model")
    parser.add_argument("--output", default="runs/latest")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)

    asyncio.run(
        run_benchmark_cli(
            backend_name=args.backend,
            output=args.output,
            base_url=args.base_url,
            model=args.model,
            runs=args.runs,
            cooldown=args.cooldown,
            report=args.report,
            command=["agentos.benchmarks.runner", *(argv or [])],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
