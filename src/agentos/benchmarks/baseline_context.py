"""Lightweight prefix cache experiment framework.

Runs Baseline vs Runtime comparison by directly calling LLMBackend
with crafted prompts. Does not require full Runtime/AgentHandler stack.
Designed for cloud API experiments and mock-based validation.
"""
from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentos.llm.base import LLMBackend, LLMRequest, LLMResponse
from agentos.metrics.model import ModelCallMetrics


@dataclass
class ExperimentScenario:
    name: str
    agents: list[tuple[str, str]]  # (agent_name, role_description)
    shared_context: str
    questions: list[str]
    max_tokens: int = 30


@dataclass
class ExperimentResult:
    scenario: str
    group: str  # "baseline" or "runtime"
    calls: list[dict[str, Any]] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_cache_hit_tokens: int = 0
    total_cache_miss_tokens: int = 0
    total_cache_hit_ratio: float = 0.0
    avg_latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CODING_3AGENT = ExperimentScenario(
    name="coding_3agent",
    agents=[
        ("planner", "You are a project planner. Analyze tasks and create execution plans."),
        ("coder", "You are a software developer. Write clean, tested code."),
        ("tester", "You are a QA engineer. Design test cases and verify correctness."),
    ],
    shared_context=(
        "The AgentOS-Runtime project builds an OS-level runtime for managing "
        "multiple AI agents sharing GPU resources. It provides agent lifecycle "
        "management, DAG-based task scheduling, shared context pooling with "
        "ContextRef, prefix canonicalization for KV cache optimization, a unified "
        "model gateway with cache metrics tracking, and fault tolerance with "
        "recovery strategies."
    ),
    questions=[
        "What should we implement next? One sentence.",
        "What is the key challenge? One sentence.",
        "How do we verify correctness? One sentence.",
    ],
)

CODING_5AGENT = ExperimentScenario(
    name="coding_5agent",
    agents=[
        ("planner", "You are a project planner. Analyze tasks and create execution plans."),
        ("coder", "You are a software developer. Write clean, tested code."),
        ("tester", "You are a QA engineer. Design test cases and verify correctness."),
        ("reviewer", "You are a code reviewer. Check code quality and suggest improvements."),
        ("deployer", "You are a deployment engineer. Manage releases and infrastructure."),
    ],
    shared_context=CODING_3AGENT.shared_context,
    questions=[
        "What should we implement next? One sentence.",
        "What is the key challenge? One sentence.",
        "How do we verify correctness? One sentence.",
        "What code patterns should we follow? One sentence.",
        "What deployment risks exist? One sentence.",
    ],
)

MINIMAL_2AGENT = ExperimentScenario(
    name="minimal_2agent",
    agents=[
        ("alice", "You are Alice, a data analyst."),
        ("bob", "You are Bob, a backend engineer."),
    ],
    shared_context="We are building a data pipeline that processes user events.",
    questions=[
        "What is the first step? One sentence.",
        "What tools should we use? One sentence.",
    ],
)


class BaselineExperiment:
    def __init__(
        self,
        backend: LLMBackend,
        request_interval: float = 0.5,
        retries: int = 3,
    ) -> None:
        self._backend = backend
        self._interval = request_interval
        self._retries = retries

    async def _call(self, request: LLMRequest) -> LLMResponse:
        for attempt in range(self._retries):
            try:
                return await self._backend.generate(request)
            except Exception:
                if attempt < self._retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    raise

    async def run_baseline(self, scenario: ExperimentScenario) -> ExperimentResult:
        calls: list[dict[str, Any]] = []
        for (name, role), question in zip(scenario.agents, scenario.questions):
            req = LLMRequest(
                agent_id=name,
                messages=[
                    {"role": "system", "content": f"{role}\n\nProject context:\n{scenario.shared_context}"},
                    {"role": "user", "content": question},
                ],
                max_tokens=scenario.max_tokens,
            )
            resp = await self._call(req)
            if resp.metrics:
                calls.append(resp.metrics.to_dict())
            if self._interval > 0:
                await asyncio.sleep(self._interval)

        return self._aggregate("baseline", scenario.name, calls)

    async def run_runtime(self, scenario: ExperimentScenario) -> ExperimentResult:
        calls: list[dict[str, Any]] = []
        for (name, role), question in zip(scenario.agents, scenario.questions):
            req = LLMRequest(
                agent_id=name,
                messages=[
                    {"role": "system", "content": f"{scenario.shared_context}\n\nYour role: {role}"},
                    {"role": "user", "content": question},
                ],
                max_tokens=scenario.max_tokens,
            )
            resp = await self._call(req)
            if resp.metrics:
                calls.append(resp.metrics.to_dict())
            if self._interval > 0:
                await asyncio.sleep(self._interval)

        return self._aggregate("runtime", scenario.name, calls)

    async def run_comparison(
        self,
        scenario: ExperimentScenario,
        cooldown: float = 10.0,
        runs: int = 1,
    ) -> dict[str, ExperimentResult]:
        all_baseline_calls: list[dict[str, Any]] = []
        all_runtime_calls: list[dict[str, Any]] = []

        for _ in range(runs):
            bl = await self.run_baseline(scenario)
            all_baseline_calls.extend(bl.calls)

            if cooldown > 0:
                await asyncio.sleep(cooldown)

            rt = await self.run_runtime(scenario)
            all_runtime_calls.extend(rt.calls)

            if cooldown > 0 and runs > 1:
                await asyncio.sleep(cooldown)

        return {
            "baseline": self._aggregate("baseline", scenario.name, all_baseline_calls),
            "runtime": self._aggregate("runtime", scenario.name, all_runtime_calls),
        }

    @staticmethod
    def _aggregate(group: str, scenario: str, calls: list[dict[str, Any]]) -> ExperimentResult:
        total_prompt = sum(c.get("prompt_tokens", 0) for c in calls)
        total_hit = sum(c.get("cache_hit_tokens", 0) for c in calls)
        total_miss = sum(c.get("cache_miss_tokens", 0) for c in calls)
        hit_plus_miss = total_hit + total_miss
        ratio = total_hit / hit_plus_miss if hit_plus_miss > 0 else 0.0
        latencies = [c.get("latency_secs", 0.0) for c in calls]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

        return ExperimentResult(
            scenario=scenario,
            group=group,
            calls=calls,
            total_prompt_tokens=total_prompt,
            total_cache_hit_tokens=total_hit,
            total_cache_miss_tokens=total_miss,
            total_cache_hit_ratio=round(ratio, 4),
            avg_latency=round(avg_lat, 4),
        )

    @staticmethod
    def export_json(results: dict[str, ExperimentResult], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in results.items()}
        data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @staticmethod
    def export_csv(results: dict[str, ExperimentResult], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for group, result in results.items():
            rows.append({
                "group": group,
                "scenario": result.scenario,
                "total_prompt_tokens": result.total_prompt_tokens,
                "cache_hit_tokens": result.total_cache_hit_tokens,
                "cache_miss_tokens": result.total_cache_miss_tokens,
                "cache_hit_ratio": result.total_cache_hit_ratio,
                "avg_latency": result.avg_latency,
                "call_count": len(result.calls),
            })
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return p
