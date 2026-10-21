"""Research Analysis scenario: Three-way framework comparison.

DAG structure (8 agents):
    planner
    ├── research_langgraph ─┐
    ├── research_autogen   ─┤── (3 parallel researchers)
    └── research_crewai    ─┘
        ├── analyze_arch        ─┐
        ├── analyze_perf        ─┤── (3 parallel analysts)
        └── analyze_reliability ─┘
            └── compare
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Sequence

from agentos.context.eviction import SegmentEvictor
from agentos.context.manager import ContextManager
from agentos.context.prefix import ContextPriority
from agentos.core.task import Task, TaskDAG
from agentos.events.log import EventLog
from agentos.llm.factory import create_backend
from agentos.runtime.engine import AgentHandler, Runtime

from agentos.scenarios.research.prompts import (
    ANALYST_ARCH_PROMPT,
    ANALYST_ARCH_SCORING_PROMPT,
    ANALYST_PERF_PROMPT,
    ANALYST_PERF_SCORING_PROMPT,
    ANALYST_RELIABILITY_PROMPT,
    ANALYST_RELIABILITY_SCORING_PROMPT,
    COMPARATOR_PROMPT,
    COMPARATOR_RECOMMENDATIONS_PROMPT,
    PLANNER_PROMPT,
    PLANNER_WEIGHTS_PROMPT,
    PLANNER_INSTRUCTIONS_PROMPT,
    RESEARCH_METHODOLOGY,
    RESEARCH_TOPIC,
    RESEARCHER_AUTOGEN_PROMPT,
    RESEARCHER_AUTOGEN_SCALABILITY_PROMPT,
    RESEARCHER_AUTOGEN_ECOSYSTEM_PROMPT,
    RESEARCHER_CREWAI_PROMPT,
    RESEARCHER_CREWAI_SCALABILITY_PROMPT,
    RESEARCHER_CREWAI_ECOSYSTEM_PROMPT,
    RESEARCHER_LANGGRAPH_PROMPT,
    RESEARCHER_LANGGRAPH_SCALABILITY_PROMPT,
    RESEARCHER_LANGGRAPH_ECOSYSTEM_PROMPT,
)

# Demo budget: pinned foundational context ~154 tokens + headroom for ~3
# intermediate segments. Low enough that Level-2 eviction visibly triggers
# for downstream agents. Recalibrate if the foundational prompts change.
SEGMENT_BUDGET = 220


class PlannerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": PLANNER_PROMPT}]
        )
        ctx.context_manager.add_shared("research-plan", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": PLANNER_WEIGHTS_PROMPT}]
        )
        ctx.context_manager.add_shared("evaluation-weights", resp2.content)

        resp3 = await ctx.llm_call(
            [{"role": "user", "content": PLANNER_INSTRUCTIONS_PROMPT}]
        )
        ctx.context_manager.add_shared("collection-instructions", resp3.content)
        return resp3.content


class ResearcherLangGraphHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_LANGGRAPH_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-langgraph-arch", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_LANGGRAPH_SCALABILITY_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-langgraph-scale", resp2.content)

        resp3 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_LANGGRAPH_ECOSYSTEM_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-langgraph", resp3.content)
        return resp3.content


class ResearcherAutoGenHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_AUTOGEN_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-autogen-arch", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_AUTOGEN_SCALABILITY_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-autogen-scale", resp2.content)

        resp3 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_AUTOGEN_ECOSYSTEM_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-autogen", resp3.content)
        return resp3.content


class ResearcherCrewAIHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_CREWAI_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-crewai-arch", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_CREWAI_SCALABILITY_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-crewai-scale", resp2.content)

        resp3 = await ctx.llm_call(
            [{"role": "user", "content": RESEARCHER_CREWAI_ECOSYSTEM_PROMPT}]
        )
        ctx.context_manager.add_shared("findings-crewai", resp3.content)
        return resp3.content


class AnalystArchHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": ANALYST_ARCH_PROMPT}]
        )
        ctx.context_manager.add_shared("analysis-arch-draft", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": ANALYST_ARCH_SCORING_PROMPT}]
        )
        return resp2.content


class AnalystPerfHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": ANALYST_PERF_PROMPT}]
        )
        ctx.context_manager.add_shared("analysis-perf-draft", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": ANALYST_PERF_SCORING_PROMPT}]
        )
        return resp2.content


class AnalystReliabilityHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": ANALYST_RELIABILITY_PROMPT}]
        )
        ctx.context_manager.add_shared("analysis-reliability-draft", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": ANALYST_RELIABILITY_SCORING_PROMPT}]
        )
        return resp2.content


class ComparatorHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": COMPARATOR_PROMPT}]
        )
        resp2 = await ctx.llm_call(
            [{"role": "user", "content": COMPARATOR_RECOMMENDATIONS_PROMPT}]
        )
        return resp2.content


RESEARCHER_IDS = ("research_langgraph", "research_autogen", "research_crewai")
ANALYST_IDS = ("analyze_arch", "analyze_perf", "analyze_reliability")


def build_research_dag() -> TaskDAG:
    dag = TaskDAG()

    dag.add_task(Task(task_id="plan", name="plan", agent_role="planner"))

    dag.add_task(Task(
        task_id="research_langgraph", name="research_langgraph",
        agent_role="researcher_langgraph", dependencies=["plan"],
    ))
    dag.add_task(Task(
        task_id="research_autogen", name="research_autogen",
        agent_role="researcher_autogen", dependencies=["plan"],
    ))
    dag.add_task(Task(
        task_id="research_crewai", name="research_crewai",
        agent_role="researcher_crewai", dependencies=["plan"],
    ))

    dag.add_task(Task(
        task_id="analyze_arch", name="analyze_arch",
        agent_role="analyst_arch", dependencies=list(RESEARCHER_IDS),
    ))
    dag.add_task(Task(
        task_id="analyze_perf", name="analyze_perf",
        agent_role="analyst_perf", dependencies=list(RESEARCHER_IDS),
    ))
    dag.add_task(Task(
        task_id="analyze_reliability", name="analyze_reliability",
        agent_role="analyst_reliability", dependencies=list(RESEARCHER_IDS),
    ))

    dag.add_task(Task(
        task_id="compare", name="compare",
        agent_role="comparator", dependencies=list(ANALYST_IDS),
    ))

    dag.add_task(Task(
        task_id="validate_data", name="validate_data", agent_role="data_validator",
        dependencies=list(RESEARCHER_IDS),
        metadata={
            "execution": "subprocess",
            "worker_type": "sleep",
            "worker_payload": {"seconds": 0.4, "result": "data validated: 3 sources, 18 claims, 0 conflicts"},
        },
    ))
    dag.add_task(Task(
        task_id="export_report", name="export_report", agent_role="report_exporter",
        dependencies=["compare", "validate_data"],
        metadata={
            "execution": "subprocess",
            "worker_type": "sleep",
            "worker_payload": {"seconds": 0.6, "result": "report exported: PDF 42 pages, data appendix attached"},
        },
    ))

    return dag


def _build_handlers() -> dict[str, AgentHandler]:
    return {
        "planner": PlannerHandler(),
        "researcher_langgraph": ResearcherLangGraphHandler(),
        "researcher_autogen": ResearcherAutoGenHandler(),
        "researcher_crewai": ResearcherCrewAIHandler(),
        "analyst_arch": AnalystArchHandler(),
        "analyst_perf": AnalystPerfHandler(),
        "analyst_reliability": AnalystReliabilityHandler(),
        "comparator": ComparatorHandler(),
    }


async def run_research(
    *,
    backend: str = "mock",
    log_path: str | Path = "runs/research/events.jsonl",
    inject_fault: bool = False,
    broadcaster: Any = None,
    delay: float = 0.0,
) -> tuple[dict[str, Any], Runtime, TaskDAG]:
    from agentos.fault.injector import FaultInjectionPlan
    from agentos.fault.recovery import ContextRetryPolicy

    ctx_mgr = ContextManager(
        evictor=SegmentEvictor(),
        segment_budget=SEGMENT_BUDGET,
    )
    ctx_mgr.add_shared_ref(
        "research-topic", RESEARCH_TOPIC, priority=ContextPriority.TASK_BACKGROUND,
    )
    ctx_mgr.add_shared_ref(
        "methodology", RESEARCH_METHODOLOGY, priority=ContextPriority.REPO_SUMMARY,
    )

    dag = build_research_dag()
    log = Path(log_path)

    fault_plan = (
        FaultInjectionPlan({"research_autogen": "kill"}) if inject_fault else None
    )
    recovery = ContextRetryPolicy(max_retries=1) if inject_fault else None

    backend_kwargs = {"latency": delay} if delay > 0 else {}
    llm_backend = create_backend(backend, **backend_kwargs)

    rt = Runtime(
        llm_backend=llm_backend,
        handlers=_build_handlers(),
        context_manager=ctx_mgr,
        event_log=EventLog(log),
        fault_injection=fault_plan,
        recovery_policy=recovery,
        broadcaster=broadcaster,
    )

    results = await rt.run(dag)
    return results, rt, dag


async def _async_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-scenario")
    parser.add_argument("--backend", default="mock", choices=["mock", "cache-sim", "vllm-sim", "deepseek-sim", "deepseek", "vllm"])
    parser.add_argument("--log", default="runs/research/events.jsonl")
    parser.add_argument("--fault", action="store_true", help="Inject kill fault on autogen researcher")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    if log_path.exists():
        log_path.unlink()

    results, rt, dag = await run_research(
        backend=args.backend, log_path=log_path, inject_fault=args.fault,
    )

    print("=" * 64)
    print("  AgentOS-Runtime Research Analysis Scenario")
    print("=" * 64)
    for task_id, task in dag.tasks.items():
        result_str = str(results.get(task_id, "-"))[:60]
        print(f"  {task_id:<24} status={task.status.value:<10} result={result_str}")
    print(f"\n  Events: {len(rt.events)}")
    print(f"  Log: {log_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
