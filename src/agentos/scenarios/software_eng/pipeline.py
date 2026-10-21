"""Software Engineering scenario: Full-stack Wishlist feature development.

DAG structure (8 agents):
    architect
    ├── db_design
    └── api_design
        └── backend (depends on both db_design + api_design)
            ├── test
            ├── security
            └── docs
                └── integrate (depends on test + security + docs)
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
from agentos.core.types import Message
from agentos.events.log import EventLog
from agentos.llm.factory import create_backend
from agentos.runtime.engine import AgentHandler, Runtime

from agentos.scenarios.software_eng.prompts import (
    API_DESIGNER_PROMPT,
    API_VALIDATION_PROMPT,
    ARCHITECT_PROMPT,
    ARCHITECT_INTERFACES_PROMPT,
    ARCHITECT_INTEGRATION_PROMPT,
    BACKEND_CODER_PROMPT,
    BACKEND_ROUTING_PROMPT,
    BACKEND_ERROR_HANDLING_PROMPT,
    DB_DESIGNER_PROMPT,
    DB_MIGRATION_PROMPT,
    DOC_WRITER_PROMPT,
    DOC_QUICKSTART_PROMPT,
    INTEGRATOR_PROMPT,
    INTEGRATOR_REPORT_PROMPT,
    PROJECT_CONTEXT,
    REPO_SUMMARY,
    SECURITY_REVIEWER_PROMPT,
    SECURITY_INPUT_VALIDATION_PROMPT,
    SECURITY_RATE_LIMIT_PROMPT,
    TEST_WRITER_PROMPT,
    TEST_EDGE_CASES_PROMPT,
    TEST_INTEGRATION_PROMPT,
)

# Demo budget: pinned foundational context ~154 tokens + headroom for ~3
# intermediate segments. Low enough that Level-2 eviction visibly triggers
# for downstream agents. Recalibrate if the foundational prompts change.
SEGMENT_BUDGET = 220


class ArchitectHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": ARCHITECT_PROMPT}]
        )
        ctx.context_manager.add_shared("architect-design", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": ARCHITECT_INTERFACES_PROMPT}]
        )
        ctx.context_manager.add_shared("architect-interfaces", resp2.content)

        resp3 = await ctx.llm_call(
            [{"role": "user", "content": ARCHITECT_INTEGRATION_PROMPT}]
        )

        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-db_design",
                content={"design": resp1.content},
                message_type="design",
                task_id=ctx.task.task_id,
                trace_id=ctx.acb.trace_id,
            )
        )
        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-api_design",
                content={"design": resp1.content},
                message_type="design",
                task_id=ctx.task.task_id,
                trace_id=ctx.acb.trace_id,
            )
        )
        return resp3.content


class DbDesignerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        design = msg.content["design"] if msg else "design the database"

        resp1 = await ctx.llm_call(
            [{"role": "user", "content": f"{DB_DESIGNER_PROMPT}\nDesign: {design}"}]
        )
        ctx.context_manager.add_shared("db-schema", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": DB_MIGRATION_PROMPT}]
        )
        ctx.context_manager.add_shared("db-migration", resp2.content)
        return resp2.content


class ApiDesignerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        design = msg.content["design"] if msg else "design the API"

        resp1 = await ctx.llm_call(
            [{"role": "user", "content": f"{API_DESIGNER_PROMPT}\nDesign: {design}"}]
        )
        ctx.context_manager.add_shared("api-spec", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": API_VALIDATION_PROMPT}]
        )
        ctx.context_manager.add_shared("api-validation", resp2.content)
        return resp2.content


class BackendCoderHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": BACKEND_CODER_PROMPT}]
        )
        ctx.context_manager.add_shared("backend-service", resp1.content)

        resp2 = await ctx.llm_call(
            [{"role": "user", "content": BACKEND_ROUTING_PROMPT}]
        )
        ctx.context_manager.add_shared("backend-routing", resp2.content)

        resp3 = await ctx.llm_call(
            [{"role": "user", "content": BACKEND_ERROR_HANDLING_PROMPT}]
        )

        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-test",
                content={"code": resp1.content},
                message_type="code",
                task_id=ctx.task.task_id,
                trace_id=ctx.acb.trace_id,
            )
        )
        await ctx.bus.send(
            Message(
                from_agent=ctx.acb.agent_id,
                to_agent="agent-security",
                content={"code": resp1.content},
                message_type="code",
                task_id=ctx.task.task_id,
                trace_id=ctx.acb.trace_id,
            )
        )
        return resp3.content


class TestWriterHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        code = msg.content["code"] if msg else "wishlist service code"

        resp1 = await ctx.llm_call(
            [{"role": "user", "content": f"{TEST_WRITER_PROMPT}\nCode: {code}"}]
        )
        resp2 = await ctx.llm_call(
            [{"role": "user", "content": TEST_EDGE_CASES_PROMPT}]
        )
        resp3 = await ctx.llm_call(
            [{"role": "user", "content": TEST_INTEGRATION_PROMPT}]
        )
        return resp3.content


class SecurityReviewerHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        msg = await ctx.bus.receive(ctx.acb.agent_id)
        code = msg.content["code"] if msg else "wishlist service code"

        resp1 = await ctx.llm_call(
            [{"role": "user", "content": f"{SECURITY_REVIEWER_PROMPT}\nCode: {code}"}]
        )
        resp2 = await ctx.llm_call(
            [{"role": "user", "content": SECURITY_INPUT_VALIDATION_PROMPT}]
        )
        resp3 = await ctx.llm_call(
            [{"role": "user", "content": SECURITY_RATE_LIMIT_PROMPT}]
        )
        ctx.context_manager.add_private(
            ctx.acb.agent_id,
            "CONFIDENTIAL: Internal vulnerability report — SQL injection in product_id "
            "parameter, IDOR on wishlist endpoints, missing rate limit on check API.",
        )
        return resp3.content


class DocWriterHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        resp1 = await ctx.llm_call(
            [{"role": "user", "content": DOC_WRITER_PROMPT}]
        )
        resp2 = await ctx.llm_call(
            [{"role": "user", "content": DOC_QUICKSTART_PROMPT}]
        )
        return resp2.content


class IntegratorHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        from agentos.context.types import ContextAccessDenied

        try:
            ctx.context_manager.read(ctx.acb.agent_id, "agent-security:private:0")
        except (ContextAccessDenied, KeyError):
            pass

        resp1 = await ctx.llm_call(
            [{"role": "user", "content": INTEGRATOR_PROMPT}]
        )
        resp2 = await ctx.llm_call(
            [{"role": "user", "content": INTEGRATOR_REPORT_PROMPT}]
        )
        return resp2.content


def build_software_eng_dag() -> TaskDAG:
    dag = TaskDAG()
    dag.add_task(Task(task_id="architect", name="architect", agent_role="architect"))
    dag.add_task(Task(
        task_id="db_design", name="db_design", agent_role="db_designer",
        dependencies=["architect"],
    ))
    dag.add_task(Task(
        task_id="api_design", name="api_design", agent_role="api_designer",
        dependencies=["architect"],
    ))
    dag.add_task(Task(
        task_id="backend", name="backend", agent_role="backend_coder",
        dependencies=["db_design", "api_design"],
    ))
    dag.add_task(Task(
        task_id="test", name="test", agent_role="test_writer",
        dependencies=["backend"],
    ))
    dag.add_task(Task(
        task_id="security", name="security", agent_role="security_reviewer",
        dependencies=["backend"],
    ))
    dag.add_task(Task(
        task_id="docs", name="docs", agent_role="doc_writer",
        dependencies=["backend"],
    ))
    dag.add_task(Task(
        task_id="build", name="build", agent_role="builder",
        dependencies=["backend"],
        metadata={
            "execution": "subprocess",
            "worker_type": "sleep",
            "worker_payload": {"seconds": 0.5, "result": "build succeeded: 0 errors, 3 warnings"},
        },
    ))
    dag.add_task(Task(
        task_id="lint", name="lint", agent_role="linter",
        dependencies=["backend"],
        metadata={
            "execution": "subprocess",
            "worker_type": "sleep",
            "worker_payload": {"seconds": 0.3, "result": "lint passed: 0 errors, 12 info"},
        },
    ))
    dag.add_task(Task(
        task_id="integrate", name="integrate", agent_role="integrator",
        dependencies=["test", "security", "docs", "build", "lint"],
    ))
    dag.add_task(Task(
        task_id="deploy", name="deploy", agent_role="deployer",
        dependencies=["integrate"],
        metadata={
            "execution": "subprocess",
            "worker_type": "sleep",
            "worker_payload": {"seconds": 1.0, "result": "deployed to staging: health check passed"},
        },
    ))
    return dag


def _build_handlers() -> dict[str, AgentHandler]:
    return {
        "architect": ArchitectHandler(),
        "db_designer": DbDesignerHandler(),
        "api_designer": ApiDesignerHandler(),
        "backend_coder": BackendCoderHandler(),
        "test_writer": TestWriterHandler(),
        "security_reviewer": SecurityReviewerHandler(),
        "doc_writer": DocWriterHandler(),
        "integrator": IntegratorHandler(),
    }


async def run_software_eng(
    *,
    backend: str = "mock",
    log_path: str | Path = "runs/software_eng/events.jsonl",
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
        "project-context", PROJECT_CONTEXT, priority=ContextPriority.TASK_BACKGROUND,
    )
    ctx_mgr.add_shared_ref(
        "repo-summary", REPO_SUMMARY, priority=ContextPriority.REPO_SUMMARY,
    )

    dag = build_software_eng_dag()
    log = Path(log_path)

    fault_plan = FaultInjectionPlan({"security": "kill"}) if inject_fault else None
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
    parser = argparse.ArgumentParser(prog="software-eng-scenario")
    parser.add_argument("--backend", default="mock", choices=["mock", "cache-sim", "vllm-sim", "deepseek-sim", "deepseek", "vllm"])
    parser.add_argument("--log", default="runs/software_eng/events.jsonl")
    parser.add_argument("--fault", action="store_true", help="Inject kill fault on security reviewer")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    if log_path.exists():
        log_path.unlink()

    results, rt, dag = await run_software_eng(
        backend=args.backend, log_path=log_path, inject_fault=args.fault,
    )

    print("=" * 64)
    print("  AgentOS-Runtime Software Engineering Scenario")
    print("=" * 64)
    for task_id, task in dag.tasks.items():
        result_str = str(results.get(task_id, "-"))[:60]
        print(f"  {task_id:<12} status={task.status.value:<10} result={result_str}")
    print(f"\n  Events: {len(rt.events)}")
    print(f"  Log: {log_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
