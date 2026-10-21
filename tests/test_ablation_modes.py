from pathlib import Path

from agentos.context.eviction import SegmentEvictor
from agentos.context.manager import ContextManager
from agentos.context.prefix import ContextPriority
from agentos.fault.injector import FaultInjectionPlan
from agentos.fault.recovery import ContextRetryPolicy
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import Runtime
from agentos.scenarios.software_eng.pipeline import (
    SEGMENT_BUDGET,
    PROJECT_CONTEXT,
    REPO_SUMMARY,
    _build_handlers,
    build_software_eng_dag,
)


def _max_parallel(rt: Runtime) -> int:
    in_progress: set[str] = set()
    peak = 0
    for e in rt.events:
        if e.event_type == "task.started":
            in_progress.add(e.task_id)
            peak = max(peak, len(in_progress))
        elif e.event_type == "task.completed":
            in_progress.discard(e.task_id)
    return peak


async def _run(rt: Runtime) -> Runtime:
    await rt.run(build_software_eng_dag())
    return rt


async def test_all_four_modes_run_with_mock(tmp_path: Path):
    llm = MockLLMBackend()

    ctx_shared = ContextManager(evictor=SegmentEvictor(), segment_budget=SEGMENT_BUDGET)
    ctx_shared.add_shared_ref("project-context", PROJECT_CONTEXT, priority=ContextPriority.TASK_BACKGROUND)
    ctx_shared.add_shared_ref("repo-summary", REPO_SUMMARY, priority=ContextPriority.REPO_SUMMARY)

    runtimes = {
        "bare": Runtime(llm_backend=MockLLMBackend(), handlers=_build_handlers(),
                        context_manager=ContextManager(), max_batch_size=1),
        "dag_only": Runtime(llm_backend=MockLLMBackend(), handlers=_build_handlers(),
                            context_manager=ContextManager()),
        "dag_context": Runtime(llm_backend=MockLLMBackend(), handlers=_build_handlers(),
                               context_manager=ctx_shared),
        "full": Runtime(
            llm_backend=MockLLMBackend(),
            handlers=_build_handlers(),
            context_manager=ContextManager(evictor=SegmentEvictor(), segment_budget=SEGMENT_BUDGET),
            fault_injection=FaultInjectionPlan({"security": "kill"}),
            recovery_policy=ContextRetryPolicy(max_retries=1),
        ),
    }

    for mode, rt in runtimes.items():
        dag = build_software_eng_dag()
        await rt.run(dag)
        assert dag.is_complete(), f"{mode} DAG did not complete"


async def test_bare_mode_has_no_context_events(tmp_path: Path):
    rt = Runtime(
        llm_backend=MockLLMBackend(),
        handlers=_build_handlers(),
        context_manager=ContextManager(),
        max_batch_size=1,
    )
    await _run(rt)

    context_events = [e for e in rt.events if e.event_type == "context.compressed"]
    assert len(context_events) == 0


async def test_dag_only_has_parallel_scheduling(tmp_path: Path):
    rt_bare = Runtime(
        llm_backend=MockLLMBackend(),
        handlers=_build_handlers(),
        context_manager=ContextManager(),
        max_batch_size=1,
    )
    rt_dag = Runtime(
        llm_backend=MockLLMBackend(),
        handlers=_build_handlers(),
        context_manager=ContextManager(),
    )
    await _run(rt_bare)
    await _run(rt_dag)

    assert _max_parallel(rt_bare) == 1
    assert _max_parallel(rt_dag) > 1
