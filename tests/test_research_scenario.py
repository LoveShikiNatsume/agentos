from pathlib import Path

from agentos.core.task import TaskStatus


async def test_research_dag_has_correct_structure():
    from agentos.scenarios.research.pipeline import build_research_dag

    dag = build_research_dag()

    assert len(dag.tasks) == 10
    assert dag.tasks["plan"].dependencies == []
    for rid in ("research_langgraph", "research_autogen", "research_crewai"):
        assert "plan" in dag.tasks[rid].dependencies
    for aid in ("analyze_arch", "analyze_perf", "analyze_reliability"):
        deps = dag.tasks[aid].dependencies
        assert "research_langgraph" in deps
        assert "research_autogen" in deps
        assert "research_crewai" in deps
    deps = dag.tasks["compare"].dependencies
    assert "analyze_arch" in deps
    assert "analyze_perf" in deps
    assert "analyze_reliability" in deps


async def test_research_dag_has_parallel_branches():
    from agentos.scenarios.research.pipeline import build_research_dag

    dag = build_research_dag()

    dag.tasks["plan"].status = TaskStatus.COMPLETED
    ready = dag.get_ready_tasks()
    ready_ids = {t.task_id for t in ready}
    assert ready_ids == {"research_langgraph", "research_autogen", "research_crewai"}


async def test_research_completes_all_tasks_mock(tmp_path: Path):
    from agentos.scenarios.research.pipeline import run_research

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_research(
        backend="mock", log_path=log_path,
    )

    assert len(results) == 10
    for task_id in dag.tasks:
        assert dag.tasks[task_id].status == TaskStatus.COMPLETED

    event_types = {e.event_type for e in rt.events}
    assert "agent.created" in event_types
    assert "llm.response_received" in event_types


async def test_research_with_cache_sim_produces_metrics(tmp_path: Path):
    from agentos.scenarios.research.pipeline import run_research

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_research(
        backend="cache-sim", log_path=log_path,
    )

    llm_events = [e for e in rt.events if e.event_type == "llm.response_received"]
    assert len(llm_events) >= 8
    for e in llm_events:
        assert "cache_miss_tokens" in e.data


async def test_research_with_fault_injection(tmp_path: Path):
    from agentos.scenarios.research.pipeline import run_research

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_research(
        backend="mock", log_path=log_path, inject_fault=True,
    )

    assert len(results) == 10
    for task_id in dag.tasks:
        assert dag.tasks[task_id].status == TaskStatus.COMPLETED

    event_types = {e.event_type for e in rt.events}
    assert "fault.injected" in event_types
    assert "fault.recovery_completed" in event_types


async def test_research_emits_compression(tmp_path: Path):
    from agentos.scenarios.research.pipeline import run_research

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_research(backend="mock", log_path=log_path)

    comp = [e for e in rt.events if e.event_type == "context.compressed"]
    assert comp, "expected context.compressed events"
    assert any(e.data["tokens_saved"] > 0 for e in comp)

    dropped: set[str] = set()
    for e in comp:
        dropped.update(e.data["dropped_segments"])
    assert "research-topic" not in dropped
    assert "methodology" not in dropped

    for task_id in dag.tasks:
        assert dag.tasks[task_id].status == TaskStatus.COMPLETED
