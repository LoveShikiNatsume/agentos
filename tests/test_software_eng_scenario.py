from pathlib import Path

from agentos.core.task import TaskStatus


async def test_software_eng_dag_has_correct_structure():
    from agentos.scenarios.software_eng.pipeline import build_software_eng_dag

    dag = build_software_eng_dag()

    assert len(dag.tasks) == 11
    assert dag.tasks["architect"].dependencies == []
    assert "architect" in dag.tasks["db_design"].dependencies
    assert "architect" in dag.tasks["api_design"].dependencies
    assert "db_design" in dag.tasks["backend"].dependencies
    assert "api_design" in dag.tasks["backend"].dependencies
    assert "backend" in dag.tasks["test"].dependencies
    assert "backend" in dag.tasks["security"].dependencies
    assert "backend" in dag.tasks["docs"].dependencies
    assert "backend" in dag.tasks["build"].dependencies
    assert "backend" in dag.tasks["lint"].dependencies
    assert dag.tasks["build"].metadata["execution"] == "subprocess"
    assert dag.tasks["lint"].metadata["execution"] == "subprocess"
    assert "build" in dag.tasks["integrate"].dependencies
    assert "lint" in dag.tasks["integrate"].dependencies
    assert "integrate" in dag.tasks["deploy"].dependencies
    assert dag.tasks["deploy"].metadata["execution"] == "subprocess"


async def test_software_eng_completes_all_tasks_mock(tmp_path: Path):
    from agentos.scenarios.software_eng.pipeline import run_software_eng

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_software_eng(
        backend="mock", log_path=log_path,
    )

    assert len(results) == 11
    for task_id in dag.tasks:
        assert dag.tasks[task_id].status == TaskStatus.COMPLETED

    event_types = {e.event_type for e in rt.events}
    assert "agent.created" in event_types
    assert "llm.response_received" in event_types
    assert "context.stats" in event_types
    assert "worker.started" in event_types
    assert log_path.exists()


async def test_software_eng_events_include_token_metrics(tmp_path: Path):
    from agentos.scenarios.software_eng.pipeline import run_software_eng

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_software_eng(
        backend="cache-sim", log_path=log_path,
    )

    llm_events = [e for e in rt.events if e.event_type == "llm.response_received"]
    assert len(llm_events) >= 8
    for e in llm_events:
        assert "tokens_used" in e.data
        assert "cache_miss_tokens" in e.data


async def test_software_eng_shared_context_produces_events(tmp_path: Path):
    from agentos.scenarios.software_eng.pipeline import run_software_eng

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_software_eng(
        backend="mock", log_path=log_path,
    )

    context_events = [e for e in rt.events if e.event_type == "context.stats"]
    assert len(context_events) >= 1
    stats = context_events[-1].data["stats"]
    assert stats["total_refs"] >= 2
    assert isinstance(stats.get("per_segment"), list)


async def test_software_eng_with_fault_injection(tmp_path: Path):
    from agentos.scenarios.software_eng.pipeline import run_software_eng

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_software_eng(
        backend="mock", log_path=log_path, inject_fault=True,
    )

    assert len(results) == 11
    for task_id in dag.tasks:
        assert dag.tasks[task_id].status == TaskStatus.COMPLETED

    event_types = {e.event_type for e in rt.events}
    assert "fault.injected" in event_types
    assert "fault.recovery_completed" in event_types


async def test_software_eng_emits_compression(tmp_path: Path):
    from agentos.scenarios.software_eng.pipeline import run_software_eng

    log_path = tmp_path / "events.jsonl"
    results, rt, dag = await run_software_eng(backend="mock", log_path=log_path)

    comp = [e for e in rt.events if e.event_type == "context.compressed"]
    assert comp, "expected context.compressed events"
    assert any(e.data["tokens_saved"] > 0 for e in comp)

    dropped: set[str] = set()
    for e in comp:
        dropped.update(e.data["dropped_segments"])
    # Pinned foundational context must never be evicted.
    assert "project-context" not in dropped
    assert "repo-summary" not in dropped

    # Compression must not break task completion.
    for task_id in dag.tasks:
        assert dag.tasks[task_id].status == TaskStatus.COMPLETED
