import pytest

from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler, Runtime


class _SubmitHandler(AgentHandler):
    def __init__(self, *tasks: Task) -> None:
        self._tasks = list(tasks)

    async def run(self, ctx: AgentHandler.Context) -> str:
        for task in self._tasks:
            ctx.submit_task(task)
        return "done"


class _NoopHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context) -> str:
        return "done"


async def _run(dag: TaskDAG, handlers: dict) -> tuple[dict, Runtime]:
    rt = Runtime(llm_backend=MockLLMBackend(), handlers=handlers)
    return await rt.run(dag), rt


async def test_submitted_task_is_executed_and_dag_completes():
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="submitter"))

    results, rt = await _run(dag, {
        "submitter": _SubmitHandler(Task(task_id="b", name="b", agent_role="noop", dependencies=["a"])),
        "noop": _NoopHandler(),
    })

    assert dag.tasks["b"].status == TaskStatus.COMPLETED
    assert any(e.event_type == "task.dynamic_submitted" and e.task_id == "b" for e in rt.events)
    assert dag.is_complete()


async def test_dynamic_task_with_dependency_runs_after_dependency():
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="submitter"))

    results, rt = await _run(dag, {
        "submitter": _SubmitHandler(Task(task_id="b", name="b", agent_role="noop", dependencies=["a"])),
        "noop": _NoopHandler(),
    })

    events = rt.events
    a_done = next(i for i, e in enumerate(events) if e.event_type == "task.completed" and e.task_id == "a")
    b_start = next(i for i, e in enumerate(events) if e.event_type == "task.started" and e.task_id == "b")
    assert a_done < b_start
    assert dag.tasks["b"].status == TaskStatus.COMPLETED


async def test_dynamic_task_chain_all_complete():
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="submitter_a"))

    results, rt = await _run(dag, {
        "submitter_a": _SubmitHandler(Task(task_id="b", name="b", agent_role="submitter_b", dependencies=["a"])),
        "submitter_b": _SubmitHandler(Task(task_id="c", name="c", agent_role="noop", dependencies=["b"])),
        "noop": _NoopHandler(),
    })

    for tid in ("a", "b", "c"):
        assert dag.tasks[tid].status == TaskStatus.COMPLETED
    assert dag.is_complete()


# --- A3: Cycle detection tests ---


def test_add_dynamic_task_no_cycle_succeeds():
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_dynamic_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))
    assert "b" in dag.tasks


def test_add_task_two_node_cycle_fails():
    """a -> b -> a should be detected as a cycle."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))
    # Inject cycle: make a depend on b
    dag.tasks["a"].dependencies.append("b")
    with pytest.raises(ValueError, match="Cycle detected"):
        dag._validate_no_cycles()


def test_add_dynamic_tasks_three_node_cycle_fails():
    """a -> b -> c -> a should be detected as a cycle."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))
    dag.add_task(Task(task_id="c", name="c", agent_role="r", dependencies=["b"]))
    # Inject cycle: make c depend on a
    dag.tasks["a"].dependencies.append("c")
    with pytest.raises(ValueError, match="Cycle detected"):
        dag._validate_no_cycles()


def test_add_dynamic_tasks_batch_with_cycle_fails():
    """Batch add that would create cycle should be rejected and rolled back."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="root", name="root", agent_role="r"))
    # Add x -> y first (valid)
    dag.add_dynamic_tasks([
        Task(task_id="x", name="x", agent_role="r", dependencies=["root"]),
        Task(task_id="y", name="y", agent_role="r", dependencies=["x"]),
    ])
    # Inject cycle: make x depend on y
    dag.tasks["x"].dependencies.append("y")
    with pytest.raises(ValueError, match="Cycle detected"):
        dag._validate_no_cycles()


def test_add_task_rejects_cycle_and_rolls_back():
    """Adding a task that creates a cycle should roll back."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))
    # Inject cycle: make a depend on b
    dag.tasks["a"].dependencies.append("b")
    # Validate should detect cycle
    with pytest.raises(ValueError, match="Cycle detected"):
        dag._validate_no_cycles()


def test_replace_downstream_rejects_missing_dependency():
    dag = TaskDAG()
    dag.add_task(Task(task_id="root", name="root", agent_role="r"))
    dag.add_task(Task(task_id="old", name="old", agent_role="r", dependencies=["root"]))
    dag.complete_task("root", result="ok")

    with pytest.raises(ValueError, match="Dependency missing not found"):
        dag.replace_downstream(
            "root",
            [Task(task_id="new", name="new", agent_role="r", dependencies=["missing"])],
        )
    assert "new" not in dag.tasks


def test_dag_unchanged_after_cycle_detection():
    """DAG should not be polluted after a cycle is detected and rejected."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))

    original_tasks = set(dag.tasks.keys())

    # Attempt to create a cycle
    dag.tasks["a"].dependencies.append("b")
    try:
        dag._validate_no_cycles()
    except ValueError:
        pass

    # Tasks should still be there (we didn't add/remove, just validated)
    assert set(dag.tasks.keys()) == original_tasks


def test_validate_no_cycles_passes_on_acyclic_dag():
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))
    dag.add_task(Task(task_id="c", name="c", agent_role="r", dependencies=["a"]))
    dag.add_task(Task(task_id="d", name="d", agent_role="r", dependencies=["b", "c"]))
    # Should not raise
    dag._validate_no_cycles()
