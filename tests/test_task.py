import pytest

from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG, TaskStatus


def test_task_creation():
    task = Task(task_id="t1", name="analyze", agent_role="analyzer")
    assert task.status == TaskStatus.PENDING
    assert task.dependencies == []


def test_dag_add_and_ready():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="plan", agent_role="planner"))
    dag.add_task(
        Task(task_id="t2", name="code", agent_role="coder", dependencies=["t1"])
    )
    dag.add_task(
        Task(task_id="t3", name="test", agent_role="tester", dependencies=["t2"])
    )

    ready = dag.get_ready_tasks()
    assert [task.task_id for task in ready] == ["t1"]


def test_dag_completion_unlocks_dependents():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="plan", agent_role="planner"))
    dag.add_task(
        Task(task_id="t2", name="code", agent_role="coder", dependencies=["t1"])
    )

    dag.complete_task("t1", result="plan done")
    assert dag.tasks["t1"].status == TaskStatus.COMPLETED

    ready = dag.get_ready_tasks()
    assert [task.task_id for task in ready] == ["t2"]


def test_ready_tasks_must_be_pending_with_completed_dependencies():
    dag = TaskDAG()
    dag.add_task(Task(task_id="done", name="done", agent_role="worker"))
    dag.add_task(Task(task_id="running", name="running", agent_role="worker"))
    dag.add_task(Task(task_id="failed", name="failed", agent_role="worker"))
    dag.add_task(Task(task_id="blocked", name="blocked", agent_role="worker", dependencies=["running"]))
    dag.add_task(Task(task_id="ready", name="ready", agent_role="worker", dependencies=["done"]))

    dag.complete_task("done", result="done")
    dag.tasks["running"].status = TaskStatus.RUNNING
    dag.fail_task("failed", error="boom")

    ready = dag.get_ready_tasks()

    assert [task.task_id for task in ready] == ["ready"]


def test_dag_parallel_tasks():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="plan", agent_role="planner"))
    dag.add_task(
        Task(task_id="t2", name="code-a", agent_role="coder", dependencies=["t1"])
    )
    dag.add_task(
        Task(task_id="t3", name="code-b", agent_role="coder", dependencies=["t1"])
    )

    dag.complete_task("t1", result="ok")
    ready = dag.get_ready_tasks()
    assert len(ready) == 2
    assert {task.task_id for task in ready} == {"t2", "t3"}


def test_dag_is_complete():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="only", agent_role="worker"))
    assert not dag.is_complete()
    dag.complete_task("t1", result="done")
    assert dag.is_complete()


def test_dag_fail_task():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="risky", agent_role="worker"))
    dag.fail_task("t1", error="timeout")
    assert dag.tasks["t1"].status == TaskStatus.FAILED
    assert dag.tasks["t1"].error == "timeout"


def test_dag_rejects_duplicate_task_id():
    dag = TaskDAG()
    dag.add_task(Task(task_id="t1", name="first", agent_role="worker"))
    try:
        dag.add_task(Task(task_id="t1", name="second", agent_role="worker"))
    except ValueError as exc:
        assert "Task t1 already exists" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_dag_rejects_unknown_dependency():
    dag = TaskDAG()
    try:
        dag.add_task(
            Task(
                task_id="t2",
                name="child",
                agent_role="worker",
                dependencies=["missing"],
            )
        )
    except ValueError as exc:
        assert "Dependency missing not found" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_dag_supports_online_task_addition():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="worker"))
    dag.complete_task("parent", result="done")

    dag.add_task(
        Task(
            task_id="child",
            name="child",
            agent_role="worker",
            dependencies=["parent"],
        )
    )

    ready = dag.get_ready_tasks()
    assert [task.task_id for task in ready] == ["child"]


def test_dag_add_dynamic_tasks_adds_valid_batch_atomically():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="worker"))
    dag.complete_task("parent", result="done")

    dag.add_dynamic_tasks([
        Task(
            task_id="child-a",
            name="child-a",
            agent_role="worker",
            dependencies=["parent"],
        ),
        Task(
            task_id="child-b",
            name="child-b",
            agent_role="worker",
            dependencies=["parent"],
        ),
    ])

    assert set(dag.tasks) == {"parent", "child-a", "child-b"}
    assert [task.task_id for task in dag.get_ready_tasks()] == [
        "child-a",
        "child-b",
    ]


def test_dag_add_dynamic_tasks_allows_ordered_intra_batch_dependencies():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="worker"))
    dag.complete_task("parent", result="done")

    dag.add_dynamic_tasks([
        Task(
            task_id="child-a",
            name="child-a",
            agent_role="worker",
            dependencies=["parent"],
        ),
        Task(
            task_id="child-b",
            name="child-b",
            agent_role="worker",
            dependencies=["child-a"],
        ),
    ])

    assert set(dag.tasks) == {"parent", "child-a", "child-b"}
    assert [task.task_id for task in dag.get_ready_tasks()] == ["child-a"]


def test_dag_add_dynamic_tasks_rejects_duplicate_ids_atomically():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="worker"))

    with pytest.raises(ValueError, match="Duplicate task id in batch"):
        dag.add_dynamic_tasks([
            Task(
                task_id="child",
                name="child",
                agent_role="worker",
                dependencies=["parent"],
            ),
            Task(
                task_id="child",
                name="duplicate",
                agent_role="worker",
                dependencies=["parent"],
            ),
        ])

    assert set(dag.tasks) == {"parent"}


def test_dag_add_dynamic_tasks_rejects_existing_id_atomically():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="worker"))

    with pytest.raises(ValueError, match="already exists"):
        dag.add_dynamic_tasks([
            Task(
                task_id="child",
                name="child",
                agent_role="worker",
                dependencies=["parent"],
            ),
            Task(
                task_id="parent",
                name="duplicate-parent",
                agent_role="worker",
            ),
        ])

    assert set(dag.tasks) == {"parent"}


def test_dag_add_dynamic_tasks_rejects_missing_dependency_atomically():
    dag = TaskDAG()
    dag.add_task(Task(task_id="parent", name="parent", agent_role="worker"))

    with pytest.raises(ValueError, match="Dependency missing not found"):
        dag.add_dynamic_tasks([
            Task(
                task_id="child-a",
                name="child-a",
                agent_role="worker",
                dependencies=["parent"],
            ),
            Task(
                task_id="child-b",
                name="child-b",
                agent_role="worker",
                dependencies=["missing"],
            ),
        ])

    assert set(dag.tasks) == {"parent"}


def test_dag_computes_critical_path_scores():
    dag = TaskDAG()
    dag.add_task(Task(task_id="root", name="root", agent_role="worker"))
    dag.add_task(
        Task(
            task_id="short",
            name="short",
            agent_role="worker",
            dependencies=["root"],
        )
    )
    dag.add_task(
        Task(
            task_id="long",
            name="long",
            agent_role="worker",
            dependencies=["root"],
        )
    )
    dag.add_task(
        Task(
            task_id="leaf",
            name="leaf",
            agent_role="worker",
            dependencies=["long"],
        )
    )

    dag.compute_critical_path_scores()

    assert dag.tasks["root"].critical_path_score == 3.0
    assert dag.tasks["long"].critical_path_score == 2.0
    assert dag.tasks["short"].critical_path_score == 1.0


def test_task_carries_scheduling_and_resource_metadata():
    quota = ResourceQuota(cpu_max_percent=50.0, memory_max_mb=512)
    task = Task(
        task_id="t1",
        name="code",
        agent_role="coder",
        priority=10,
        resource_requirements=quota,
        context_refs=["ctx://repo-summary"],
        critical_path_score=3.5,
        metadata={"kind": "coding"},
    )
    assert task.task_id == "t1"
    assert task.name == "code"
    assert task.agent_role == "coder"
    assert task.dependencies == []
    assert task.status == TaskStatus.PENDING
    assert task.assigned_agent_id is None
    assert task.priority == 10
    assert task.resource_requirements.memory_max_mb == 512
    assert task.context_refs == ["ctx://repo-summary"]
    assert task.critical_path_score == 3.5
    assert task.result is None
    assert task.error is None
    assert task.metadata["kind"] == "coding"
    assert task.created_at <= task.updated_at


def test_task_status_includes_skipped():
    """SKIPPED status should be a valid TaskStatus."""
    assert TaskStatus.SKIPPED.value == "skipped"


def test_task_status_includes_cancelled():
    assert TaskStatus.CANCELLED.value == "cancelled"


def test_skipped_task_not_completed():
    """SKIPPED should not equal COMPLETED."""
    assert TaskStatus.SKIPPED != TaskStatus.COMPLETED


def test_dag_cancel_pending_task():
    dag = TaskDAG()
    dag.add_task(Task(task_id="root", name="root", agent_role="worker"))
    dag.add_task(Task(task_id="child", name="child", agent_role="worker", dependencies=["root"]))

    cancelled = dag.cancel_task("child", reason="branch obsolete")

    assert cancelled is dag.tasks["child"]
    assert dag.tasks["child"].status == TaskStatus.CANCELLED
    assert dag.tasks["child"].error == "branch obsolete"


def test_dag_rejects_cancelling_running_or_completed_task():
    dag = TaskDAG()
    dag.add_task(Task(task_id="running", name="running", agent_role="worker"))
    dag.add_task(Task(task_id="done", name="done", agent_role="worker"))
    dag.tasks["running"].status = TaskStatus.RUNNING
    dag.complete_task("done", result="ok")

    with pytest.raises(ValueError, match="cannot be cancelled"):
        dag.cancel_task("running")
    with pytest.raises(ValueError, match="cannot be cancelled"):
        dag.cancel_task("done")


def test_dag_cancel_downstream_only_pending_tasks():
    dag = TaskDAG()
    dag.add_task(Task(task_id="root", name="root", agent_role="worker"))
    dag.add_task(Task(task_id="done", name="done", agent_role="worker", dependencies=["root"]))
    dag.add_task(Task(task_id="pending", name="pending", agent_role="worker", dependencies=["root"]))
    dag.add_task(Task(task_id="leaf", name="leaf", agent_role="worker", dependencies=["pending"]))
    dag.complete_task("done", result="ok")

    cancelled = dag.cancel_downstream("root", reason="replace branch")

    assert [task.task_id for task in cancelled] == ["pending", "leaf"]
    assert dag.tasks["done"].status == TaskStatus.COMPLETED
    assert dag.tasks["pending"].status == TaskStatus.CANCELLED
    assert dag.tasks["leaf"].status == TaskStatus.CANCELLED


def test_dag_replace_downstream_cancels_old_pending_branch_and_adds_replacements():
    dag = TaskDAG()
    dag.add_task(Task(task_id="root", name="root", agent_role="worker"))
    dag.add_task(Task(task_id="old-a", name="old-a", agent_role="worker", dependencies=["root"]))
    dag.add_task(Task(task_id="old-b", name="old-b", agent_role="worker", dependencies=["old-a"]))
    dag.complete_task("root", result="ok")

    cancelled, replacements = dag.replace_downstream(
        "root",
        [
            Task(task_id="new-a", name="new-a", agent_role="worker", dependencies=["root"]),
            Task(task_id="new-b", name="new-b", agent_role="worker", dependencies=["new-a"]),
        ],
        reason="new plan",
    )

    assert [task.task_id for task in cancelled] == ["old-a", "old-b"]
    assert [task.task_id for task in replacements] == ["new-a", "new-b"]
    assert dag.tasks["old-a"].status == TaskStatus.CANCELLED
    assert dag.tasks["old-b"].status == TaskStatus.CANCELLED
    assert dag.tasks["new-a"].status == TaskStatus.PENDING
    assert [task.task_id for task in dag.get_ready_tasks()] == ["new-a"]


def test_dag_replace_downstream_validates_replacements_before_cancelling():
    dag = TaskDAG()
    dag.add_task(Task(task_id="root", name="root", agent_role="worker"))
    dag.add_task(Task(task_id="old", name="old", agent_role="worker", dependencies=["root"]))

    with pytest.raises(ValueError, match="Dependency missing not found"):
        dag.replace_downstream(
            "root",
            [Task(task_id="new", name="new", agent_role="worker", dependencies=["missing"])],
        )

    assert dag.tasks["old"].status == TaskStatus.PENDING
    assert "new" not in dag.tasks


def test_get_downstream_tasks_direct():
    """Should return direct downstream tasks with depth=1."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))
    dag.add_task(Task(task_id="c", name="c", agent_role="r", dependencies=["b"]))

    downstream = dag.get_downstream_tasks("a")

    assert ("b", 1) in downstream
    assert ("c", 2) in downstream
    assert len(downstream) == 2


def test_get_downstream_tasks_with_depth_limit():
    """Should respect max_depth parameter."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))
    dag.add_task(Task(task_id="c", name="c", agent_role="r", dependencies=["b"]))

    downstream = dag.get_downstream_tasks("a", max_depth=1)

    assert ("b", 1) in downstream
    assert len(downstream) == 1


def test_get_downstream_tasks_no_downstream():
    """Should return empty list for leaf tasks."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))

    downstream = dag.get_downstream_tasks("a")

    assert downstream == []


def test_get_downstream_tasks_diamond():
    """Should handle diamond dependency correctly."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))
    dag.add_task(Task(task_id="c", name="c", agent_role="r", dependencies=["a"]))
    dag.add_task(Task(task_id="d", name="d", agent_role="r", dependencies=["b", "c"]))

    downstream = dag.get_downstream_tasks("a")

    assert ("b", 1) in downstream
    assert ("c", 1) in downstream
    assert ("d", 2) in downstream
    assert len(downstream) == 3


def test_is_complete_includes_skipped():
    """DAG should be complete when all tasks are COMPLETED, FAILED, or SKIPPED."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))

    dag.tasks["a"].status = TaskStatus.COMPLETED
    dag.tasks["b"].status = TaskStatus.SKIPPED

    assert dag.is_complete() is True


def test_is_complete_includes_cancelled():
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.tasks["a"].status = TaskStatus.CANCELLED

    assert dag.is_complete() is True


def test_is_not_complete_when_pending():
    """DAG should not be complete when tasks are PENDING."""
    dag = TaskDAG()
    dag.add_task(Task(task_id="a", name="a", agent_role="r"))
    dag.add_task(Task(task_id="b", name="b", agent_role="r", dependencies=["a"]))

    dag.tasks["a"].status = TaskStatus.COMPLETED
    # b is still PENDING

    assert dag.is_complete() is False
