import pytest

from agentos.context.cache_tracker import CacheHeatTracker
from agentos.context.manager import ContextManager
from agentos.core.resources import ResourceQuota
from agentos.core.task import Task, TaskDAG, TaskStatus
from agentos.metrics.model import ModelCallMetrics
from agentos.scheduler.affinity import CacheAffinityAdvisor
from agentos.scheduler.dag import DAGScheduler
from agentos.scheduler.fifo import FIFOScheduler


class TestFIFOScheduler:
    async def test_submit_and_next(self):
        scheduler = FIFOScheduler()
        task = Task(task_id="t1", name="work", agent_role="worker")
        await scheduler.submit(task)
        result = await scheduler.next()
        assert result is not None
        assert result.task_id == "t1"

    async def test_fifo_order(self):
        scheduler = FIFOScheduler()
        await scheduler.submit(Task(task_id="t1", name="first", agent_role="w"))
        await scheduler.submit(Task(task_id="t2", name="second", agent_role="w"))
        result1 = await scheduler.next()
        result2 = await scheduler.next()
        assert result1 is not None
        assert result2 is not None
        assert result1.task_id == "t1"
        assert result2.task_id == "t2"

    async def test_empty_returns_none(self):
        scheduler = FIFOScheduler()
        assert await scheduler.next() is None


class TestDAGScheduler:
    async def test_warning_pressure_halves_only_batch_kernel_weight(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="batch",
                name="batch",
                agent_role="worker",
                sla_class="batch",
                resource_requirements=ResourceQuota(cpu_shares=200),
            )
        )
        dag.add_task(
            Task(
                task_id="normal",
                name="normal",
                agent_role="worker",
                sla_class="normal",
                resource_requirements=ResourceQuota(cpu_shares=200),
            )
        )
        scheduler = DAGScheduler(dag, pressure_level="warning")

        await scheduler.get_next_batch()
        tasks = scheduler.last_decision["tasks"]

        assert tasks["batch"]["queue_class"] == "batch"
        assert tasks["batch"]["kernel_weight"] == 100
        assert tasks["batch"]["kernel_policy_base"]["kernel_weight"] == 200
        assert tasks["batch"]["kernel_policy_adjustment"] == (
            "batch_weight_halved_for_warning_pressure"
        )
        assert tasks["normal"]["queue_class"] == "normal"
        assert tasks["normal"]["kernel_weight"] == 200

    async def test_respects_dependencies(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="plan", agent_role="planner"))
        dag.add_task(
            Task(task_id="t2", name="code", agent_role="coder", dependencies=["t1"])
        )
        scheduler = DAGScheduler(dag)

        ready = await scheduler.get_next_batch()
        assert [task.task_id for task in ready] == ["t1"]
        assert dag.tasks["t1"].status == TaskStatus.RUNNING

    async def test_unlocks_after_completion(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="plan", agent_role="planner"))
        dag.add_task(
            Task(task_id="t2", name="code", agent_role="coder", dependencies=["t1"])
        )
        scheduler = DAGScheduler(dag)

        await scheduler.get_next_batch()
        dag.complete_task("t1", result="done")

        batch = await scheduler.get_next_batch()
        assert [task.task_id for task in batch] == ["t2"]
        assert dag.tasks["t2"].status == TaskStatus.RUNNING

    async def test_parallel_tasks(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="plan", agent_role="planner"))
        dag.add_task(
            Task(task_id="t2", name="code-a", agent_role="coder", dependencies=["t1"])
        )
        dag.add_task(
            Task(task_id="t3", name="code-b", agent_role="coder", dependencies=["t1"])
        )
        scheduler = DAGScheduler(dag)

        dag.complete_task("t1", result="done")
        batch = await scheduler.get_next_batch()
        assert len(batch) == 2
        assert {task.task_id for task in batch} == {"t2", "t3"}
        assert {task.status for task in batch} == {TaskStatus.RUNNING}

    async def test_dispatched_tasks_are_not_returned_twice(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="only", agent_role="worker"))
        scheduler = DAGScheduler(dag)

        first = await scheduler.get_next_batch()
        second = await scheduler.get_next_batch()

        assert [task.task_id for task in first] == ["t1"]
        assert second == []
        assert dag.tasks["t1"].status == TaskStatus.RUNNING

    async def test_dispatch_updates_task_updated_at(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="only", agent_role="worker"))
        scheduler = DAGScheduler(dag)
        before_dispatch = dag.tasks["t1"].updated_at

        batch = await scheduler.get_next_batch()

        assert [task.task_id for task in batch] == ["t1"]
        assert dag.tasks["t1"].updated_at >= before_dispatch

    async def test_empty_when_all_done(self):
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="only", agent_role="w"))
        scheduler = DAGScheduler(dag)
        dag.complete_task("t1", result="done")
        batch = await scheduler.get_next_batch()
        assert batch == []

    async def test_max_batch_size_limits_ready_tasks(self):
        dag = TaskDAG()
        for task_id in ("t1", "t2", "t3"):
            dag.add_task(Task(task_id=task_id, name=task_id, agent_role="worker"))
        scheduler = DAGScheduler(dag, max_batch_size=1)

        first = await scheduler.get_next_batch()
        second = await scheduler.get_next_batch()
        third = await scheduler.get_next_batch()
        fourth = await scheduler.get_next_batch()

        assert len(first) == 1
        assert len(second) == 1
        assert len(third) == 1
        assert fourth == []
        assert {task.task_id for task in first + second + third} == {"t1", "t2", "t3"}
        assert {task.status for task in first + second + third} == {TaskStatus.RUNNING}

    def test_rejects_non_positive_max_batch_size(self):
        dag = TaskDAG()

        with pytest.raises(ValueError, match="max_batch_size"):
            DAGScheduler(dag, max_batch_size=0)

    async def test_orders_ready_tasks_by_critical_path(self):
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
        scheduler = DAGScheduler(dag)

        dag.complete_task("root", result="done")
        batch = await scheduler.get_next_batch()

        assert [task.task_id for task in batch] == ["long", "short"]
        assert dag.tasks["long"].critical_path_score > dag.tasks["short"].critical_path_score

    async def test_priority_breaks_ties_after_critical_path(self):
        dag = TaskDAG()
        dag.add_task(
            Task(task_id="low", name="low", agent_role="worker", priority=1)
        )
        dag.add_task(
            Task(task_id="high", name="high", agent_role="worker", priority=10)
        )
        dag.add_task(
            Task(task_id="mid", name="mid", agent_role="worker", priority=5)
        )
        scheduler = DAGScheduler(dag)

        batch = await scheduler.get_next_batch()

        assert [task.task_id for task in batch] == ["high", "mid", "low"]

    async def test_created_at_and_task_id_make_order_stable(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="newer",
                name="newer",
                agent_role="worker",
                created_at=20.0,
            )
        )
        dag.add_task(
            Task(
                task_id="older-b",
                name="older-b",
                agent_role="worker",
                created_at=10.0,
            )
        )
        dag.add_task(
            Task(
                task_id="older-a",
                name="older-a",
                agent_role="worker",
                created_at=10.0,
            )
        )
        scheduler = DAGScheduler(dag)

        batch = await scheduler.get_next_batch()

        assert [task.task_id for task in batch] == ["older-a", "older-b", "newer"]

    async def test_critical_path_order_differs_from_fifo_order(self):
        fifo = FIFOScheduler()
        await fifo.submit(Task(task_id="short", name="short", agent_role="worker"))
        await fifo.submit(Task(task_id="long", name="long", agent_role="worker"))

        fifo_first = await fifo.next()
        fifo_second = await fifo.next()

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
        dag.complete_task("root", result="done")
        critical_path_batch = await DAGScheduler(dag).get_next_batch()

        assert fifo_first is not None
        assert fifo_second is not None
        assert [fifo_first.task_id, fifo_second.task_id] == ["short", "long"]
        assert [task.task_id for task in critical_path_batch] == ["long", "short"]

    async def test_filters_tasks_when_resources_are_insufficient(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="large",
                name="large",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=512),
            )
        )
        dag.add_task(
            Task(
                task_id="small",
                name="small",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=128),
            )
        )
        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(memory_max_mb=256),
        )

        batch = await scheduler.get_next_batch()
        assert [task.task_id for task in batch] == ["small"]
        assert dag.tasks["large"].status.value == "pending"

        scheduler.set_available_resources(ResourceQuota(memory_max_mb=1024))
        batch = await scheduler.get_next_batch()
        assert [task.task_id for task in batch] == ["large"]

    async def test_resource_filtering_checks_cpu_pids_and_io_limits(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="too-much-cpu",
                name="cpu",
                agent_role="worker",
                resource_requirements=ResourceQuota(cpu_max_percent=90.0),
            )
        )
        dag.add_task(
            Task(
                task_id="too-many-pids",
                name="pids",
                agent_role="worker",
                resource_requirements=ResourceQuota(pids_max=20),
            )
        )
        dag.add_task(
            Task(
                task_id="too-much-io",
                name="io",
                agent_role="worker",
                resource_requirements=ResourceQuota(io_read_bps=2048),
            )
        )
        dag.add_task(
            Task(
                task_id="fits",
                name="fits",
                agent_role="worker",
                resource_requirements=ResourceQuota(
                    cpu_max_percent=50.0,
                    pids_max=4,
                    io_read_bps=512,
                ),
            )
        )
        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(
                cpu_max_percent=75.0,
                pids_max=8,
                io_read_bps=1024,
            ),
        )

        batch = await scheduler.get_next_batch()

        assert [task.task_id for task in batch] == ["fits"]
        assert dag.tasks["too-much-cpu"].status == TaskStatus.PENDING
        assert dag.tasks["too-many-pids"].status == TaskStatus.PENDING
        assert dag.tasks["too-much-io"].status == TaskStatus.PENDING

    async def test_resource_filtering_checks_network_filesystem_and_tools(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="network",
                name="network",
                agent_role="worker",
                resource_requirements=ResourceQuota(allow_network=True),
            )
        )
        dag.add_task(
            Task(
                task_id="filesystem",
                name="filesystem",
                agent_role="worker",
                resource_requirements=ResourceQuota(allow_filesystem_write=True),
            )
        )
        dag.add_task(
            Task(
                task_id="tool-denied",
                name="tool-denied",
                agent_role="worker",
                resource_requirements=ResourceQuota(allowed_tools=["pytest", "bash"]),
            )
        )
        dag.add_task(
            Task(
                task_id="tool-allowed",
                name="tool-allowed",
                agent_role="worker",
                resource_requirements=ResourceQuota(allowed_tools=["pytest"]),
            )
        )
        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(
                allowed_tools=["pytest"],
                allow_network=False,
                allow_filesystem_write=False,
            ),
        )

        batch = await scheduler.get_next_batch()

        assert [task.task_id for task in batch] == ["tool-allowed"]
        assert dag.tasks["network"].status == TaskStatus.PENDING
        assert dag.tasks["filesystem"].status == TaskStatus.PENDING
        assert dag.tasks["tool-denied"].status == TaskStatus.PENDING

    async def test_batch_reserves_memory_across_selected_tasks(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="critical-a",
                name="critical-a",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=200),
            )
        )
        dag.add_task(
            Task(
                task_id="critical-b",
                name="critical-b",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=200),
            )
        )
        dag.add_task(
            Task(
                task_id="small",
                name="small",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=50),
            )
        )
        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(memory_max_mb=256),
        )

        first = await scheduler.get_next_batch()
        second = await scheduler.get_next_batch()

        assert [task.task_id for task in first] == ["critical-a", "small"]
        assert [task.task_id for task in second] == ["critical-b"]

    async def test_batch_reserves_cpu_and_pids_across_selected_tasks(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="cpu-a",
                name="cpu-a",
                agent_role="worker",
                resource_requirements=ResourceQuota(cpu_max_percent=60.0, pids_max=4),
            )
        )
        dag.add_task(
            Task(
                task_id="cpu-b",
                name="cpu-b",
                agent_role="worker",
                resource_requirements=ResourceQuota(cpu_max_percent=60.0, pids_max=4),
            )
        )
        dag.add_task(
            Task(
                task_id="light",
                name="light",
                agent_role="worker",
                resource_requirements=ResourceQuota(cpu_max_percent=20.0, pids_max=2),
            )
        )
        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(cpu_max_percent=100.0, pids_max=8),
        )

        first = await scheduler.get_next_batch()
        second = await scheduler.get_next_batch()

        assert [task.task_id for task in first] == ["cpu-a", "light"]
        assert [task.task_id for task in second] == ["cpu-b"]

    async def test_resource_estimator_reduces_batch_reservation(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="lint-a",
                name="lint-a",
                agent_role="lint",
                resource_requirements=ResourceQuota(memory_max_mb=300),
            )
        )
        dag.add_task(
            Task(
                task_id="lint-b",
                name="lint-b",
                agent_role="lint",
                resource_requirements=ResourceQuota(memory_max_mb=300),
            )
        )

        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(memory_max_mb=512),
            resource_estimator=lambda task: ResourceQuota(memory_max_mb=80),
        )

        batch = await scheduler.get_next_batch()

        assert [task.task_id for task in batch] == ["lint-a", "lint-b"]

    async def test_resource_estimator_does_not_bypass_static_hard_limit(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="too-large",
                name="too-large",
                agent_role="lint",
                resource_requirements=ResourceQuota(memory_max_mb=700),
            )
        )
        dag.add_task(
            Task(
                task_id="fits",
                name="fits",
                agent_role="lint",
                resource_requirements=ResourceQuota(memory_max_mb=256),
            )
        )

        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(memory_max_mb=512),
            resource_estimator=lambda task: ResourceQuota(memory_max_mb=80),
        )

        batch = await scheduler.get_next_batch()

        assert [task.task_id for task in batch] == ["fits"]
        assert dag.tasks["too-large"].status == TaskStatus.PENDING

    async def test_records_scheduler_decision_with_selected_deferred_and_blocked_tasks(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="too-large",
                name="too-large",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=700),
            )
        )
        dag.add_task(
            Task(
                task_id="first",
                name="first",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=300),
            )
        )
        dag.add_task(
            Task(
                task_id="second",
                name="second",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=300),
            )
        )

        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(memory_max_mb=512),
        )

        batch = await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert [task.task_id for task in batch] == ["first"]
        assert decision is not None
        assert decision["strategy"] == "critical_path"
        assert decision["ready_task_ids"] == ["too-large", "first", "second"]
        assert decision["selected_task_ids"] == ["first"]
        assert decision["blocked_task_ids"] == ["too-large"]
        assert decision["deferred_task_ids"] == ["second"]
        assert decision["tasks"]["too-large"]["reason"] == "exceeds_available_resources"
        assert decision["tasks"]["first"]["reason"] == "selected"
        assert decision["tasks"]["second"]["reason"] == "resource_batch_full"
        assert decision["tasks"]["first"]["resource_requirements"]["memory_max_mb"] == 300
        assert decision["available_resources"]["memory_max_mb"] == 512

    async def test_records_scheduler_decision_v2_policy_fields(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="chat",
                name="serve user request",
                agent_role="assistant",
                workload_hint="interactive",
                priority=9,
            )
        )
        scheduler = DAGScheduler(dag)

        batch = await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert [task.task_id for task in batch] == ["chat"]
        assert decision is not None
        assert decision["schema_version"] == "scheduler.decision.v2"
        assert decision["strategy"] == "critical_path"
        assert decision["policy"] == "latency_first"
        assert decision["workload_class"] == "interactive"
        assert decision["policy_reason"] == "interactive workload or high priority"
        assert decision["selected"] == ["chat"]
        assert decision["blocked"] == []
        assert decision["deferred"] == []
        assert decision["score_breakdown"]["candidate_count"] == 1
        assert decision["tasks"]["chat"]["policy"] == "latency_first"
        assert decision["tasks"]["chat"]["workload_class"] == "interactive"
        assert decision["tasks"]["chat"]["policy_reason"] == (
            "interactive workload or high priority"
        )

    async def test_throughput_policy_orders_high_parallelism_batch_first(self):
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="normal",
                name="normal",
                agent_role="worker",
            )
        )
        dag.add_task(
            Task(
                task_id="bulk",
                name="offline reduce",
                agent_role="worker",
                workload_hint="batch",
                expected_parallelism=8,
            )
        )
        scheduler = DAGScheduler(dag)

        batch = await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert [task.task_id for task in batch] == ["bulk", "normal"]
        assert decision is not None
        assert decision["policy"] == "throughput_first"
        assert decision["workload_class"] == "batch"

    async def test_hot_prefix_policy_uses_cache_affinity_and_records_policy(self):
        ctx = ContextManager()
        ctx.add_shared_ref("hot-ctx", "hot context content")
        ctx.add_shared_ref("cold-ctx", "cold context content")
        tracker = CacheHeatTracker(ttl=300.0)
        hot_pid = ctx.resolve_prefix_id(["hot-ctx"])
        tracker.record_call(
            hot_pid,
            ModelCallMetrics(
                backend="mock",
                model="mock",
                prompt_tokens=100,
                cache_hit_tokens=90,
                cache_miss_tokens=10,
            ),
        )
        advisor = CacheAffinityAdvisor(ctx, tracker)

        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="cold",
                name="generate cold answer",
                agent_role="assistant",
                workload_hint="ai_inference",
                context_refs=["cold-ctx"],
            )
        )
        dag.add_task(
            Task(
                task_id="hot",
                name="generate hot answer",
                agent_role="assistant",
                workload_hint="ai_inference",
                context_refs=["hot-ctx"],
            )
        )
        scheduler = DAGScheduler(dag, affinity_advisor=advisor)

        batch = await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert [task.task_id for task in batch] == ["hot", "cold"]
        assert decision is not None
        assert decision["policy"] == "cache_affinity"
        assert decision["workload_class"] == "ai_inference"
        assert decision["score_breakdown"]["max_prefix_heat"] > 0.7

    async def test_decision_includes_fairness_summary(self):
        """Test that decision includes fairness summary."""
        from agentos.scheduler.fairness import FairnessTracker

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

        tracker = FairnessTracker(now=lambda: 10.0)
        scheduler = DAGScheduler(dag, fairness_tracker=tracker)

        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision is not None
        assert "fairness" in decision
        assert "fairness_score" in decision["fairness"]
        assert "starvation_risk" in decision["fairness"]
        assert decision["fairness"]["starvation_risk"] in {"low", "medium", "high"}
        assert 0.0 <= decision["fairness"]["fairness_score"] <= 1.0

    async def test_task_level_decision_includes_fairness_fields(self):
        """Test that task-level decision includes wait/deferred/blocked fields."""
        from agentos.scheduler.fairness import FairnessTracker

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

        tracker = FairnessTracker(now=lambda: 10.0)
        scheduler = DAGScheduler(dag, max_batch_size=1, fairness_tracker=tracker)

        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision is not None
        # t1 should be selected, t2 should be deferred
        assert decision["tasks"]["t1"]["selected_count"] == 1
        assert decision["tasks"]["t2"]["deferred_count"] == 1
        assert "wait_ms" in decision["tasks"]["t1"]
        assert "slowdown" in decision["tasks"]["t1"]
        assert "starvation_risk" in decision["tasks"]["t1"]

    async def test_high_starvation_risk_triggers_fairness_balanced_policy(self):
        """Test that high starvation risk triggers fairness_balanced policy."""
        from agentos.scheduler.fairness import FairnessTracker

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

        # Use low thresholds to trigger high risk quickly
        tracker = FairnessTracker(
            medium_wait_ms=10,
            high_wait_ms=20,
            medium_deferred_count=1,
            high_deferred_count=2,
            now=lambda: 10.0,
        )
        scheduler = DAGScheduler(dag, max_batch_size=1, fairness_tracker=tracker)

        # First batch: t1 selected, t2 deferred
        await scheduler.get_next_batch()
        dag.tasks["t1"].status = TaskStatus.COMPLETED

        # Advance time and trigger more deferrals
        tracker._now = lambda: 10.025  # 25ms > high threshold
        await scheduler.get_next_batch()

        decision = scheduler.last_decision
        assert decision is not None
        assert decision["fairness"]["starvation_risk"] == "high"
        assert decision["policy"] == "fairness_balanced"

    async def test_fairness_balanced_policy_sorts_high_wait_first(self):
        """Test that fairness_balanced policy sorts high wait/deferred tasks first."""
        from agentos.scheduler.fairness import FairnessTracker

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))
        dag.add_task(Task(task_id="t3", name="work3", agent_role="worker"))

        # Use low thresholds to trigger high risk quickly
        tracker = FairnessTracker(
            medium_wait_ms=10,
            high_wait_ms=20,
            medium_deferred_count=1,
            high_deferred_count=2,
            now=lambda: 10.0,
        )
        scheduler = DAGScheduler(dag, max_batch_size=1, fairness_tracker=tracker)

        # First batch: t1 selected, t2 and t3 deferred
        await scheduler.get_next_batch()
        dag.tasks["t1"].status = TaskStatus.COMPLETED

        # Advance time significantly
        tracker._now = lambda: 10.030  # 30ms > high threshold
        await scheduler.get_next_batch()

        decision = scheduler.last_decision
        assert decision is not None
        assert decision["policy"] == "fairness_balanced"

        # t2 and t3 should have high starvation risk due to wait time
        assert decision["tasks"]["t2"]["starvation_risk"] == "high"
        assert decision["tasks"]["t3"]["starvation_risk"] == "high"

        # The deferred task (t2 or t3) should be selected first in next batch
        # Both have same wait time, so order depends on other factors
        dag.tasks[decision["selected"][0]].status = TaskStatus.COMPLETED
        await scheduler.get_next_batch()
        decision2 = scheduler.last_decision
        assert decision2 is not None
        # All remaining tasks should be selected
        assert len(decision2["selected"]) > 0

    async def test_decision_fairness_updates_immediately_after_defer(self):
        """Test that decision["fairness"] and task-level starvation_risk update immediately after defer."""
        from agentos.scheduler.fairness import FairnessTracker

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

        # Use low thresholds to trigger medium risk quickly
        tracker = FairnessTracker(
            medium_wait_ms=5,
            high_wait_ms=20,
            medium_deferred_count=1,
            high_deferred_count=2,
            now=lambda: 10.0,
        )
        scheduler = DAGScheduler(dag, max_batch_size=1, fairness_tracker=tracker)

        # First batch: t1 selected, t2 deferred
        await scheduler.get_next_batch()
        decision1 = scheduler.last_decision

        # t2 should have been deferred once
        assert decision1["tasks"]["t2"]["deferred_count"] == 1
        # Fairness summary should reflect the defer
        assert decision1["fairness"]["starvation_risk"] in {"low", "medium"}

        # Advance time to trigger medium risk from wait
        tracker._now = lambda: 10.006  # 6ms > medium threshold (5ms)
        dag.tasks["t1"].status = TaskStatus.COMPLETED
        await scheduler.get_next_batch()
        decision2 = scheduler.last_decision

        # t2 should now have medium starvation risk from wait time
        assert decision2["tasks"]["t2"]["starvation_risk"] == "medium"
        # Fairness summary should reflect medium risk
        assert decision2["fairness"]["starvation_risk"] == "medium"

    async def test_scheduler_decision_includes_kernel_probe(self):
        """Test that scheduler.decision includes kernel_probe when provided."""
        from agentos.isolation.sched_ext import KernelProbeResult

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        probe = KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="test",
        )
        scheduler = DAGScheduler(dag, kernel_probe=probe)

        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision is not None
        assert "kernel_probe" in decision
        assert decision["kernel_probe"]["sched_ext"] == "unavailable"
        assert decision["kernel_probe"]["bpf_map"] == "mock"
        assert decision["kernel_probe"]["fallback_mode"] == "mock_bpf_map"

    async def test_scheduler_decision_without_kernel_probe(self):
        """Test that scheduler.decision works without kernel_probe."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision is not None
        assert "kernel_probe" not in decision

    async def test_scheduler_returns_task_decisions_without_publishing(self):
        """Scheduler decisions stay independent from kernel publication."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))
        dag.add_task(Task(task_id="t2", name="work2", agent_role="worker"))

        scheduler = DAGScheduler(dag)

        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision is not None
        assert set(decision["tasks"]) == {"t1", "t2"}
        assert decision["tasks"]["t1"]["policy"] == decision["policy"]
        assert "starvation_risk" in decision["tasks"]["t1"]
        assert "bpf_map" not in decision

    async def test_scheduler_decision_without_bpf_map_adapter(self):
        """Test that scheduler.decision works without bpf_map_adapter."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision is not None
        assert "bpf_map" not in decision

    async def test_scheduler_decision_includes_os_integration_fields(self):
        """Test that scheduler.decision includes OS integration fields."""
        from agentos.isolation.sched_ext import KernelProbeResult

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        probe = KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="forced mock mode",
        )
        scheduler = DAGScheduler(dag, kernel_probe=probe)

        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision is not None
        assert decision["runtime_mode"] == "user_space_runtime"
        assert decision["kernel_integration"] in {"mock", "probe", "unavailable"}
        assert "fallback_reason" in decision

    async def test_scheduler_decision_kernel_integration_mock(self):
        """Test kernel_integration is 'mock' when bpf_map is mock."""
        from agentos.isolation.sched_ext import KernelProbeResult

        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        probe = KernelProbeResult(
            sched_ext="unavailable",
            bpf_map="mock",
            fallback_mode="mock_bpf_map",
            reason="forced mock mode",
        )
        scheduler = DAGScheduler(dag, kernel_probe=probe)

        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["kernel_integration"] == "mock"
        assert decision["fallback_reason"] == "forced mock mode"

    async def test_scheduler_decision_kernel_integration_without_probe(self):
        """Test kernel_integration is 'unavailable' when no probe."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["kernel_integration"] == "unavailable"
        assert decision["fallback_reason"] is None

    async def test_blocked_task_has_resource_explanation(self):
        """Test that blocked task has resource explanation fields."""
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="large",
                name="large",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=700),
            )
        )
        dag.add_task(
            Task(
                task_id="small",
                name="small",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=128),
            )
        )
        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(memory_max_mb=256),
        )

        batch = await scheduler.get_next_batch()
        decision = scheduler.last_decision

        # large should be blocked due to memory
        assert decision["tasks"]["large"]["reason"] == "exceeds_available_resources"
        assert decision["tasks"]["large"]["blocked_resource"] == "memory"
        assert decision["tasks"]["large"]["required_value"] == 700
        assert decision["tasks"]["large"]["available_value"] == 256

        # small should be selected
        assert decision["tasks"]["small"]["reason"] == "selected"
        assert decision["tasks"]["small"]["blocked_resource"] is None

    async def test_unblocked_task_has_no_resource_explanation(self):
        """Test that unblocked task has None resource explanation fields."""
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="fits",
                name="fits",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=128),
            )
        )
        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(memory_max_mb=256),
        )

        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert decision["tasks"]["fits"]["blocked_resource"] is None
        assert decision["tasks"]["fits"]["required_value"] is None
        assert decision["tasks"]["fits"]["available_value"] is None

    async def test_scheduler_decision_contains_runtime_health(self):
        """Test that scheduler.decision contains runtime_health."""
        dag = TaskDAG()
        dag.add_task(Task(task_id="t1", name="work", agent_role="worker"))

        scheduler = DAGScheduler(dag)
        await scheduler.get_next_batch()
        decision = scheduler.last_decision

        assert "runtime_health" in decision
        health = decision["runtime_health"]
        assert "runtime_health_score" in health
        assert "queue_pressure" in health
        assert "resource_pressure" in health
        assert "fairness_pressure" in health
        assert "dependency_pressure" in health
        assert "pressure_level" in health
        assert "health_warnings" in health

    async def test_runtime_health_reflects_blocked_resources(self):
        """Test that runtime_health reflects blocked resources."""
        dag = TaskDAG()
        dag.add_task(
            Task(
                task_id="large",
                name="large",
                agent_role="worker",
                resource_requirements=ResourceQuota(memory_max_mb=1024),
            )
        )

        scheduler = DAGScheduler(
            dag,
            available_resources=ResourceQuota(memory_max_mb=128),
        )

        await scheduler.get_next_batch()
        health = scheduler.last_decision["runtime_health"]

        assert health["resource_pressure"] > 0.0
        assert health["runtime_health_score"] < 1.0
