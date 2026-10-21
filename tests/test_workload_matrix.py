"""Tests for workload matrix, interaction overhead, and dashboard integration."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agentos.benchmarks.cache_sim import CacheSimulatingMockBackend
from agentos.comm.bus import MessageBus
from agentos.core.types import Message
from agentos.dashboard.builder import render_dashboard, load_benchmark_summary
from scripts.probe_backend import probe_backend, BackendProbeResult


# ── MessageBus stats aggregation ──


async def test_bus_stats_payload_bytes_accumulate():
    bus = MessageBus()
    for i in range(5):
        await bus.send(
            Message(
                from_agent="a1",
                to_agent="a2",
                content={"data": "x" * 100},
                context_refs=[f"ctx://ref-{i}"],
            )
        )
    stats = bus.stats()
    assert stats["message_count"] == 5
    assert stats["context_ref_count"] == 5
    assert stats["payload_bytes"] > 0


async def test_bus_payload_savings_shows_bytes_saved():
    bus = MessageBus()
    large = "A" * 1000
    await bus.send(
        Message(
            from_agent="a1",
            to_agent="a2",
            content={"full_text": large},
            message_type="inline",
        )
    )
    await bus.send(
        Message(
            from_agent="a1",
            to_agent="a2",
            content={"ref": "ctx://shared"},
            message_type="context_ref",
            context_refs=["ctx://shared"],
        )
    )
    savings = bus.payload_savings()
    assert savings["inline_payload_bytes"] > savings["ref_payload_bytes"]
    assert savings["bytes_saved"] > 0
    assert 0 <= savings["payload_reduction_ratio"] <= 1


# ── WorkloadResult interaction_overhead ──


async def test_workload_result_contains_interaction_overhead():
    from scripts.workloads._base import WorkloadResult, InteractionOverhead
    r = WorkloadResult(
        workload_type="test",
        mode="runtime",
        interaction_overhead=InteractionOverhead(
            runtime_events=42,
            message_count=10,
            payload_bytes=500,
            contextref_bytes_saved=200,
        ),
    )
    d = r.to_dict()
    assert "interaction_overhead" in d
    assert d["interaction_overhead"]["runtime_events"] == 42
    assert d["interaction_overhead"]["contextref_bytes_saved"] == 200


async def test_workload_result_contains_normalized_metrics():
    from scripts.workloads._base import WorkloadResult, NormalizedMetrics
    r = WorkloadResult(
        workload_type="test",
        mode="runtime",
        normalized=NormalizedMetrics(
            prompt_tokens_per_completed_task=150.0,
            cache_hit_ratio_improvement=0.15,
            fault_recovery_rate=1.0,
        ),
    )
    d = r.to_dict()
    assert "normalized" in d
    assert d["normalized"]["prompt_tokens_per_completed_task"] == 150.0
    assert d["normalized"]["fault_recovery_rate"] == 1.0


# ── run_workload produces overhead + normalized ──


async def test_run_workload_produces_overhead_and_normalized():
    from scripts.workloads import ai_inference_pressure
    backend = CacheSimulatingMockBackend()
    results = await ai_inference_pressure.run(backend=backend)

    rt = results["runtime"]
    bl = results["baseline"]

    assert rt.interaction_overhead.runtime_events > 0
    assert rt.interaction_overhead.contextref_bytes_saved > 0

    assert rt.normalized.prompt_tokens_per_completed_task > 0
    assert rt.normalized.cache_hit_ratio_improvement > 0
    assert rt.normalized.tasks_completed_delta > 0
    assert rt.normalized.fault_recovery_rate == 1.0

    assert bl.normalized.prompt_tokens_per_completed_task > 0
    assert bl.normalized.cache_hit_ratio_improvement == 0.0


# ── summary.json schema ──


async def test_workload_matrix_summary_schema(tmp_path: Path):
    from scripts.run_workload_matrix import run_matrix
    summary = await run_matrix("mock", tmp_path / "out", workloads=["ai_inference"])

    assert "metadata" in summary
    assert "workloads" in summary
    assert "aggregate" in summary
    assert "kernel_probe" in summary

    agg = summary["aggregate"]
    assert "interaction_overhead" in agg
    oh = agg["interaction_overhead"]
    assert "total_runtime_events" in oh
    assert "total_message_count" in oh
    assert "total_payload_bytes" in oh
    assert "total_contextref_bytes_saved" in oh

    assert "prompt_tokens_per_task_baseline" in agg
    assert "prompt_tokens_per_task_runtime" in agg

    wl = summary["workloads"]["ai_inference"]
    for mode in ("baseline", "runtime"):
        assert "interaction_overhead" in wl[mode]
        assert "normalized" in wl[mode]


async def test_workload_matrix_summary_json_roundtrips(tmp_path: Path):
    from scripts.run_workload_matrix import run_matrix
    out = tmp_path / "out"
    await run_matrix("mock", out, workloads=["compile"])

    json_path = out / "summary.json"
    assert json_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["workloads"]["compile"]["runtime"]["interaction_overhead"]["runtime_events"] > 0
    assert loaded["workloads"]["compile"]["runtime"]["normalized"]["prompt_tokens_per_completed_task"] > 0


# ── Dashboard renders interaction overhead fields ──


def test_dashboard_renders_workload_matrix_with_overhead():
    matrix = {
        "metadata": {"backend": "mock", "total_duration_secs": 1.0},
        "workloads": {
            "test_wl": {
                "baseline": {
                    "total_latency_secs": 0.1,
                    "cache_hit_ratio": 0.5,
                    "tasks_completed": 3,
                    "tasks_total": 7,
                    "fault_injected": True,
                    "fault_recovered": False,
                    "event_count": 10,
                    "normalized": {
                        "prompt_tokens_per_completed_task": 150,
                        "latency_per_completed_task_secs": 0.033,
                        "cache_hit_ratio_improvement": 0.0,
                        "fault_recovery_rate": 0.0,
                    },
                    "interaction_overhead": {
                        "runtime_events": 10,
                        "message_count": 0,
                        "payload_bytes": 0,
                        "contextref_bytes_saved": 0,
                        "event_latency_ms_avg": 1.0,
                        "worker_startup_ms_avg": 0.0,
                    },
                },
                "runtime": {
                    "total_latency_secs": 0.2,
                    "cache_hit_ratio": 0.75,
                    "tasks_completed": 7,
                    "tasks_total": 7,
                    "fault_injected": True,
                    "fault_recovered": True,
                    "event_count": 80,
                    "normalized": {
                        "prompt_tokens_per_completed_task": 120,
                        "latency_per_completed_task_secs": 0.028,
                        "cache_hit_ratio_improvement": 0.25,
                        "fault_recovery_rate": 1.0,
                    },
                    "interaction_overhead": {
                        "runtime_events": 80,
                        "message_count": 5,
                        "payload_bytes": 300,
                        "contextref_bytes_saved": 1500,
                        "event_latency_ms_avg": 2.5,
                        "worker_startup_ms_avg": 10.0,
                    },
                },
            },
        },
        "aggregate": {
            "cache_hit_ratio_improvement": 0.25,
            "faults_recovered": 1,
            "faults_total": 1,
            "total_tasks_completed_baseline": 3,
            "total_tasks_completed_runtime": 7,
            "prompt_tokens_per_task_baseline": 150,
            "prompt_tokens_per_task_runtime": 120,
            "interaction_overhead": {
                "total_runtime_events": 80,
                "total_message_count": 5,
                "total_payload_bytes": 300,
                "total_contextref_bytes_saved": 1500,
            },
        },
        "kernel_probe": {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "reason": "missing /sys/kernel/sched_ext",
        },
    }

    html = render_dashboard([], workload_matrix=matrix)

    assert "Workload Matrix Evidence" in html
    assert "Normalized Metrics" in html
    assert "Interaction Overhead" in html
    assert "sched_ext" in html
    assert "BPF Map" in html
    assert "ContextRef Saved" in html
    assert "Tok/Task" in html
    assert "per completed task" in html
    assert "1500" in html  # contextref saved bytes


def test_dashboard_graceful_without_workload_matrix():
    html = render_dashboard([], workload_matrix=None)
    assert "No workload matrix summary loaded" in html


def test_dashboard_graceful_with_empty_overhead():
    matrix = {
        "metadata": {"backend": "mock", "total_duration_secs": 0.5},
        "workloads": {
            "minimal": {
                "baseline": {
                    "total_latency_secs": 0.05,
                    "cache_hit_ratio": 0.3,
                    "tasks_completed": 2,
                    "tasks_total": 3,
                    "fault_injected": False,
                    "fault_recovered": False,
                    "event_count": 5,
                },
                "runtime": {
                    "total_latency_secs": 0.06,
                    "cache_hit_ratio": 0.6,
                    "tasks_completed": 3,
                    "tasks_total": 3,
                    "fault_injected": False,
                    "fault_recovered": False,
                    "event_count": 15,
                },
            },
        },
        "aggregate": {
            "cache_hit_ratio_improvement": 0.3,
            "faults_recovered": 0,
            "faults_total": 0,
            "total_tasks_completed_baseline": 2,
            "total_tasks_completed_runtime": 3,
        },
    }

    html = render_dashboard([], workload_matrix=matrix)
    assert "Workload Matrix Evidence" in html


# ── Phase 3: Backend probe ──


async def test_probe_backend_mock_always_available():
    result = await probe_backend("mock")
    assert result.available is True
    assert result.fallback_reason == ""
    assert result.backend == "mock"


async def test_probe_backend_cache_sim_always_available():
    result = await probe_backend("cache-sim")
    assert result.available is True


async def test_probe_backend_vllm_unavailable_graceful():
    result = await probe_backend("vllm", base_url="http://127.0.0.1:59999")
    assert result.available is False
    assert result.fallback_reason != ""
    assert result.backend == "vllm"


async def test_probe_backend_unknown_backend():
    result = await probe_backend("nonexistent")
    assert result.available is False
    assert "unknown" in result.fallback_reason


async def test_probe_backend_to_dict_schema():
    result = await probe_backend("mock")
    d = result.to_dict()
    for key in ("backend", "available", "base_url", "model", "fallback_reason",
                "models_endpoint", "metrics_endpoint", "prefix_cache_metrics"):
        assert key in d


# ── Phase 3: Backend probe in summary ──


async def test_summary_contains_backend_probe(tmp_path: Path):
    from scripts.run_workload_matrix import run_matrix
    summary = await run_matrix("mock", tmp_path / "out", workloads=["ai_inference"])
    assert "backend_probe" in summary
    bp = summary["backend_probe"]
    assert bp["backend"] in ("mock", "cache-sim")
    assert bp["available"] is True


async def test_summary_vllm_fallback_in_backend_probe(tmp_path: Path):
    from scripts.run_workload_matrix import run_matrix
    summary = await run_matrix(
        "vllm", tmp_path / "out", workloads=["ai_inference"],
        base_url="http://127.0.0.1:59999",
    )
    bp = summary["backend_probe"]
    assert bp["available"] is False
    assert bp["fallback_reason"] != ""
    assert summary["metadata"]["backend"] == "cache-sim"


# ── Phase 3: interaction_probe in summary ──


async def test_summary_contains_interaction_probe(tmp_path: Path):
    from scripts.run_workload_matrix import run_matrix
    summary = await run_matrix("mock", tmp_path / "out", workloads=["ai_inference"])
    assert "interaction_probe" in summary
    ip = summary["interaction_probe"]
    assert "message_count" in ip
    assert "context_ref_count" in ip
    assert "payload_bytes" in ip
    assert "bytes_saved" in ip
    assert "payload_reduction_ratio" in ip
    assert "contextref_bytes_saved_total" in ip
    assert ip["message_count"] > 0
    assert ip["workload_count"] == 1


# ── Phase 3: probe_sched_ext enhancements ──


def test_probe_sched_ext_has_cgroup_v2_fields():
    from scripts.probe_sched_ext import probe
    result = probe()
    d = result.to_dict()
    assert "cgroup_v2_mounted" in d
    assert "cgroup_v2_path" in d
    assert "cgroup_v2_controllers" in d
    assert "openeuler_version" in d


# ── Phase 3: Dashboard renders new sections ──


def test_dashboard_renders_backend_probe():
    matrix = {
        "metadata": {"backend": "vllm", "total_duration_secs": 2.0},
        "backend_probe": {
            "backend": "vllm",
            "available": False,
            "base_url": "http://127.0.0.1:8000",
            "model": "",
            "models_endpoint": False,
            "metrics_endpoint": False,
            "prefix_cache_metrics": False,
            "health_latency_ms": 0.0,
            "fallback_reason": "vllm at http://127.0.0.1:8000 not reachable",
        },
        "interaction_probe": {
            "message_count": 12,
            "context_ref_count": 12,
            "payload_bytes": 800,
            "inline_payload_bytes": 5000,
            "ref_payload_bytes": 800,
            "bytes_saved": 4200,
            "payload_reduction_ratio": 0.84,
            "contextref_bytes_saved_total": 9000,
            "workload_count": 4,
        },
        "workloads": {
            "test_wl": {
                "baseline": {
                    "total_latency_secs": 0.1,
                    "cache_hit_ratio": 0.5,
                    "tasks_completed": 3, "tasks_total": 7,
                    "fault_injected": True, "fault_recovered": False,
                    "event_count": 10,
                },
                "runtime": {
                    "total_latency_secs": 0.2,
                    "cache_hit_ratio": 0.75,
                    "tasks_completed": 7, "tasks_total": 7,
                    "fault_injected": True, "fault_recovered": True,
                    "event_count": 80,
                },
            },
        },
        "aggregate": {
            "cache_hit_ratio_improvement": 0.25,
            "faults_recovered": 1, "faults_total": 1,
            "total_tasks_completed_baseline": 3,
            "total_tasks_completed_runtime": 7,
        },
        "kernel_probe": {"sched_ext": "unavailable", "bpf_map": "mock", "reason": "test"},
        "kernel_probe_detail": {
            "cgroup_v2_mounted": True,
            "cgroup_v2_path": "/sys/fs/cgroup",
            "cgroup_v2_controllers": ["cpu", "memory", "io"],
            "is_openeuler": True,
            "openeuler_version": "22.03 LTS SP3",
        },
    }

    html = render_dashboard([], workload_matrix=matrix)

    assert "Real Backend Probe" in html
    assert "not reachable" in html
    assert "MessageBus / ContextRef Communication Evidence" in html
    assert "Payload Reduction Ratio" in html
    assert "84.00%" in html
    assert "9000" in html
    assert "cgroup v2" in html
    assert "mounted" in html
    assert "openEuler" in html
    assert "22.03 LTS SP3" in html


# ── Phase 4: interaction_probe bytes_saved > 0 ──


async def test_interaction_probe_bytes_saved_positive(tmp_path: Path):
    from scripts.run_workload_matrix import run_matrix
    summary = await run_matrix("mock", tmp_path / "out", workloads=["ai_inference"])
    ip = summary["interaction_probe"]
    assert ip["bytes_saved"] > 0
    assert ip["payload_reduction_ratio"] > 0
    assert ip["inline_payload_bytes"] > ip["ref_payload_bytes"]


async def test_interaction_probe_full_matrix_bytes_saved(tmp_path: Path):
    from scripts.run_workload_matrix import run_matrix
    summary = await run_matrix("mock", tmp_path / "out")
    ip = summary["interaction_probe"]
    assert ip["bytes_saved"] > 0
    assert ip["workload_count"] == 4
    assert ip["contextref_bytes_saved_total"] > 0


# ── Phase 4: non-openEuler dashboard does not show openEuler version ──


def test_dashboard_non_openeuler_no_version():
    matrix = {
        "metadata": {"backend": "mock", "total_duration_secs": 0.5},
        "workloads": {
            "minimal": {
                "baseline": {
                    "total_latency_secs": 0.05, "cache_hit_ratio": 0.3,
                    "tasks_completed": 2, "tasks_total": 3,
                    "fault_injected": False, "fault_recovered": False,
                    "event_count": 5,
                },
                "runtime": {
                    "total_latency_secs": 0.06, "cache_hit_ratio": 0.6,
                    "tasks_completed": 3, "tasks_total": 3,
                    "fault_injected": False, "fault_recovered": False,
                    "event_count": 15,
                },
            },
        },
        "aggregate": {
            "cache_hit_ratio_improvement": 0.3,
            "faults_recovered": 0, "faults_total": 0,
            "total_tasks_completed_baseline": 2,
            "total_tasks_completed_runtime": 3,
        },
        "kernel_probe": {"sched_ext": "unavailable", "bpf_map": "mock", "reason": "test"},
        "kernel_probe_detail": {
            "is_openeuler": False,
            "openeuler_version": "",
            "cgroup_v2_mounted": True,
            "cgroup_v2_path": "/sys/fs/cgroup",
        },
    }
    html = render_dashboard([], workload_matrix=matrix)
    assert "cgroup v2" in html
    assert "mounted" in html
    assert "openEuler" not in html


# ── Phase 4: dashboard renders bytes_saved from interaction_probe ──


def test_dashboard_renders_interaction_probe_savings():
    matrix = {
        "metadata": {"backend": "mock", "total_duration_secs": 1.0},
        "interaction_probe": {
            "message_count": 28,
            "context_ref_count": 14,
            "payload_bytes": 20000,
            "inline_payload_bytes": 18000,
            "ref_payload_bytes": 2000,
            "bytes_saved": 16000,
            "payload_reduction_ratio": 0.889,
            "contextref_bytes_saved_total": 12000,
            "workload_count": 4,
        },
        "workloads": {
            "test": {
                "baseline": {
                    "total_latency_secs": 0.1, "cache_hit_ratio": 0.5,
                    "tasks_completed": 3, "tasks_total": 7,
                    "fault_injected": False, "fault_recovered": False,
                    "event_count": 10,
                },
                "runtime": {
                    "total_latency_secs": 0.2, "cache_hit_ratio": 0.7,
                    "tasks_completed": 7, "tasks_total": 7,
                    "fault_injected": False, "fault_recovered": False,
                    "event_count": 50,
                },
            },
        },
        "aggregate": {
            "cache_hit_ratio_improvement": 0.2,
            "faults_recovered": 0, "faults_total": 0,
            "total_tasks_completed_baseline": 3,
            "total_tasks_completed_runtime": 7,
        },
    }
    html = render_dashboard([], workload_matrix=matrix)
    assert "MessageBus / ContextRef Communication Evidence" in html
    assert "16000" in html
    assert "88.90%" in html
    assert "12000" in html


# ── force_policy passthrough (A1.2) ──


async def test_run_workload_accepts_force_policy():
    """Verify run_workload passes force_policy to Runtime."""
    from scripts.workloads import ai_inference_pressure
    from scripts.workloads._base import run_workload
    from scripts.workloads.ai_inference_pressure import (
        _build_dag, build_handlers, SYSTEM_PREFIX, TASK_CONTEXT,
    )

    backend = CacheSimulatingMockBackend()
    baseline_h, runtime_h = build_handlers()
    results = await run_workload(
        workload_type="ai_inference",
        dag=_build_dag(),
        baseline_handlers=baseline_h,
        runtime_handlers=runtime_h,
        shared_context={"system-prefix": SYSTEM_PREFIX, "task-context": TASK_CONTEXT},
        backend=backend,
        fault_plan={"generate": "kill"},
        force_policy="throughput_first",
    )

    rt = results["runtime"]
    # Should complete without error
    assert rt.tasks_completed > 0
    assert rt.total_latency_secs > 0


# ── Phase 5: _extract_kernel_adapter_evidence() direct unit tests ──


def _make_pressure_event(level: str, scope: str = "system", nested: bool = True) -> dict:
    """Build a resource.pressure_sampled event."""
    if nested:
        data = {"scope": scope, "pressure": {"pressure_level": level}}
    else:
        data = {"scope": scope, "pressure_level": level}
    return {"event_type": "resource.pressure_sampled", "data": data}


def _make_cgroup_quota_event(path: str = "/sys/fs/cgroup/agentos/agent_0") -> dict:
    return {
        "event_type": "resource.cgroup_quota_applied",
        "data": {"path": path},
    }


def _make_bpf_update_event(mode: str = "mock") -> dict:
    return {"event_type": "scheduler.bpf_map_updated", "data": {"mode": mode}}


def _make_decision_event_with_schema(schema_version: str = "bpf.task_record.v2") -> dict:
    return {
        "event_type": "scheduler.decision",
        "data": {
            "bpf_map": {
                "tasks": {
                    "agent_0": {"schema_version": schema_version, "pid": 0},
                }
            }
        },
    }


def _make_kernel_probe_event(reason: str = "test_fallback") -> dict:
    return {
        "event_type": "scheduler.kernel_probe",
        "data": {"reason": reason},
    }


class TestExtractKernelAdapterEvidence:
    """Direct unit tests for _extract_kernel_adapter_evidence()."""

    def test_nested_pressure_level_format(self):
        """data.pressure.pressure_level nested format is extracted correctly."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [_make_pressure_event("warning", nested=True)]
        result = _extract_kernel_adapter_evidence(events)
        assert result["pressure_level_counts"].get("warning", 0) == 1
        assert result["worst_pressure_level"] == "warning"
        assert result["pressure_sample_count"] == 1

    def test_legacy_pressure_level_format(self):
        """Old data.pressure_level top-level format is extracted correctly."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [_make_pressure_event("critical", nested=False)]
        result = _extract_kernel_adapter_evidence(events)
        assert result["pressure_level_counts"].get("critical", 0) == 1
        assert result["worst_pressure_level"] == "critical"

    def test_pressure_sort_ordering(self):
        """Pressure ordering: normal < warning < overloaded < critical."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [
            _make_pressure_event("normal"),
            _make_pressure_event("warning"),
            _make_pressure_event("overloaded"),
            _make_pressure_event("critical"),
        ]
        result = _extract_kernel_adapter_evidence(events)
        assert result["worst_pressure_level"] == "critical"
        # overloaded > warning
        events2 = [_make_pressure_event("overloaded"), _make_pressure_event("warning")]
        r2 = _extract_kernel_adapter_evidence(events2)
        assert r2["worst_pressure_level"] == "overloaded"
        # warning > normal
        events3 = [_make_pressure_event("normal"), _make_pressure_event("warning")]
        r3 = _extract_kernel_adapter_evidence(events3)
        assert r3["worst_pressure_level"] == "warning"

    def test_system_and_agent_cgroup_scope_counting(self):
        """system and agent_cgroup scope events are counted separately."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [
            _make_pressure_event("normal", scope="system"),
            _make_pressure_event("normal", scope="system"),
            _make_pressure_event("normal", scope="agent_cgroup"),
        ]
        result = _extract_kernel_adapter_evidence(events)
        ec = result["event_counts"]
        assert ec["pressure_system"] == 2
        assert ec["pressure_agent_cgroup"] == 1
        assert ec["pressure_sampled"] == 3

    def test_quota_path_dedup_and_sort(self):
        """Quota paths are deduplicated and sorted."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [
            _make_cgroup_quota_event("/sys/fs/cgroup/agentos/agent_1"),
            _make_cgroup_quota_event("/sys/fs/cgroup/agentos/agent_0"),
            _make_cgroup_quota_event("/sys/fs/cgroup/agentos/agent_1"),  # duplicate
        ]
        result = _extract_kernel_adapter_evidence(events)
        paths = result["quota_paths"]
        assert len(paths) == 2  # deduplicated
        assert paths == sorted(paths)  # sorted
        assert "/sys/fs/cgroup/agentos/agent_0" in paths

    def test_bpf_mode_dedup(self):
        """BPF modes are deduplicated."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [
            _make_bpf_update_event("mock"),
            _make_bpf_update_event("mock"),
            _make_bpf_update_event("mock"),
        ]
        result = _extract_kernel_adapter_evidence(events)
        assert result["bpf_modes"] == ["mock"]
        assert result["event_counts"]["bpf_map_updated"] == 3

    def test_bpf_task_record_schema_version(self):
        """BPFTaskRecord schema_version is extracted from scheduler.decision events."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [_make_decision_event_with_schema("bpf.task_record.v2")]
        result = _extract_kernel_adapter_evidence(events)
        assert "bpf.task_record.v2" in result["bpf_record_schema_versions"]

    def test_kernel_probe_and_fallback_reason_aggregation(self):
        """Fallback reasons from events and kernel_probe are aggregated."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [_make_kernel_probe_event("event_fallback_reason")]
        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": "probe_fallback_reason",
        }
        result = _extract_kernel_adapter_evidence(events, kernel_probe=probe)
        assert "event_fallback_reason" in result["fallback_reasons"]
        assert result["fallback_active"] is True

    def test_kernel_probe_fallback_mode_activates_fallback(self):
        """fallback_active=True when kernel_probe.fallback_mode != 'none'."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = []
        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": "probe_only_reason",
        }
        result = _extract_kernel_adapter_evidence(events, kernel_probe=probe)
        assert result["fallback_active"] is True

    def test_safe_defaults_no_events(self):
        """With no events, returns sensible zero-value defaults."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        result = _extract_kernel_adapter_evidence([])
        assert result["pressure_sample_count"] == 0
        assert result["bpf_map_update_count"] == 0
        assert result["cgroup_quota_applied_count"] == 0
        assert result["worst_pressure_level"] == "normal"
        assert result["quota_paths"] == []
        assert result["bpf_modes"] == []
        assert result["fallback_reasons"] == []
        assert result["cgroup_control_observed"] is False
        assert "schema_version" in result
        assert "event_counts" in result

    def test_cgroup_observed_adapter_mode(self):
        """With cgroup_quota_applied events, adapter_mode uses cgroup_v2 prefix."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [_make_cgroup_quota_event()]
        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": "test",
        }
        result = _extract_kernel_adapter_evidence(events, kernel_probe=probe)
        assert result["adapter_mode"].startswith("cgroup_v2+")
        assert result["cgroup_control_observed"] is True

    def test_no_cgroup_adapter_mode(self):
        """Without cgroup events, adapter_mode uses user_space prefix."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [_make_bpf_update_event("mock")]
        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": "test",
        }
        result = _extract_kernel_adapter_evidence(events, kernel_probe=probe)
        assert result["adapter_mode"].startswith("user_space+")
        assert result["cgroup_control_observed"] is False

    def test_pinned_probe_adapter_mode(self):
        """pinned_probe BPF mode results in +pinned_probe suffix."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = []
        probe = {
            "sched_ext": "available",
            "bpf_map": "pinned_probe",
            "fallback_mode": "probe_only",
            "reason": "pinned map read-only probe succeeded",
        }
        result = _extract_kernel_adapter_evidence(events, kernel_probe=probe)
        assert result["adapter_mode"] == "user_space+pinned_probe"

    def test_unavailable_bpf_adapter_mode(self):
        """When kernel_probe is None, adapter_mode indicates unavailable BPF."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        result = _extract_kernel_adapter_evidence([], kernel_probe=None)
        assert result["adapter_mode"] == "user_space+unavailable"

    def test_summary_includes_kernel_adapter_evidence(self, tmp_path):
        """run_matrix summary.json contains kernel_adapter_evidence."""
        import asyncio
        from scripts.run_workload_matrix import run_matrix

        summary = asyncio.run(
            run_matrix("mock", tmp_path / "out", workloads=["ai_inference"])
        )
        kae = summary.get("kernel_adapter_evidence", {})
        assert kae.get("schema_version") == "kernel_adapter_evidence.v1"
        assert "adapter_mode" in kae
        assert "+" in kae["adapter_mode"]  # composite format

    def test_kernel_probe_reason_in_fallback_reasons(self):
        """kernel_probe.reason must appear in fallback_reasons list."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": "sched_ext not present; using mock BPF map",
        }
        result = _extract_kernel_adapter_evidence([], kernel_probe=probe)
        assert "sched_ext not present; using mock BPF map" in result["fallback_reasons"]
        assert result["fallback_active"] is True

    def test_probe_reason_and_event_reason_merged(self):
        """Event reasons and probe reason are merged, deduplicated, sorted."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        events = [_make_kernel_probe_event("event_reason")]
        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": "probe_reason",
        }
        result = _extract_kernel_adapter_evidence(events, kernel_probe=probe)
        assert "event_reason" in result["fallback_reasons"]
        assert "probe_reason" in result["fallback_reasons"]
        assert result["fallback_reasons"] == sorted(result["fallback_reasons"])

    def test_duplicate_probe_and_event_reason_deduplicated(self):
        """Same reason from event and probe appears only once."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        shared = "shared_fallback_reason"
        events = [_make_kernel_probe_event(shared)]
        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": shared,
        }
        result = _extract_kernel_adapter_evidence(events, kernel_probe=probe)
        assert result["fallback_reasons"].count(shared) == 1

    def test_empty_probe_reason_not_added(self):
        """Empty or whitespace-only probe reason is not added to fallback_reasons."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": "   ",
        }
        result = _extract_kernel_adapter_evidence([], kernel_probe=probe)
        assert "" not in result["fallback_reasons"]
        assert "   " not in result["fallback_reasons"]

    def test_probe_only_reason_saved_without_events(self):
        """Probe reason is preserved even with empty event list."""
        from scripts.run_workload_matrix import _extract_kernel_adapter_evidence

        probe = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
            "reason": "only_probe_reason",
        }
        result = _extract_kernel_adapter_evidence([], kernel_probe=probe)
        assert "only_probe_reason" in result["fallback_reasons"]
