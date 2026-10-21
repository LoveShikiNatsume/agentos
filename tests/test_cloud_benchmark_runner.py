"""Tests for cloud benchmark runner: schema, cost calculation, skip behavior."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.cloud_bench.common import RunMetrics, FrameworkResult, DEEPSEEK_INPUT_PRICE_CNY, DEEPSEEK_CACHED_INPUT_PRICE_CNY, DEEPSEEK_OUTPUT_PRICE_CNY, USD_CNY
from scripts.cloud_bench.software_task import build_task, AgentStep
from scripts.cloud_bench.deps import check_deps


def test_run_metrics_cost_calculation():
    """Cost is computed from real token counts."""
    m = RunMetrics(prompt_tokens=1000, completion_tokens=500, cache_hit_tokens=200, cache_miss_tokens=800)
    m.compute_cost()
    expected_cny = 200 * 0.1 / 1_000_000 + 800 * 0.5 / 1_000_000 + 500 * 2.0 / 1_000_000
    expected_usd = expected_cny / 7.2
    assert abs(m.api_cost - round(expected_usd, 6)) < 0.000001


def test_run_metrics_cost_zero_tokens():
    """Cost is 0 when no tokens."""
    m = RunMetrics()
    m.compute_cost()
    assert m.api_cost == 0.0


def test_run_metrics_context_reuse():
    """Context reuse is cache_hit_tokens / prompt_tokens * 100."""
    m = RunMetrics(prompt_tokens=1000, cache_hit_tokens=300)
    m.compute_context_reuse()
    assert m.context_reuse == 30.0


def test_run_metrics_context_reuse_zero_prompt():
    """Context reuse is 0 when prompt_tokens is 0."""
    m = RunMetrics(prompt_tokens=0, cache_hit_tokens=0)
    m.compute_context_reuse()
    assert m.context_reuse == 0.0


def test_run_metrics_to_display_dict():
    """Display dict has correct fields and scaling."""
    m = RunMetrics(
        avg_latency=3.14159,
        prompt_tokens=123456,
        api_cost=0.123,
        context_reuse=45.6,
        recovered_faults=2,
        observable_events=100,
    )
    d = m.to_display_dict()
    assert d["avg_latency"] == 3.14
    assert d["prompt_tokens"] == 123.5
    assert d["api_cost"] == 0.123
    assert d["context_reuse"] == 45.6
    assert d["recovered_faults"] == 2
    assert d["observable_events"] == 100


def test_run_metrics_display_dict_provenance_defaults():
    """Default provenance fields: token_source always present, others omitted."""
    m = RunMetrics(avg_latency=1.0, prompt_tokens=100)
    d = m.to_display_dict()
    assert "status" not in d          # default "ok" omitted
    assert d["token_source"] == "usage"  # always present
    assert "adapter_source" not in d  # default "sdk" omitted
    assert "note" not in d            # default "" omitted


def test_run_metrics_display_dict_provenance_non_defaults():
    """Non-default provenance fields ARE in display dict."""
    m = RunMetrics(
        avg_latency=1.0, prompt_tokens=100,
        status="degraded", token_source="estimated",
        adapter_source="direct_adapter", note="test note",
    )
    d = m.to_display_dict()
    assert d["status"] == "degraded"
    assert d["token_source"] == "estimated"
    assert d["adapter_source"] == "direct_adapter"
    assert d["note"] == "test note"


def test_framework_result_improvement():
    """Improvement computed when both sides ok + usage."""
    fr = FrameworkResult(
        framework="test",
        native=RunMetrics(avg_latency=4.0, prompt_tokens=1000, api_cost=0.2, context_reuse=20.0, observable_events=100),
        runtime=RunMetrics(avg_latency=2.0, prompt_tokens=600, api_cost=0.1, context_reuse=40.0, observable_events=500),
    )
    fr.compute_improvement()
    assert fr.improvement["latency"] == -50.0
    assert fr.improvement["tokens"] == -40.0
    assert fr.improvement["cost"] == -50.0
    assert fr.improvement["context_reuse"] == 20.0
    assert fr.improvement["events"] == 5.0


def test_framework_result_improvement_missing_native():
    """Improvement is empty when native is None."""
    fr = FrameworkResult(framework="test", native=None, runtime=RunMetrics())
    fr.compute_improvement()
    assert fr.improvement == {}


def test_framework_result_improvement_empty_when_native_error():
    """Improvement is empty when native has error status."""
    fr = FrameworkResult(
        framework="test",
        native=RunMetrics(status="error", error="something failed"),
        runtime=RunMetrics(avg_latency=2.0, prompt_tokens=600, api_cost=0.1, context_reuse=40.0, observable_events=500),
    )
    fr.compute_improvement()
    assert fr.improvement == {}


def test_framework_result_improvement_empty_when_runtime_error():
    """Improvement is empty when runtime has error status."""
    fr = FrameworkResult(
        framework="test",
        native=RunMetrics(avg_latency=4.0, prompt_tokens=1000, api_cost=0.2, context_reuse=20.0, observable_events=100),
        runtime=RunMetrics(status="error", error="runtime failed"),
    )
    fr.compute_improvement()
    assert fr.improvement == {}


def test_framework_result_improvement_empty_when_native_degraded():
    """Improvement is empty when native is degraded (even with non-zero metrics)."""
    fr = FrameworkResult(
        framework="test",
        native=RunMetrics(avg_latency=4.0, prompt_tokens=1000, api_cost=0.2, context_reuse=20.0, observable_events=100, status="degraded"),
        runtime=RunMetrics(avg_latency=2.0, prompt_tokens=600, api_cost=0.1, context_reuse=40.0, observable_events=500),
    )
    fr.compute_improvement()
    assert fr.improvement == {}


def test_framework_result_improvement_empty_when_native_estimated():
    """Improvement is empty when native has token_source=estimated."""
    fr = FrameworkResult(
        framework="test",
        native=RunMetrics(avg_latency=4.0, prompt_tokens=1000, api_cost=0.2, context_reuse=20.0, observable_events=100, token_source="estimated"),
        runtime=RunMetrics(avg_latency=2.0, prompt_tokens=600, api_cost=0.1, context_reuse=40.0, observable_events=500),
    )
    fr.compute_improvement()
    assert fr.improvement == {}


def test_framework_result_improvement_empty_when_runtime_estimated():
    """Improvement is empty when runtime has token_source=estimated."""
    fr = FrameworkResult(
        framework="test",
        native=RunMetrics(avg_latency=4.0, prompt_tokens=1000, api_cost=0.2, context_reuse=20.0, observable_events=100),
        runtime=RunMetrics(avg_latency=2.0, prompt_tokens=600, api_cost=0.1, context_reuse=40.0, observable_events=500, token_source="estimated"),
    )
    fr.compute_improvement()
    assert fr.improvement == {}


def test_framework_result_to_dict_shows_error_status():
    """to_dict includes status/error when not ok/degraded."""
    fr = FrameworkResult(
        framework="Test FW",
        native=RunMetrics(status="skipped", error="not installed"),
        runtime=RunMetrics(status="skipped", error="not installed"),
    )
    d = fr.to_dict()
    assert d["native"]["status"] == "skipped"
    assert "error" in d["native"]
    assert d["improvement"] == {}


def test_framework_result_to_dict_shows_degraded_provenance():
    """to_dict includes status/token_source/adapter_source/note when degraded."""
    fr = FrameworkResult(
        framework="Test FW",
        native=RunMetrics(
            avg_latency=4.0, prompt_tokens=1000, api_cost=0.2, context_reuse=20.0,
            status="degraded", token_source="estimated",
            adapter_source="direct_adapter", note="CrewAI estimated",
        ),
        runtime=RunMetrics(avg_latency=2.0, prompt_tokens=600, api_cost=0.1, context_reuse=40.0, observable_events=500),
    )
    fr.compute_improvement()
    d = fr.to_dict()
    assert d["native"]["status"] == "degraded"
    assert d["native"]["token_source"] == "estimated"
    assert d["native"]["adapter_source"] == "direct_adapter"
    assert d["native"]["note"] == "CrewAI estimated"
    # improvement is empty because native is degraded
    assert d["improvement"] == {}


def test_framework_result_to_dict():
    """to_dict matches cloud_benchmark.py schema for ok+usage."""
    fr = FrameworkResult(
        framework="Test FW",
        native=RunMetrics(avg_latency=4.0, prompt_tokens=600000, api_cost=0.18, context_reuse=20.0, recovered_faults=5, observable_events=100),
        runtime=RunMetrics(avg_latency=2.0, prompt_tokens=400000, api_cost=0.12, context_reuse=35.0, recovered_faults=5, observable_events=500),
    )
    fr.compute_improvement()
    d = fr.to_dict()
    assert d["framework"] == "Test FW"
    assert "avg_latency" in d["native"]
    assert "prompt_tokens" in d["native"]
    assert "api_cost" in d["native"]
    assert "latency" in d["improvement"]
    assert "tokens" in d["improvement"]


def test_build_task_returns_steps():
    """build_task returns 8 agent steps with correct DAG."""
    steps = build_task(max_tokens=256)
    assert len(steps) == 8
    roles = [s.role for s in steps]
    assert "architect" in roles
    assert "integrator" in roles
    arch = next(s for s in steps if s.role == "architect")
    assert arch.depends_on == []
    integ = next(s for s in steps if s.role == "integrator")
    assert "test_writer" in integ.depends_on
    assert "security_reviewer" in integ.depends_on
    assert "doc_writer" in integ.depends_on


def test_deps_check_returns_status():
    """check_deps returns a DepStatus for each framework."""
    for fw in ["openai_agents", "langgraph", "autogen", "crewai", "google_adk"]:
        status = check_deps(fw)
        assert status.name == fw
        assert isinstance(status.available, bool)
        assert isinstance(status.error, str)


def test_deps_check_unknown_framework():
    """Unknown framework returns not available."""
    status = check_deps("nonexistent_framework")
    assert not status.available
    assert "unknown" in status.error


def test_runtime_contract_only_on_runtime_side():
    """runtime_contract appears in runtime dict but not native dict."""
    fr = FrameworkResult(
        framework="Test",
        native=RunMetrics(
            avg_latency=4.0, prompt_tokens=1000, api_cost=0.2, context_reuse=20.0,
            orchestration_source="sdk_runner", model_path="deepseek_direct",
            runtime_contract="framework_orchestrated_runtime",
        ),
        runtime=RunMetrics(
            avg_latency=2.0, prompt_tokens=600, api_cost=0.1, context_reuse=40.0,
            orchestration_source="sdk_runner", model_path="agentos_model_gateway",
            runtime_contract="framework_orchestrated_runtime",
        ),
    )
    fr.compute_improvement()
    d = fr.to_dict()
    # Native side should NOT have runtime_contract
    assert "runtime_contract" not in d["native"]
    # But should have orchestration_source and model_path
    assert d["native"]["orchestration_source"] == "sdk_runner"
    assert d["native"]["model_path"] == "deepseek_direct"
    # Runtime side SHOULD have runtime_contract
    assert d["runtime"]["runtime_contract"] == "framework_orchestrated_runtime"
    assert d["runtime"]["orchestration_source"] == "sdk_runner"
    assert d["runtime"]["model_path"] == "agentos_model_gateway"


def test_verification_fields_in_display_dict():
    """framework_entrypoint_invoked and model_gateway_invoked in display dict."""
    m = RunMetrics(
        avg_latency=2.0, prompt_tokens=600,
        framework_entrypoint_invoked=True,
        model_gateway_invoked=True,
        runtime_mode_actual="framework_orchestrated",
    )
    d = m.to_display_dict(include_runtime_contract=True)
    assert d["framework_entrypoint_invoked"] is True
    assert d["model_gateway_invoked"] is True
    assert d["runtime_mode_actual"] == "framework_orchestrated"


def test_verification_fields_not_present_when_false():
    """Verification fields omitted when False (default)."""
    m = RunMetrics(avg_latency=2.0, prompt_tokens=600)
    d = m.to_display_dict()
    assert "framework_entrypoint_invoked" not in d
    assert "model_gateway_invoked" not in d
    assert "runtime_mode_actual" not in d
