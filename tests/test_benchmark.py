import json
from pathlib import Path

import pytest

from agentos.benchmarks.baseline_context import (
    BaselineExperiment,
    ExperimentResult,
    ExperimentScenario,
    CODING_3AGENT,
    CODING_5AGENT,
    MINIMAL_2AGENT,
)
from agentos.benchmarks.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    check_vllm_health,
    main,
    run_benchmark_cli,
)
from agentos.core.task import Task
from agentos.llm.mock import MockLLMBackend
from agentos.runtime.engine import AgentHandler


class _SimpleHandler(AgentHandler):
    async def run(self, ctx: AgentHandler.Context):
        resp = await ctx.llm_call([{"role": "user", "content": "do work"}])
        return resp.content


def _make_tasks() -> list[Task]:
    return [
        Task(task_id="t1", name="step1", agent_role="worker"),
        Task(task_id="t2", name="step2", agent_role="worker", dependencies=["t1"]),
    ]


class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult(mode="baseline", scenario="test")
        d = r.to_dict()
        assert d["mode"] == "baseline"
        assert d["scenario"] == "test"
        assert "total_latency_secs" in d
        assert "recovery_time_secs" in d
        assert "fault_count" in d
        assert "event_count" in d

    def test_to_json(self):
        r = BenchmarkResult(mode="runtime", scenario="test", llm_call_count=3)
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["llm_call_count"] == 3


class TestBenchmarkRunner:
    async def test_run_baseline(self):
        mock = MockLLMBackend(responses={"default": "baseline output"})
        runner = BenchmarkRunner(
            backend=mock,
            handlers={"worker": _SimpleHandler()},
            scenario_name="test_scenario",
        )
        result = await runner.run_baseline(_make_tasks())
        assert result.mode == "baseline"
        assert result.scenario == "test_scenario"
        assert result.llm_call_count == 2
        assert "t1" in result.task_results
        assert "t2" in result.task_results

    async def test_run_runtime(self):
        mock = MockLLMBackend(responses={"default": "runtime output"})
        runner = BenchmarkRunner(
            backend=mock,
            handlers={"worker": _SimpleHandler()},
        )
        shared = {"project_bg": "This is a fibonacci project"}
        result = await runner.run_runtime(_make_tasks(), shared_context=shared)
        assert result.mode == "runtime"
        assert result.llm_call_count == 2

    async def test_run_comparison(self):
        mock = MockLLMBackend(responses={"default": "output"})
        runner = BenchmarkRunner(
            backend=mock,
            handlers={"worker": _SimpleHandler()},
        )
        results = await runner.run_comparison(
            tasks_baseline=_make_tasks(),
            tasks_runtime=_make_tasks(),
            shared_context={"bg": "shared background"},
        )
        assert "baseline" in results
        assert "runtime" in results
        assert results["baseline"].llm_call_count == 2
        assert results["runtime"].llm_call_count == 2

    async def test_per_call_metrics_collected(self):
        mock = MockLLMBackend(responses={"default": "reply"})
        runner = BenchmarkRunner(
            backend=mock,
            handlers={"worker": _SimpleHandler()},
        )
        result = await runner.run_baseline(_make_tasks())
        assert len(result.per_call_metrics) == 2
        for m in result.per_call_metrics:
            assert "agent_id" in m
            assert "latency_secs" in m

    async def test_save_results(self, tmp_path: Path):
        mock = MockLLMBackend(responses={"default": "reply"})
        runner = BenchmarkRunner(
            backend=mock,
            handlers={"worker": _SimpleHandler()},
        )
        results = await runner.run_comparison(
            tasks_baseline=_make_tasks(),
            tasks_runtime=_make_tasks(),
        )
        output = BenchmarkRunner.save_results(results, tmp_path / "out")
        assert output.exists()
        data = json.loads(output.read_text())
        assert "baseline" in data
        assert "runtime" in data
        assert "fault_count" in data["baseline"]

    async def test_save_csv_results(self, tmp_path: Path):
        results = {
            "baseline": BenchmarkResult(
                mode="baseline",
                scenario="test",
                fault_count=1,
                recovery_time_secs=0.2,
                event_count=3,
            ),
            "runtime": BenchmarkResult(
                mode="runtime",
                scenario="test",
                fault_count=0,
                recovery_time_secs=0.0,
                event_count=4,
            ),
        }

        output = BenchmarkRunner.save_csv(results, tmp_path / "out")

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "recovery_time_secs" in content
        assert "fault_count" in content
        assert "event_count" in content

    def test_benchmark_cli_writes_json_and_csv(self, tmp_path: Path):
        rc = main(["--backend", "mock", "--output", str(tmp_path)])

        assert rc == 0
        assert (tmp_path / "benchmark_results.json").exists()
        assert (tmp_path / "benchmark_results.csv").exists()
        data = json.loads((tmp_path / "benchmark_results.json").read_text())
        assert "baseline" in data
        assert "runtime" in data

    async def test_benchmark_cli_writes_formal_report(self, tmp_path: Path):
        await run_benchmark_cli(
            backend_name="mock",
            output=tmp_path,
            runs=2,
            cooldown=0,
            report=True,
            command=["benchmark", "--backend", "mock"],
        )

        assert (tmp_path / "run_001" / "benchmark_results.json").exists()
        assert (tmp_path / "run_002" / "benchmark_results.csv").exists()
        assert (tmp_path / "benchmark_results.json").exists()
        assert (tmp_path / "benchmark_results.csv").exists()
        assert (tmp_path / "benchmark_summary.json").exists()
        assert (tmp_path / "benchmark_report.md").exists()

        data = json.loads((tmp_path / "benchmark_results.json").read_text())
        summary = json.loads((tmp_path / "benchmark_summary.json").read_text())
        report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
        assert len(data["runs"]) == 2
        assert summary["run_count"] == 2
        assert "comparison" in summary
        assert "AgentOS Runtime vLLM Benchmark Report" in report
        assert "Prompt token savings" in report

    def test_summarize_runs_computes_comparison(self):
        runs = [
            {
                "baseline": BenchmarkResult(
                    mode="baseline",
                    scenario="s",
                    total_latency_secs=10.0,
                    total_prompt_tokens=100,
                ),
                "runtime": BenchmarkResult(
                    mode="runtime",
                    scenario="s",
                    total_latency_secs=8.0,
                    total_prompt_tokens=70,
                    cache_hit_ratio=0.5,
                ),
            },
            {
                "baseline": BenchmarkResult(
                    mode="baseline",
                    scenario="s",
                    total_latency_secs=12.0,
                    total_prompt_tokens=100,
                ),
                "runtime": BenchmarkResult(
                    mode="runtime",
                    scenario="s",
                    total_latency_secs=9.0,
                    total_prompt_tokens=80,
                    cache_hit_ratio=0.7,
                ),
            },
        ]

        summary = BenchmarkRunner.summarize_runs(runs)

        assert summary["modes"]["baseline"]["total_latency_secs"]["mean"] == 11.0
        assert summary["modes"]["runtime"]["cache_hit_ratio"]["mean"] == pytest.approx(0.6)
        assert summary["comparison"]["prompt_token_savings"] == 25.0
        assert summary["comparison"]["prompt_token_savings_pct"] == 0.25

    async def test_print_comparison(self, capsys):
        r1 = BenchmarkResult(mode="baseline", scenario="test", llm_call_count=2, cache_hit_ratio=0.0)
        r2 = BenchmarkResult(mode="runtime", scenario="test", llm_call_count=2, cache_hit_ratio=0.5)
        output = BenchmarkRunner.print_comparison({"baseline": r1, "runtime": r2})
        assert "BASELINE" in output
        assert "RUNTIME" in output
        assert "50.00%" in output


class TestVLLMHealthCheck:
    async def test_check_vllm_health_success(self, monkeypatch):
        class _Resp:
            def __init__(self, *, data=None, text=""):
                self._data = data or {}
                self.text = text

            def raise_for_status(self):
                return None

            def json(self):
                return self._data

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url):
                if url.endswith("/v1/models"):
                    return _Resp(data={"data": [{"id": "qwen2.5-7b-instruct"}]})
                return _Resp(text="vllm:prefix_cache_hits_total{} 1\n")

        monkeypatch.setattr("agentos.benchmarks.runner.httpx.AsyncClient", _Client)

        health = await check_vllm_health("http://localhost:8000")

        assert health["model"] == "qwen2.5-7b-instruct"
        assert health["metrics_available"] is True
        assert health["prefix_cache_metrics"] is True

    async def test_check_vllm_health_failure(self, monkeypatch):
        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url):
                raise RuntimeError("offline")

        monkeypatch.setattr("agentos.benchmarks.runner.httpx.AsyncClient", _Client)

        with pytest.raises(RuntimeError, match="vLLM health check failed"):
            await check_vllm_health("http://localhost:8000")


class TestExperimentScenario:
    def test_builtin_scenarios_have_matching_lengths(self):
        for sc in [CODING_3AGENT, CODING_5AGENT, MINIMAL_2AGENT]:
            assert len(sc.agents) == len(sc.questions), f"{sc.name}: agents/questions mismatch"
            assert len(sc.shared_context) > 0

    def test_scenario_fields(self):
        sc = ExperimentScenario(
            name="custom",
            agents=[("a", "role a")],
            shared_context="ctx",
            questions=["q?"],
            max_tokens=50,
        )
        assert sc.name == "custom"
        assert sc.max_tokens == 50


class TestBaselineExperiment:
    async def test_run_baseline(self):
        mock = MockLLMBackend()
        exp = BaselineExperiment(backend=mock, request_interval=0)
        result = await exp.run_baseline(MINIMAL_2AGENT)
        assert result.group == "baseline"
        assert result.scenario == "minimal_2agent"
        assert len(result.calls) == 2

    async def test_run_runtime(self):
        mock = MockLLMBackend()
        exp = BaselineExperiment(backend=mock, request_interval=0)
        result = await exp.run_runtime(MINIMAL_2AGENT)
        assert result.group == "runtime"
        assert len(result.calls) == 2

    async def test_run_comparison(self):
        mock = MockLLMBackend()
        exp = BaselineExperiment(backend=mock, request_interval=0)
        results = await exp.run_comparison(MINIMAL_2AGENT, cooldown=0)
        assert "baseline" in results
        assert "runtime" in results
        assert results["baseline"].group == "baseline"
        assert results["runtime"].group == "runtime"

    async def test_run_comparison_multiple_runs(self):
        mock = MockLLMBackend()
        exp = BaselineExperiment(backend=mock, request_interval=0)
        results = await exp.run_comparison(MINIMAL_2AGENT, cooldown=0, runs=2)
        assert len(results["baseline"].calls) == 4
        assert len(results["runtime"].calls) == 4

    async def test_3agent_scenario(self):
        mock = MockLLMBackend()
        exp = BaselineExperiment(backend=mock, request_interval=0)
        result = await exp.run_baseline(CODING_3AGENT)
        assert len(result.calls) == 3

    async def test_5agent_scenario(self):
        mock = MockLLMBackend()
        exp = BaselineExperiment(backend=mock, request_interval=0)
        result = await exp.run_runtime(CODING_5AGENT)
        assert len(result.calls) == 5

    async def test_export_json(self, tmp_path: Path):
        mock = MockLLMBackend()
        exp = BaselineExperiment(backend=mock, request_interval=0)
        results = await exp.run_comparison(MINIMAL_2AGENT, cooldown=0)
        out = BaselineExperiment.export_json(results, tmp_path / "report.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "baseline" in data
        assert "runtime" in data
        assert "timestamp" in data

    async def test_export_csv(self, tmp_path: Path):
        mock = MockLLMBackend()
        exp = BaselineExperiment(backend=mock, request_interval=0)
        results = await exp.run_comparison(MINIMAL_2AGENT, cooldown=0)
        out = BaselineExperiment.export_csv(results, tmp_path / "report.csv")
        assert out.exists()
        content = out.read_text()
        assert "baseline" in content
        assert "runtime" in content

    async def test_result_to_dict(self):
        r = ExperimentResult(scenario="test", group="baseline")
        d = r.to_dict()
        assert d["scenario"] == "test"
        assert d["group"] == "baseline"
