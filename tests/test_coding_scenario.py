from pathlib import Path

from agentos.scenarios.coding.pipeline import run_coding_pipeline


async def test_coding_scenario_runs_with_mock_backend(tmp_path: Path):
    log_path = tmp_path / "events.jsonl"

    results = await run_coding_pipeline(backend="mock", log_path=log_path)

    assert {"plan", "code", "test", "fix"}.issubset(results)
    assert log_path.exists()
    assert "fix" in results
