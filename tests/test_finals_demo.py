import json

import pytest

from agentos.cli import main
from agentos.events.log import EventLog
from agentos.scenarios.finals import run_finals_demo


def test_finals_command_writes_a_complete_honestly_labelled_bundle(tmp_path):
    output = tmp_path / "demo"
    assert main(["finals-demo", "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    requests = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    events = list(EventLog(output / "events.jsonl").read(skip_bad_lines=False))
    assert summary["passed"] is True
    assert summary["synthetic_inputs"] is True
    assert summary["performance_benchmark"] is False
    assert summary["token_source"] == "mock_character_estimate"
    assert len(requests) == 6
    assert {item["model"] for item in requests} == {"mock"}
    assert all(item["provider_usage"] is None for item in requests)
    assert {e.data["demo_case"] for e in events} == {"all_shared", "selected", "isolation", "retry"}
    assert len([e for e in events if e.event_type == "context.assembled"]) == 6
    assert "单分支" in (output / "report.md").read_text(encoding="utf-8")
    assert (output / "dashboard.html").stat().st_size > 1000


async def test_finals_demo_does_not_mix_or_overwrite_runs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("existing result")
    with pytest.raises(FileExistsError):
        await run_finals_demo(output)
    assert sentinel.read_text() == "existing result"
