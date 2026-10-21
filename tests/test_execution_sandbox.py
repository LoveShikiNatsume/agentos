"""New CLI/Runtime command-Worker integration; uses no model API calls."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agentos.cli import build_parser, main
from agentos.execution.cli import overrides
from agentos.execution.bench import run_suite
from agentos.execution.config import resolve_config
from agentos.execution.service import execute
from agentos.llm.base import LLMResponse
from agentos.worker.sandbox import SandboxUnavailable


@pytest.fixture(scope="module")
def bubblewrap():
    executable = shutil.which("bwrap")
    if not executable:
        pytest.skip("bubblewrap is not installed")
    probe = subprocess.run([executable, "--unshare-all", "--ro-bind", "/", "/", "--", "/usr/bin/true"], capture_output=True)
    if probe.returncode:
        pytest.skip("host does not permit bubblewrap namespaces")
    return executable


def native_response(action):
    content = json.dumps(action) if "tasks" in action else action.get("final", "")
    message = {"role": "assistant", "content": content, "reasoning_content": "explicit test fixture"}
    if "tool" in action:
        message["tool_calls"] = [{"id": "call-sandbox", "type": "function",
                                   "function": {"name": action["tool"], "arguments": json.dumps(action["arguments"])}}]
    return LLMResponse(content, 10, "explicit-test", 0, metadata={
        "assistant_message": message, "finish_reason": "tool_calls" if "tool" in action else "stop",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    })


class SandboxBackend:
    def __init__(self, outside, dependency):
        self.outside, self.dependency = outside, dependency
        self.requests = []
        self.worker_calls = 0
        self.observation = None

    async def generate(self, request):
        self.requests.append(request)
        if request.task_id == "planner":
            action = {"tasks": [{"id": "check", "goal": "Verify the configured command boundary", "dependencies": []}]}
        else:
            self.worker_calls += 1
            if self.worker_calls == 1:
                code = (
                    "from pathlib import Path\nimport json,os\n"
                    "Path('proof.txt').write_text('workspace write works')\n"
                    f"facts={{'outside_visible':Path({str(self.outside)!r}).exists(),"
                    "'secret_present':'UNRELATED_HOST_SECRET' in os.environ}\n"
                    f"facts['dependency_content']=Path({str(self.dependency)!r}).read_text()\n"
                    "try:\n"
                    f"    Path({str(self.dependency)!r}).write_text('must not change')\n"
                    "except OSError as exc:\n    facts['write_errno']=exc.errno\n"
                    "print(json.dumps(facts))\n"
                )
                action = {"tool": "run_command", "arguments": {"argv": ["/usr/bin/python3", "-c", code]}}
            else:
                tool = next(m for m in reversed(request.messages) if m["role"] == "tool")
                self.observation = json.loads(tool["content"])
                assert self.observation["ok"], self.observation
                assert self.observation["output"]["returncode"] == 0, self.observation
                action = {"final": "Actual sandbox command completed."}
        return native_response(action)


@pytest.mark.parametrize("profile", ["runtime", "native"])
async def test_model_tool_uses_actual_sandbox_with_trusted_owner(tmp_path, monkeypatch, bubblewrap, profile):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("UNRELATED_HOST_SECRET", "do-not-inherit")
    workspace, dependency_dir = tmp_path / "workspace", tmp_path / "dependency"
    workspace.mkdir()
    dependency_dir.mkdir()
    outside, dependency = tmp_path / "private.txt", dependency_dir / "module.txt"
    outside.write_text("private")
    dependency.write_text("public dependency")
    backend = SandboxBackend(outside, dependency)
    result = await execute("Check the actual command boundary", cwd=workspace, backend=backend,
                           overrides={"model": "explicit-test", "base_url": "http://localhost:1",
                                      "profile": profile, "output": str(tmp_path / "records"),
                                      "command_sandbox": "bubblewrap", "command_read_only_paths": [str(dependency_dir)]})
    assert result["workflow_status"] == "completed", result["error"]
    assert result["backend_injected_for_test"] is True
    observation = backend.observation["output"]
    assert observation["command_backend"] == "bubblewrap"
    assert observation["worker_agent_id"] == "agent-check"
    assert observation["worker_pid"] > 0 and observation["worker_trace_id"]
    assert observation["sandbox"]["cgroup_limits_applied"] is False
    facts = json.loads(observation["stdout"])
    assert facts == {"outside_visible": False, "secret_present": False,
                     "dependency_content": "public dependency", "write_errno": 30}
    assert (workspace / "proof.txt").read_text() == "workspace write works"
    assert dependency.read_text() == "public dependency" and outside.read_text() == "private"
    events = [json.loads(line) for line in (Path(result["output"]) / "events.jsonl").read_text().splitlines()]
    started = [e for e in events if e["event_type"] == "execution.command_worker_started"]
    assert len(started) == 1 and started[0]["task_id"] == "check"
    assert started[0]["agent_id"] == "agent-check"
    assert started[0]["data"]["worker_pid"] == observation["worker_pid"]
    assert result["config"]["command_sandbox"] == "bubblewrap"


def test_cli_sandbox_flags_override_project_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    workspace, dependency = tmp_path / "workspace", tmp_path / "dependency"
    workspace.mkdir()
    dependency.mkdir()
    (workspace / ".agentos.toml").write_text('model="test"\nbase_url="http://localhost:1"\ncommand_network=true\n')
    args = build_parser().parse_args(["exec", "task", "--command-sandbox", "bubblewrap",
                                     "--command-read-only-path", str(dependency), "--no-command-network"])
    config = resolve_config(workspace, overrides(args))
    assert config["command_sandbox"] == "bubblewrap"
    assert config["command_read_only_paths"] == [str(dependency)]
    assert config["command_network"] is False


async def test_missing_sandbox_stops_before_model_request(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    monkeypatch.setattr("agentos.worker.sandbox.shutil.which", lambda _: None)
    backend = SandboxBackend(tmp_path / "unused", tmp_path / "unused2")
    with pytest.raises(SandboxUnavailable):
        await execute("must not call model", cwd=tmp_path, backend=backend,
                      overrides={"model": "explicit-test", "base_url": "http://localhost:1",
                                 "command_sandbox": "bubblewrap"})
    assert not backend.requests


def test_missing_sandbox_is_a_clear_cli_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    monkeypatch.setattr("agentos.worker.sandbox.shutil.which", lambda _: None)
    code = main(["exec", "must not call model", "--cwd", str(tmp_path), "--model", "test",
                 "--base-url", "http://localhost:1", "--command-sandbox", "bubblewrap"])
    assert code == 2
    assert "host fallback is disabled" in capsys.readouterr().err


async def test_batch_command_scorer_uses_same_os_boundary(tmp_path, monkeypatch, bubblewrap):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("UNRELATED_HOST_SECRET", "do-not-inherit")
    workspace, dependencies = tmp_path / "source", tmp_path / "dependencies"
    workspace.mkdir()
    dependencies.mkdir()
    outside, dependency = tmp_path / "private.txt", dependencies / "module.txt"
    outside.write_text("private")
    dependency.write_text("public dependency")
    check = ("from pathlib import Path; import os; "
             "assert Path('proof.txt').read_text()=='workspace write works'; "
             f"assert not Path({str(outside)!r}).exists(); "
             "assert 'UNRELATED_HOST_SECRET' not in os.environ")
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"schema": "agentos.bench.v1", "config": {
        "model": "explicit-test", "base_url": "http://localhost:1", "command_sandbox": "bubblewrap",
        "command_read_only_paths": [str(dependencies)]},
        "tasks": [{"id": "case", "cwd": "source", "task": "Check sandbox",
                   "score": {"type": "command", "argv": ["/usr/bin/python3", "-c", check]}}]}))
    batch = await run_suite(str(manifest), output=str(tmp_path / "batches"),
                            backend_factory=lambda: SandboxBackend(outside, dependency))
    for row in batch["runs"]:
        result = json.loads((Path(row["output"]) / "summary.json").read_text())
        assert result["score"]["passed"] is True, result["score"]
        assert result["score"]["result"]["command_backend"] == "bubblewrap"
    assert not (workspace / "proof.txt").exists()


async def test_candidate_cannot_disable_scorer_sandbox_by_editing_config(tmp_path, monkeypatch, bubblewrap):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / ".agentos.toml").write_text('model="explicit-test"\nbase_url="http://localhost:1"\ncommand_sandbox="bubblewrap"\n')
    private = tmp_path / "private.txt"
    private.write_text("private")

    class EditConfig:
        def __init__(self): self.step = 0
        async def generate(self, request):
            if request.task_id == "planner":
                return native_response({"tasks": [{"id": "edit_config", "goal": "Try changing project policy", "dependencies": []}]})
            self.step += 1
            if self.step == 1:
                return native_response({"tool": "write_file", "arguments": {
                    "path": ".agentos.toml", "content": 'model="explicit-test"\nbase_url="http://localhost:1"\ncommand_sandbox="none"\n'}})
            return native_response({"final": "Changed project config in the isolated task copy."})

    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"schema": "agentos.bench.v1", "tasks": [
        {"id": "case", "cwd": "source", "task": "Exercise policy freezing", "score": {
            "type": "command", "argv": ["/usr/bin/python3", "-c",
                                          f"from pathlib import Path; assert not Path({str(private)!r}).exists()"]}}
    ]}))
    batch = await run_suite(str(manifest), compare="runtime", output=str(tmp_path / "batches"), backend_factory=EditConfig)
    result = json.loads((Path(batch["runs"][0]["output"]) / "summary.json").read_text())
    assert result["score"]["passed"] is True, result["score"]
    assert result["score"]["result"]["command_backend"] == "bubblewrap"
    assert 'command_sandbox="bubblewrap"' in (workspace / ".agentos.toml").read_text()


async def test_configured_credential_cannot_return_to_model_in_tool_content(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("TEST_PROVIDER_KEY", "private-test-credential")
    (tmp_path / "input.txt").write_text("provider=private-test-credential")

    class ReadInput:
        def __init__(self): self.step = 0
        async def generate(self, request):
            assert "private-test-credential" not in json.dumps(request.messages)
            if request.task_id == "planner":
                return native_response({"tasks": [{"id": "read", "goal": "Read input", "dependencies": [], "read_only": True}]})
            self.step += 1
            if self.step == 1:
                return native_response({"tool": "read_file", "arguments": {"path": "input.txt"}})
            assert "[REDACTED]" in json.dumps(request.messages)
            return native_response({"final": "Provider credential was not sent back as file content."})

    result = await execute("Read input", cwd=tmp_path, backend=ReadInput(), overrides={
        "model": "explicit-test", "base_url": "http://localhost:1", "api_key_env": "TEST_PROVIDER_KEY"})
    assert result["workflow_status"] == "completed", result["error"]
