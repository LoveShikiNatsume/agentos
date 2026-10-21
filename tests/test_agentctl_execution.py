from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from agentos.cli import main
from agentos.execution.bench import report, run_suite
from agentos.execution.config import resolve_config
from agentos.execution.service import execute
from agentos.llm.base import LLMResponse


def native_response(action, usage=None):
    """Provider-shaped fixture, including legal empty-content tool responses."""
    import uuid
    if "tasks" in action:
        content = json.dumps(action)
        message = {"role": "assistant", "content": content, "reasoning_content": "test planner reasoning"}
        finish = "stop"
    elif "tool" in action:
        content = ""
        message = {"role": "assistant", "content": None, "reasoning_content": "test tool reasoning",
                   "tool_calls": [{"id": "call_" + uuid.uuid4().hex, "type": "function",
                                   "function": {"name": action["tool"], "arguments": json.dumps(action.get("arguments", {}))}}]}
        finish = "tool_calls"
    else:
        content = action["final"]
        message = {"role": "assistant", "content": content, "reasoning_content": "test final reasoning"}
        finish = "stop"
    return LLMResponse(content, (usage or {}).get("total_tokens", 0), "test", 0,
                       metadata={"usage": usage, "assistant_message": message, "finish_reason": finish})


class ScriptedBackend:
    """Explicit deterministic test injection; never exposed as a CLI backend."""
    def __init__(self, *, usage=True, fail_first=False, wrong=False):
        self.requests = []
        self.counts = {}
        self.usage, self.fail_first, self.wrong = usage, fail_first, wrong

    async def generate(self, request):
        self.requests.append(request)
        tid = request.task_id
        n = self.counts[tid] = self.counts.get(tid, 0) + 1
        if self.fail_first and len(self.requests) == 1:
            raise RuntimeError("test transport failure")
        if tid == "planner":
            action = {"tasks": [{"id": "edit", "goal": "Read and update greeting.txt", "dependencies": []},
                                  {"id": "verify", "goal": "Verify actual contents", "dependencies": ["edit"]}]}
        elif tid == "edit":
            action = [{"tool": "read_file", "arguments": {"path": "greeting.txt"}},
                      {"tool": "write_file", "arguments": {"path": "greeting.txt", "content": "wrong\n" if self.wrong else "Hello, AgentOS!\n"}},
                      {"final": "Updated greeting.txt using actual tools."}][n - 1]
        else:
            action = [{"tool": "run_command", "arguments": {"argv": ["python3", "-c", "from pathlib import Path; print(Path('greeting.txt').read_text())"]}},
                      {"final": "Verified actual file."}][n - 1]
        return native_response(action, {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10} if self.usage else None)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "greeting.txt").write_text("hello\n")
    return root


def config(tmp_path, profile="runtime"):
    return {"model": "explicit-test", "base_url": "http://localhost:1", "output": str(tmp_path / "output"), "profile": profile}


def events(result):
    return [json.loads(line) for line in (Path(result["output"]) / "events.jsonl").read_text().splitlines()]


def test_config_precedence(workspace, tmp_path):
    user = tmp_path / "user.toml"
    user.write_text('model="user"\nbase_url="http://localhost:1"\nmax_steps=5\n')
    (workspace / ".agentos.toml").write_text('model="project"\nmax_tokens=88\n')
    result = resolve_config(workspace, {"model": "cli"}, user_path=user)
    assert result["model"] == "cli"
    assert result["max_steps"] == 5 and result["max_tokens"] == 88
    assert resolve_config(workspace, user_path=user)["model"] == "project"
    (workspace / ".agentos.toml").write_text('api_key="secret"\n')
    with pytest.raises(ValueError, match="api_key_env"):
        resolve_config(workspace, user_path=user)


@pytest.mark.parametrize("profile", ["runtime", "native"])
async def test_real_tools_and_profile(workspace, tmp_path, profile):
    backend = ScriptedBackend()
    result = await execute("Natural input: change the greeting and verify it", cwd=workspace,
                           overrides=config(tmp_path, profile), backend=backend)
    assert result["workflow_status"] == "completed", result["error"]
    assert (workspace / "greeting.txt").read_text() == "Hello, AgentOS!\n"
    assert all(status == "completed" for status in result["task_statuses"].values())
    assert result["usage"]["prompt_tokens"] == 42  # planner + 3 edit + 2 verify
    assert result["score"] == {"status": "not_requested", "passed": None}
    assert any("Natural input" in m["content"] for m in backend.requests[0].messages)
    kinds = [event["event_type"] for event in events(result)]
    assert "execution.tool_finished" in kinds
    if profile == "runtime":
        assert "task.completed" in kinds
        assert "tool.call_finished" in kinds
        assert "llm.response_received" in kinds
        assert "context.result_published" in kinds
        assert "context.assembled" in kinds
        assert "native.task_started" not in kinds
    else:
        assert "native.task_started" in kinds
        assert "task.completed" not in kinds
        assert "tool.call_finished" not in kinds
    verify = next(r for r in backend.requests if r.task_id == "verify")
    assert "result/edit" in verify.context_refs
    assert any("Updated greeting.txt" in str(m) for m in verify.messages)
    assert (Path(result["output"]) / "artifacts/files/greeting.txt").read_text() == "Hello, AgentOS!\n"


async def test_missing_usage_and_retry(workspace, tmp_path):
    result = await execute("change greeting", cwd=workspace, overrides=config(tmp_path),
                           backend=ScriptedBackend(usage=False, fail_first=True))
    assert result["workflow_status"] == "completed", result["error"]
    assert result["usage"]["retries"] == 1
    assert result["usage"]["failed_attempts"] == 1
    assert result["usage"]["prompt_tokens"] is None
    assert result["usage"]["missing_usage_calls"] == 6
    assert result["usage"]["failed_attempt_usage_unknown"] is True


async def test_failure_persists_and_redacts(workspace, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-token-value")
    class FailedBackend:
        async def generate(self, request):
            raise RuntimeError("failure with secret-token-value")
    result = await execute("do work", cwd=workspace, overrides=config(tmp_path), backend=FailedBackend())
    assert result["workflow_status"] == "failed"
    assert result["usage"]["attempts"] == 2
    assert result["usage"]["retries"] == 1
    for path in Path(result["output"]).glob("*.json*"):
        assert "secret-token-value" not in path.read_text()
    assert "[REDACTED]" in (Path(result["output"]) / "calls.jsonl").read_text()


async def test_batch_isolation_actual_scoring_and_report(workspace, tmp_path):
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"schema": "agentos.bench.v1", "tasks": [{
        "id": "one", "cwd": "workspace", "task": "update greeting", "score": {
            "type": "file_equals", "path": "greeting.txt", "expected": "Hello, AgentOS!\n"}}]}))
    counter = 0
    def factory():
        nonlocal counter
        counter += 1
        return ScriptedBackend(wrong=counter == 2)
    batch = await run_suite(str(manifest), output=str(tmp_path / "batches"), overrides=config(tmp_path), backend_factory=factory)
    assert (workspace / "greeting.txt").read_text() == "hello\n"
    rows = list(csv.DictReader(io.StringIO(report(batch["output"]))))
    assert [r["passed"] for r in rows] == ["True", "False"]
    assert [r["workflow_status"] for r in rows] == ["completed", "completed"]
    starts = [json.loads((Path(r["output"]) / "start.json").read_text())["files"] for r in batch["runs"]]
    assert starts[0] == starts[1]
    assert len({r["run_id"] for r in batch["runs"]}) == 2


def test_cli_forms_share_service(workspace, monkeypatch, capsys):
    calls = []
    async def fake(task, **kwargs):
        calls.append((task, kwargs["cwd"], kwargs["overrides"]))
        return {"workflow_status": "completed"}
    monkeypatch.setattr("agentos.execution.service.execute", fake)
    assert main(["do actual work", "--cwd", str(workspace), "--model", "m", "--json"]) == 0
    assert main(["exec", "do actual work", "--cwd", str(workspace), "--model", "m", "--json"]) == 0
    assert calls[0] == calls[1]
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_no_backend_no_fallback(workspace, capsys):
    assert main(["do work", "--cwd", str(workspace), "--json"]) == 2
    assert "no mock fallback" in capsys.readouterr().out


async def test_unconfigured_suite():
    with pytest.raises(ValueError, match="not configured"):
        await run_suite("swe-verified-30")


@pytest.mark.parametrize("profile", ["runtime", "native"])
async def test_real_fanout_fanin_and_read_overlap(workspace, tmp_path, profile):
    import asyncio
    class ParallelBackend:
        async def generate(self, request):
            if request.task_id == "planner":
                action = {"tasks": [
                    {"id": "first", "goal": "read first", "read_only": True},
                    {"id": "second", "goal": "read second", "read_only": True},
                    {"id": "consumer_a", "goal": "combine", "dependencies": ["first", "second"], "read_only": True},
                    {"id": "consumer_b", "goal": "combine also", "dependencies": ["first", "second"], "read_only": True}]}
            else:
                await asyncio.sleep(.03)
                action = {"final": f"result of {request.task_id}"}
            return native_response(action)
    result = await execute("inspect independently", cwd=workspace,
                           overrides={**config(tmp_path, profile), "max_parallel": 2}, backend=ParallelBackend())
    assert result["workflow_status"] == "completed", result["error"]
    stream = events(result)
    acquired = {e["task_id"]: e["monotonic_ns"] for e in stream if e["event_type"] == "workspace.access_acquired"}
    released = {e["task_id"]: e["monotonic_ns"] for e in stream if e["event_type"] == "workspace.access_released"}
    assert max(acquired["first"], acquired["second"]) < min(released["first"], released["second"])
    received = {(e["task_id"], e["data"]["sender"], e["data"]["ref"]) for e in stream if e["event_type"] == "message.reference_received"}
    if profile == "native":
        assert not received
        return
    for consumer in ("consumer_a", "consumer_b"):
        for producer in ("first", "second"):
            assert (consumer, f"agent-{producer}", f"result/{producer}") in received
    sent = [e for e in stream if e["event_type"] == "message.reference_sent"]
    assert len(sent) == len(received) == 8  # four planner notifications plus four dependency edges
    communication = result["communication"]
    assert communication["stats"]["message_count"] == communication["stats"]["context_ref_count"] == 8
    assert communication["stats"]["payload_bytes"] > 0
    assert "not model tokens" in communication["payload_bytes_scope"]
    for producer in ("first", "second"):
        published = next(e["monotonic_ns"] for e in stream if e["event_type"] == "context.result_published" and e["task_id"] == producer)
        assert all(e["monotonic_ns"] >= published for e in sent if e["task_id"] == producer)


@pytest.mark.parametrize("profile", ["runtime", "native"])
async def test_readonly_capability_and_write_exclusion(workspace, tmp_path, profile):
    import asyncio
    class AccessBackend:
        def __init__(self):
            self.counts = {}
        async def generate(self, request):
            tid = request.task_id
            self.counts[tid] = self.counts.get(tid, 0) + 1
            if tid == "planner":
                action = {"tasks": [
                    {"id": "reader", "goal": "try forbidden tools", "read_only": True},
                    {"id": "writer_a", "goal": "write"}, {"id": "writer_b", "goal": "write"}]}
            elif tid == "reader" and self.counts[tid] == 1:
                action = {"tool": "write_file", "arguments": {"path": "forbidden.txt", "content": "bad"}}
            elif tid == "reader" and self.counts[tid] == 2:
                action = {"tool": "run_command", "arguments": {"argv": ["touch", "forbidden-command.txt"]}}
            else:
                await asyncio.sleep(.02)
                action = {"final": "done"}
            return native_response(action)
    result = await execute("test capabilities", cwd=workspace,
                           overrides={**config(tmp_path, profile), "max_parallel": 3}, backend=AccessBackend())
    assert result["workflow_status"] == "completed", result["error"]
    assert not (workspace / "forbidden.txt").exists()
    assert not (workspace / "forbidden-command.txt").exists()
    stream = events(result)
    denied = [e for e in stream if e["event_type"] == "execution.tool_finished" and e["task_id"] == "reader"]
    assert len(denied) == 2 and all(not e["data"]["ok"] for e in denied)
    active = set()
    for event in stream:
        if event["event_type"] == "workspace.access_acquired":
            assert not active  # this plan has one reader and two exclusive writers
            active.add(event["task_id"])
        if event["event_type"] == "workspace.access_released":
            active.remove(event["task_id"])


async def test_failed_producer_never_publishes_or_notifies(workspace, tmp_path):
    class BadProducer:
        async def generate(self, request):
            if request.task_id == "planner":
                value = {"tasks": [{"id": "bad", "goal": "fail"},
                                    {"id": "consumer", "goal": "consume", "dependencies": ["bad"]},
                                    {"id": "independent", "goal": "independent read", "read_only": True}]}
            elif request.task_id == "independent":
                value = {"final": "Independent branch completed"}
            else:
                raise RuntimeError("producer failed")
            return native_response(value)
    result = await execute("failed task", cwd=workspace, overrides=config(tmp_path), backend=BadProducer())
    assert result["workflow_status"] == "failed"
    assert result["task_statuses"]["consumer"] == "skipped"
    assert result["task_statuses"]["independent"] == "completed"
    assert not any(e["task_id"] == "bad" and e["event_type"] in {"context.result_published", "message.reference_sent"} for e in events(result))


async def test_openai_wire_protocol_and_missing_usage(monkeypatch):
    import httpx
    from agentos.execution.backend import OpenAIBackend
    from agentos.llm.base import LLMRequest
    observed = []
    def handler(request):
        observed.append(request)
        return httpx.Response(200, json={"model": "actual-server-model", "choices": [
            {"message": {"content": '{"final":"done"}'}, "finish_reason": "stop"}]})
    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv("WIRE_TEST_KEY", "wire-secret")
    backend = OpenAIBackend({"model": "requested", "base_url": "http://local/v1", "api_key_env": "WIRE_TEST_KEY",
                             "timeout": 1, "temperature": 0.2, "max_tokens": 55, "reasoning_effort": "low", "response_format": "json_object"})
    response = await backend.generate(LLMRequest("agent", [{"role": "user", "content": "natural task"}]))
    assert str(observed[0].url) == "http://local/v1/chat/completions"
    assert observed[0].headers["Authorization"] == "Bearer wire-secret"
    payload = json.loads(observed[0].content)
    assert payload["model"] == "requested" and payload["max_tokens"] == 55 and payload["temperature"] == .2
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"] == {"type": "json_object"}
    assert response.model == "actual-server-model"
    assert response.metadata["usage"] is None
    assert response.tokens_used == 0  # core API placeholder, NOT reported as measured usage


async def test_truncated_response_is_not_workflow_success(workspace, tmp_path):
    class Truncated:
        async def generate(self, request):
            return LLMResponse('{"tasks":[{"id":"one","goal":"work"}]}', 5, "test", 0,
                metadata={"finish_reason": "length", "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    result = await execute("do work", cwd=workspace, overrides=config(tmp_path), backend=Truncated())
    assert result["workflow_status"] == "failed"
    assert result["usage"]["prompt_tokens"] == 3
    assert result["usage"]["completed_calls"] == 1


async def test_failed_attempt_usage_remains_unknown_even_after_retry(workspace, tmp_path):
    result = await execute("do work", cwd=workspace, overrides=config(tmp_path), backend=ScriptedBackend(fail_first=True))
    assert result["workflow_status"] == "completed"
    assert not result["usage"]["usage_complete"]
    assert result["usage"]["prompt_tokens"] is None
    assert result["usage"]["known_prompt_tokens"] == 42
    assert result["usage"]["usage_unknown_attempts"] == 1


async def test_model_credential_not_inherited_by_command(workspace, monkeypatch):
    import sys
    from types import SimpleNamespace
    from agentos.execution.tools import make_tools
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'must-not-reach-tool')
    manager = make_tools(workspace, {'command_timeout': 5, 'api_key_env': 'DEEPSEEK_API_KEY'},
                         SimpleNamespace(emit=lambda *args, **kwargs: None))
    result = await manager.call('run_command', allowed_tools=['run_command'], kwargs={'argv': [
        sys.executable, '-c', 'import os; print(os.environ.get("DEEPSEEK_API_KEY", "absent"))']})
    assert result.output['stdout'].strip() == 'absent'
    assert result.output['returncode'] == 0


@pytest.mark.parametrize("profile", ["runtime", "native"])
async def test_native_tools_wire_preserves_provider_messages(workspace, tmp_path, monkeypatch, profile):
    import httpx
    requests = []
    original_assistant = {"role": "assistant", "content": None, "reasoning_content": "provider reasoning preserved exactly α\nline2",
        "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"greeting.txt"}'}},
                       {"id": "call_b", "type": "function", "function": {"name": "list_files", "arguments": '{}'}}]}
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert "max_tokens" not in payload and "temperature" not in payload
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "high"
        if len(requests) == 1:
            assert "tools" not in payload
            message = {"role": "assistant", "reasoning_content": "planner thought", "content": json.dumps({"tasks": [{"id": "inspect", "goal": "Read greeting", "read_only": True}]})}
            finish = "stop"
        elif len(requests) == 2:
            assert {t['function']['name'] for t in payload['tools']} == {'read_file', 'list_files', 'search_files'}
            assert 'ctx' not in json.dumps(payload['tools'])
            message, finish = original_assistant, "tool_calls"
        else:
            assert len(requests) == 3
            assistant = next(m for m in payload['messages'] if m['role'] == 'assistant')
            assert assistant == original_assistant
            results = [m for m in payload['messages'] if m['role'] == 'tool']
            assert {m['tool_call_id'] for m in results} == {'call_a', 'call_b'}
            assert all(json.loads(m['content'])['ok'] for m in results)
            message, finish = {'role': 'assistant', 'content': 'Read actual greeting: hello', 'reasoning_content': 'final thought'}, 'stop'
        return httpx.Response(200, json={'model': 'deepseek-flash', 'choices': [{'message': message, 'finish_reason': finish}],
                                        'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    result = await execute('read greeting', cwd=workspace, overrides={**config(tmp_path, profile),
                           'max_tokens': None, 'thinking': 'enabled', 'reasoning_effort': 'high'})
    assert result['workflow_status'] == 'completed', result['error']
    assert len(requests) == 3
    assert result['config']['context_window'] == result['config']['generation_limit'] == 'provider_default'
    rows = [json.loads(line) for line in (Path(result['output'])/'calls.jsonl').read_text().splitlines()]
    assert rows[1]['assistant_message'] == original_assistant
    assert rows[1]['response'] == ''
    assert rows[2]['messages'][-3] == original_assistant
    assert result['usage']['prompt_tokens'] == 60
    assert len([e for e in events(result) if e['event_type'] == 'execution.tool_finished']) == 2


def test_explicit_provider_default_overrides_project_generation_cap(workspace, tmp_path):
    (workspace/'.agentos.toml').write_text('model="m"\nbase_url="http://localhost:1"\nmax_tokens=88\n')
    assert resolve_config(workspace, {'max_tokens': None})['max_tokens'] is None
    assert resolve_config(workspace, {'max_tokens': 'provider_default'})['generation_limit'] == 'provider_default'


async def test_total_run_timeout_records_cancelled_model_attempt(workspace, tmp_path):
    import asyncio
    class SlowBackend:
        async def generate(self, request):
            await asyncio.sleep(10)
    result = await execute('read file', cwd=workspace, overrides={**config(tmp_path), 'run_timeout': .03}, backend=SlowBackend())
    assert result['workflow_status'] == 'failed'
    assert result['usage']['attempts'] == result['usage']['failed_attempts'] == 1
    assert not result['usage']['usage_complete']
    assert result['usage']['prompt_tokens'] is None
