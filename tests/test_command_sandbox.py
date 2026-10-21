"""Exercise actual OS boundaries in temporary workspaces, without model calls."""
import asyncio
import json
import os
import shutil
import socket
import subprocess

import pytest

from agentos.core.resources import ResourceQuota
from agentos.worker.process import WorkerSpec
from agentos.worker.sandbox import BubblewrapCommandWorker, SandboxUnavailable


@pytest.fixture(scope="module")
def bubblewrap():
    executable = shutil.which("bwrap")
    if not executable:
        pytest.skip("bubblewrap is not installed")
    probe = subprocess.run([executable, "--unshare-all", "--ro-bind", "/", "/", "--", "/usr/bin/true"], capture_output=True)
    if probe.returncode:
        pytest.skip("host does not permit bubblewrap namespaces: " + probe.stderr.decode(errors="replace"))
    return executable


def command(code, *, writable=False, network=False):
    return WorkerSpec("command", "task-check", "agent-check", "run-check",
                      payload={"argv": ["/usr/bin/python3", "-c", code]},
                      resource_quota=ResourceQuota(allow_filesystem_write=writable, allow_network=network))


async def test_workspace_write_external_files_hidden_and_credentials_absent(tmp_path, monkeypatch, bubblewrap):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "private.txt"
    external.write_text("host-private")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "do-not-inherit")
    code = ("from pathlib import Path; import os,json; "
            "Path('result.txt').write_text('ok'); "
            f"print(json.dumps({{'external_visible':Path({str(external)!r}).exists(),"
            "'credential_present':'DEEPSEEK_API_KEY' in os.environ,'cwd':os.getcwd()}))")
    result = await BubblewrapCommandWorker(workspace, executable=bubblewrap).run(command(code, writable=True))
    assert result.ok, result.stderr
    facts = json.loads(result.stdout)
    assert facts == {"external_visible": False, "credential_present": False, "cwd": str(workspace)}
    assert (workspace / "result.txt").read_text() == "ok"
    assert external.read_text() == "host-private"
    assert result.task_id == "task-check" and result.agent_id == "agent-check" and result.trace_id == "run-check"
    assert result.result["sandbox"]["cgroup_limits_applied"] is False


async def test_readonly_workspace_write_is_denied_by_os(tmp_path, bubblewrap):
    (tmp_path / "kept.txt").write_text("original")
    result = await BubblewrapCommandWorker(tmp_path, executable=bubblewrap).run(command("open('kept.txt','w').write('changed')"))
    assert not result.ok
    assert "Read-only file system" in result.stderr
    assert (tmp_path / "kept.txt").read_text() == "original"


async def test_explicit_dependency_visible_but_not_writable(tmp_path, bubblewrap):
    workspace, dependency = tmp_path / "workspace", tmp_path / "dependency"
    workspace.mkdir()
    dependency.mkdir()
    data = dependency / "module.txt"
    data.write_text("dependency")
    worker = BubblewrapCommandWorker(workspace, read_only_paths=[dependency], executable=bubblewrap)
    result = await worker.run(command(f"print(open({str(data)!r}).read()); open({str(data)!r},'w').write('bad')", writable=True))
    assert not result.ok and "dependency" in result.stdout
    assert "Read-only file system" in result.stderr
    assert data.read_text() == "dependency"


async def test_host_network_unreachable_unless_explicitly_allowed(tmp_path, bubblewrap):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        code = f"import socket; s=socket.create_connection(('127.0.0.1',{port}),0.3); s.close()"
        worker = BubblewrapCommandWorker(tmp_path, executable=bubblewrap)
        isolated = await worker.run(command(code))
        assert not isolated.ok
        shared = await worker.run(command(code, network=True))
        assert shared.ok, shared.stderr


async def test_timeout_removes_descendant_that_could_write_later(tmp_path, bubblewrap):
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "subprocess.Popen([sys.executable,'-c',\"import time; from pathlib import Path; time.sleep(1.5); Path('orphan.txt').write_text('bad')\"]); "
        "Path('ready.txt').write_text('ready'); print('ready',flush=True); time.sleep(30)"
    )
    result = await BubblewrapCommandWorker(tmp_path, executable=bubblewrap).run(command(code, writable=True), timeout_seconds=0.7)
    assert result.timed_out and not result.ok
    assert (tmp_path / "ready.txt").exists(), result.stderr
    await asyncio.sleep(1.2)
    assert not (tmp_path / "orphan.txt").exists()


async def test_large_output_is_drained_with_bounded_retention(tmp_path, bubblewrap):
    result = await BubblewrapCommandWorker(tmp_path, executable=bubblewrap, max_output_bytes=1024).run(command("import sys; sys.stdout.write('x'*200000); sys.stderr.write('y'*100000)"))
    assert result.ok, result.stderr
    assert len(result.stdout) == len(result.stderr) == 1024
    assert result.result["stdout_bytes"] == 200000 and result.result["stderr_bytes"] == 100000
    assert result.result["stdout_truncated"] and result.result["stderr_truncated"]


async def test_failed_command_does_not_break_next_command(tmp_path, bubblewrap):
    worker = BubblewrapCommandWorker(tmp_path, executable=bubblewrap)
    failed = await worker.run(command("raise SystemExit(7)"))
    assert failed.exit_code == 7 and not failed.ok
    good = await worker.run(command("print('still works')"))
    assert good.ok and good.stdout.strip() == "still works"


async def test_missing_backend_never_executes_on_host(tmp_path):
    worker = BubblewrapCommandWorker(tmp_path, executable="/nonexistent/agentos-bwrap")
    with pytest.raises(SandboxUnavailable):
        await worker.run(command("open('must-not-exist','w').write('bad')", writable=True))
    assert not (tmp_path / "must-not-exist").exists()


async def test_unimplemented_resource_limit_is_rejected(tmp_path, bubblewrap):
    spec = command("print('must not execute')")
    spec.resource_quota.memory_max_mb = 128
    with pytest.raises(ValueError, match="cgroup limits are not applied"):
        await BubblewrapCommandWorker(tmp_path, executable=bubblewrap).run(spec)


def test_dependency_cannot_expose_workspace_ancestor(tmp_path, bubblewrap):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="not contain the workspace"):
        BubblewrapCommandWorker(workspace, read_only_paths=[tmp_path], executable=bubblewrap)


@pytest.mark.parametrize("permission", ["allow_network", "allow_filesystem_write"])
async def test_string_permission_cannot_enable_access(tmp_path, permission):
    spec = command("print('must not execute')")
    setattr(spec.resource_quota, permission, "false")
    worker = BubblewrapCommandWorker(tmp_path, executable="/not-used-for-invalid-policy")
    with pytest.raises(ValueError, match="must be a boolean"):
        await worker.run(spec)
