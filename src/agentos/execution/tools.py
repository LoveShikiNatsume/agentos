from __future__ import annotations

import asyncio
import os
import signal
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from agentos.tools.manager import ToolManager, ToolSpec
from agentos.execution.search import search_files as search_workspace
from agentos.worker.process import WorkerSpec
from agentos.worker.sandbox import BubblewrapCommandWorker

_INVOCATION_CONTEXT = ContextVar("agentos_tool_invocation_context", default=None)


@contextmanager
def tool_invocation(ctx):
    """Bind trusted framework context; never accept owner IDs from model arguments."""
    token = _INVOCATION_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _INVOCATION_CONTEXT.reset(token)

TOOL_HELP = """Available tools (one JSON action per turn):
list_files: {path: "."} lists one directory.
read_file: {path: "relative/file", start: 1, lines: 200} reads UTF-8 text.
search_files: {query: "literal symbol", path: ".", file_glob: "*.py"} searches bounded workspace text.
write_file: {path: "relative/file", content: "full new content"} writes a file.
run_command: {argv: ["python", "-m", "pytest", "tests/test_example.py"]} runs an executable in the workspace.
Commands are real processes, not a sandbox. Respect the user's task scope, preserve unrelated work, and do not publish, commit, or access credentials.
"""
READ_ONLY_TOOL_NAMES = ("list_files", "read_file", "search_files")
TOOL_NAMES = READ_ONLY_TOOL_NAMES + ("write_file", "run_command")


async def run_command(argv: list[str], cwd: Path, timeout: float, *, env: dict | None = None) -> dict:
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ValueError("argv must be a nonempty array of strings")
    process = await asyncio.create_subprocess_exec(*argv, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.PIPE, start_new_session=True, env=env)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except (TimeoutError, asyncio.CancelledError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.communicate()
        raise
    return {"argv": argv, "returncode": process.returncode,
            "stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}


def make_tools(root: Path, config: dict, records) -> ToolManager:
    sandbox = (BubblewrapCommandWorker(root, read_only_paths=config.get("command_read_only_paths", ()))
               if config.get("command_sandbox", "none") == "bubblewrap" else None)
    def path(value):
        candidate = (root / value).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("file path escapes working directory")
        return candidate

    def list_files(path="."):
        return [{"name": p.name, "directory": p.is_dir()} for p in sorted(resolve(path).iterdir())]

    resolve = path

    def read_file(path, start=1, lines=200):
        if start < 1 or lines < 1 or lines > 1000:
            raise ValueError("start >= 1 and 1 <= lines <= 1000 required")
        return "".join(resolve(path).read_text().splitlines(keepends=True)[start - 1:start - 1 + lines])

    def search_files(query, path=".", file_glob="*", max_results=50, case_sensitive=True):
        return search_workspace(root, query, path, file_glob, max_results, case_sensitive)

    def write_file(path, content):
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {"path": path, "bytes": len(content.encode())}

    async def command(argv):
        ctx = _INVOCATION_CONTEXT.get()
        task_id = ctx.task.task_id if ctx is not None else None
        if sandbox is not None:
            if ctx is None:
                raise ValueError("sandboxed command requires trusted task invocation context")
            acb = getattr(ctx, "acb", None)
            spec = WorkerSpec(
                "command", task_id, acb.agent_id if acb else f"agent-{task_id}",
                acb.trace_id if acb else records.run_id, payload={"argv": argv},
                resource_quota=ctx.task.resource_requirements,
            )
            def started(pid):
                records.emit("execution.command_worker_started", task_id, worker_pid=pid,
                             worker_agent_id=spec.agent_id, worker_trace_id=spec.trace_id,
                             command_backend="bubblewrap")
            worker = await sandbox.run(spec, timeout_seconds=config["command_timeout"], on_start=started)
            result = {"argv": argv, "returncode": worker.exit_code,
                      "stdout": worker.stdout, "stderr": worker.stderr,
                      "worker_pid": worker.pid, "worker_agent_id": worker.agent_id,
                      "worker_trace_id": worker.trace_id, "timed_out": worker.timed_out,
                      "command_backend": "bubblewrap", **worker.result}
            records.emit("execution.command_worker_finished" if worker.ok else "execution.command_worker_failed",
                         task_id, worker_pid=worker.pid, returncode=worker.exit_code,
                         timed_out=worker.timed_out, duration_seconds=worker.duration_secs)
            records.emit("execution.command_result", task_id, **result)
            if worker.timed_out:
                raise TimeoutError(worker.error)
            return result
        tool_env = dict(os.environ)
        tool_env.pop(config["api_key_env"], None)
        result = await run_command(argv, root, config["command_timeout"], env=tool_env)
        result["command_backend"] = "host"
        records.emit("execution.command_result", task_id, **result)
        return result

    manager = ToolManager()
    for name, function in [("list_files", list_files), ("read_file", read_file), ("search_files", search_files), ("write_file", write_file), ("run_command", command)]:
        manager.register(ToolSpec(name, function, timeout_seconds=config["command_timeout"] + 5))
    return manager


def make_langgraph_tools(session):
    """Standard LangChain tool schemas + LangGraph injected execution context."""
    from typing import Annotated, Any
    from pydantic import create_model
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import InjectedState

    definitions = {
        "list_files": ("List entries of a directory inside the workspace.", {"path": (str, ".")}),
        "read_file": ("Read a selected range of lines from a UTF-8 workspace file.", {
            "path": (str, ...), "start": (int, 1), "lines": (int, 200)}),
        "search_files": ("Search workspace UTF-8 files for literal text without running commands or writing files. "
                         "file_glob matches workspace-relative paths. Returns paths, line numbers and bounded excerpts; "
                         "truncation/scan limits and skipped files are explicit. Narrow the path/query when truncated.", {
            "query": (str, ...), "path": (str, "."), "file_glob": (str, "*"),
            "max_results": (int, 50), "case_sensitive": (bool, True)}),
        "write_file": ("Replace a workspace file with the supplied complete content.", {
            "path": (str, ...), "content": (str, ...)}),
        "run_command": (("Run executable argv in the workspace inside bubblewrap; only the workspace, private /tmp and configured read-only dependencies are visible. "
                         + ("Network is allowed. " if session.config["command_network"] else "Network is disabled. ")
                         + "Output byte limits and truncation flags are reported explicitly. Commands may modify the workspace."
                         if session.config["command_sandbox"] == "bubblewrap" else
                         "Run an executable argv in the workspace, using the current user's OS permissions. Commands may modify files."), {
            "argv": (list[str], ...)})}
    # ToolNode may execute calls from one response concurrently. Serialize actual
    # functions so a batch cannot race its own writes; task-level readers still
    # overlap in model work, while the workspace lock owns inter-task access.
    operation_lock = asyncio.Lock()
    def build(name, description, fields):
        schema = create_model(name + "Arguments", **fields, ctx=(Annotated[Any, InjectedState("ctx")], ...))
        async def invoke(ctx, **kwargs):
            async with operation_lock:
                session.records.emit("execution.tool_started", ctx.task.task_id, tool=name, arguments=kwargs)
                try:
                    with tool_invocation(ctx):
                        result = await ctx.tool_call(name, **kwargs)
                    observation = {"ok": result.ok, "output": result.output, "error": result.error}
                except Exception as exc:
                    observation = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                # The provider credential must not enter model context via a
                # file/tool result, even when the caller selected host commands.
                observation = session.records.clean(observation)
                session.records.emit("execution.tool_finished", ctx.task.task_id, tool=name, **observation)
                return observation
        return StructuredTool.from_function(coroutine=invoke, name=name, description=description, args_schema=schema)
    return [build(name, description, fields) for name, (description, fields) in definitions.items()]
