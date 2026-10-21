"""Linux command Worker with explicit filesystem/network/process boundaries.

This uses the existing WorkerSpec/WorkerResult contract. It does not apply cgroup
quotas, implement a general security sandbox, or silently fall back to host
execution. Only commands routed through this Worker receive these boundaries.
"""
from __future__ import annotations

import asyncio
import inspect
import math
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Sequence

from agentos.core.resources import ResourceQuota
from agentos.worker.process import OnStart, WorkerResult, WorkerSpec


class SandboxUnavailable(RuntimeError):
    """The required OS command isolation backend is unavailable."""


class BubblewrapCommandWorker:
    """Expose one workspace and explicit read-only dependencies to a command.

    The workspace may be writable according to ResourceQuota. /tmp is private
    and writable; system executables/libraries and dependency mounts are read
    only. Host home, /run and environment variables are not inherited. The
    process callback observes the supervisor PID; it is not a quota start barrier.
    """

    _SYSTEM_PATHS = (
        "/usr", "/bin", "/sbin", "/lib", "/lib64",
        "/etc/ld.so.cache", "/etc/passwd", "/etc/group", "/etc/nsswitch.conf",
    )
    _UNSUPPORTED_LIMITS = (
        "cpu_shares", "cpu_max_percent", "memory_max_mb", "pids_max",
        "io_read_bps", "io_write_bps",
    )

    def __init__(
        self,
        workspace: str | Path,
        *,
        read_only_paths: Sequence[str | Path] = (),
        executable: str | None = None,
        max_output_bytes: int = 65536,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        if self.workspace == Path("/") or self.workspace == Path.home().resolve():
            raise ValueError("workspace must be a project directory, not root or the whole home")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1:
            raise ValueError("max_output_bytes must be a positive integer")
        self.max_output_bytes = max_output_bytes
        self.executable = executable or shutil.which("bwrap")
        if not self.executable:
            raise SandboxUnavailable("bubblewrap is required; host fallback is disabled")
        self.read_only_paths = tuple(dict.fromkeys(
            Path(p).expanduser().resolve(strict=True) for p in read_only_paths
        ))
        for path in self.read_only_paths:
            if not (path.is_dir() or path.is_file()):
                raise ValueError(f"dependency must be a regular file or directory: {path}")
            if path == Path("/") or self.workspace.is_relative_to(path) or path.is_relative_to(self.workspace):
                raise ValueError("dependency mounts must be outside and not contain the workspace")
            if any(path == Path(root) or path.is_relative_to(root) for root in ("/proc", "/sys", "/dev")):
                raise ValueError("host kernel/device directories cannot be dependency mounts")

    def _command(self, spec: WorkerSpec) -> tuple[list[str], list[str], ResourceQuota]:
        if spec.worker_type != "command":
            raise ValueError("BubblewrapCommandWorker supports only worker_type='command'")
        argv = spec.payload.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and "\0" not in x for x in argv):
            raise ValueError("command argv must be a nonempty list of strings without NUL")
        if not argv[0]:
            raise ValueError("command executable must not be empty")
        quota = spec.resource_quota or ResourceQuota()
        for name in ("allow_network", "allow_filesystem_write"):
            if not isinstance(getattr(quota, name), bool):
                raise ValueError(f"{name} must be a boolean")
        requested = [name for name in self._UNSUPPORTED_LIMITS if getattr(quota, name) is not None]
        if requested:
            raise ValueError("cgroup limits are not applied by this Worker: " + ", ".join(requested))
        command = [self.executable, "--unshare-all", "--die-with-parent", "--new-session",
                   "--cap-drop", "ALL", "--clearenv"]
        if quota.allow_network:
            command.append("--share-net")
        for name in self._SYSTEM_PATHS:
            if Path(name).exists():
                command.extend(["--ro-bind", name, name])
        if quota.allow_network:
            for name in ("/etc/resolv.conf", "/etc/hosts", "/etc/ssl/certs"):
                if Path(name).exists():
                    command.extend(["--ro-bind", name, name])
        command.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                        "--dir", "/tmp/home", "--dir", "/tmp/cache", "--dir", "/run"])
        for path in self.read_only_paths:
            command.extend(["--ro-bind", str(path), str(path)])
        command.extend(["--bind" if quota.allow_filesystem_write else "--ro-bind",
                        str(self.workspace), str(self.workspace)])
        bins = [str(p / "bin") for p in self.read_only_paths if p.is_dir() and (p / "bin").is_dir()]
        environment = {
            "PATH": ":".join([*bins, "/usr/local/bin", "/usr/bin", "/bin"]),
            "HOME": "/tmp/home", "TMPDIR": "/tmp", "XDG_CACHE_HOME": "/tmp/cache",
            "LANG": "C.UTF-8", "USER": "agentos", "LOGNAME": "agentos",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
        }
        for key, value in environment.items():
            command.extend(["--setenv", key, value])
        command.extend(["--chdir", str(self.workspace), "--remount-ro", "/", "--", *argv])
        return command, argv, quota

    async def _capture(self, stream: asyncio.StreamReader) -> tuple[bytes, int]:
        retained = bytearray()
        total = 0
        while chunk := await stream.read(65536):
            total += len(chunk)
            remaining = self.max_output_bytes - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
        return bytes(retained), total

    @staticmethod
    async def _stop(process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.wait()

    async def run(
        self,
        spec: WorkerSpec,
        *,
        timeout_seconds: float | None = None,
        on_start: OnStart | None = None,
    ) -> WorkerResult:
        command, argv, quota = self._command(spec)
        timeout = timeout_seconds if timeout_seconds is not None else quota.timeout_seconds
        timeout = 60.0 if timeout is None else timeout
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        start = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailable("bubblewrap executable is unavailable; no host fallback") from exc
        readers = [asyncio.create_task(self._capture(stream)) for stream in (process.stdout, process.stderr)]

        async def wait() -> None:
            if on_start is not None:
                observed = on_start(process.pid)
                if inspect.isawaitable(observed):
                    await observed
            await process.wait()

        timed_out = False
        try:
            await asyncio.wait_for(wait(), timeout)
        except TimeoutError:
            timed_out = True
            await self._stop(process)
        except BaseException:
            await self._stop(process)
            await asyncio.gather(*readers)
            raise
        (stdout, stdout_bytes), (stderr, stderr_bytes) = await asyncio.gather(*readers)
        ok = not timed_out and process.returncode == 0
        result = {
            "argv": argv,
            "stdout_bytes": stdout_bytes, "stderr_bytes": stderr_bytes,
            "stdout_truncated": stdout_bytes > len(stdout), "stderr_truncated": stderr_bytes > len(stderr),
            "sandbox": {
                "backend": "bubblewrap", "workspace": str(self.workspace),
                "workspace_write_requested": quota.allow_filesystem_write,
                "read_only_dependencies": [str(p) for p in self.read_only_paths],
                "network_requested": "shared" if quota.allow_network else "isolated",
                "private_temporary_directory": True, "supervisor_pid": process.pid,
                "cgroup_limits_applied": False,
            },
        }
        error = "sandboxed command timed out" if timed_out else (None if ok else f"sandbox or command exited with {process.returncode}")
        return WorkerResult(
            ok=ok, worker_type=spec.worker_type, task_id=spec.task_id, agent_id=spec.agent_id,
            trace_id=spec.trace_id, result=result, error=error, pid=process.pid,
            exit_code=process.returncode, stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"), duration_secs=time.monotonic() - start,
            timed_out=timed_out, killed=timed_out,
        )
