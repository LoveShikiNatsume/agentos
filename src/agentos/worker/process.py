from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from agentos.core.resources import ResourceQuota


class SubprocessWorkerError(RuntimeError):
    """Raised when a subprocess worker exits unsuccessfully."""


@dataclass(frozen=True)
class WorkerSpec:
    worker_type: str
    task_id: str
    agent_id: str
    trace_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    resource_quota: ResourceQuota | None = None
    attempt: int = 0
    fault: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "worker_type": self.worker_type,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
            "payload": self.payload,
            "resource_quota": (
                self.resource_quota.to_dict()
                if self.resource_quota is not None
                else None
            ),
            "attempt": self.attempt,
            "fault": self.fault,
        }


@dataclass(frozen=True)
class WorkerResult:
    ok: bool
    worker_type: str
    task_id: str
    agent_id: str
    trace_id: str | None = None
    result: Any = None
    error: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_secs: float = 0.0
    timed_out: bool = False
    killed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OnStart = Callable[[int], None | Awaitable[None]]
OnSample = Callable[[], None | Awaitable[None]]


class SubprocessWorkerManager:
    def __init__(
        self,
        *,
        python_executable: str | None = None,
        module: str = "agentos.worker.entrypoint",
        allowed_cpus: set[int] | None = None,
        startup_barrier: bool = False,
        startup_barrier_timeout_seconds: float = 5.0,
    ) -> None:
        if startup_barrier_timeout_seconds <= 0:
            raise ValueError("startup_barrier_timeout_seconds must be positive")
        self.python_executable = python_executable or sys.executable
        self.module = module
        self.allowed_cpus = set(allowed_cpus) if allowed_cpus is not None else None
        self.startup_barrier = startup_barrier
        self.startup_barrier_timeout_seconds = startup_barrier_timeout_seconds

    def _prepare_child(self) -> None:
        if self.allowed_cpus is not None:
            os.sched_setaffinity(0, self.allowed_cpus)

    @staticmethod
    def _process_state(pid: int) -> str | None:
        try:
            raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            return None
        closing = raw.rfind(")")
        if closing < 0 or closing + 2 >= len(raw):
            return None
        return raw[closing + 2]

    async def _wait_startup_barrier(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stderr is None:
            raise SubprocessWorkerError("worker stderr pipe is unavailable")
        try:
            line = await asyncio.wait_for(
                proc.stderr.readline(), timeout=self.startup_barrier_timeout_seconds
            )
        except TimeoutError as exc:
            raise SubprocessWorkerError("worker startup barrier timed out") from exc
        if line != b"READY\n":
            raise SubprocessWorkerError(
                "worker startup barrier failed: "
                f"{line.decode('utf-8', errors='replace')!r}"
            )
        deadline = time.monotonic() + self.startup_barrier_timeout_seconds
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                raise SubprocessWorkerError(
                    f"worker exited before startup barrier: {proc.returncode}"
                )
            if self._process_state(proc.pid) == "S":
                return
            await asyncio.sleep(0.001)
        raise SubprocessWorkerError("worker did not block on startup barrier")

    async def run(
        self,
        spec: WorkerSpec,
        *,
        timeout_seconds: float | None = None,
        on_start: OnStart | None = None,
        on_sample: OnSample | None = None,
        sample_interval_seconds: float = 0.1,
        kill_after_start: bool = False,
    ) -> WorkerResult:
        if on_sample is not None and sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")

        start = time.monotonic()
        child_env = None
        child_module = self.module
        if self.startup_barrier:
            child_env = dict(os.environ)
            child_env["AGENTOS_WORKER_TARGET_MODULE"] = self.module
            child_module = "agentos.worker.startup_barrier"
        proc = await asyncio.create_subprocess_exec(
            self.python_executable,
            "-m",
            child_module,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=self._prepare_child if self.allowed_cpus is not None else None,
            env=child_env,
        )

        if self.startup_barrier:
            try:
                await self._wait_startup_barrier(proc)
            except BaseException:
                proc.kill()
                await proc.communicate()
                raise

        if on_start is not None:
            try:
                maybe_awaitable = on_start(proc.pid)
                if maybe_awaitable is not None:
                    await maybe_awaitable
            except BaseException:
                proc.kill()
                await proc.communicate()
                raise

        if kill_after_start:
            proc.kill()
            stdout_bytes, stderr_bytes = await proc.communicate()
            return WorkerResult(
                ok=False,
                worker_type=spec.worker_type,
                task_id=spec.task_id,
                agent_id=spec.agent_id,
                trace_id=spec.trace_id,
                error="worker killed after start",
                pid=proc.pid,
                exit_code=proc.returncode,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                duration_secs=time.monotonic() - start,
                killed=True,
            )

        input_bytes = json.dumps(spec.to_payload()).encode("utf-8")
        communicate_task = asyncio.create_task(
            proc.communicate(input=input_bytes)
        )
        try:
            if on_sample is None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=timeout_seconds,
                )
            else:
                deadline = (
                    time.monotonic() + timeout_seconds
                    if timeout_seconds is not None
                    else None
                )
                while True:
                    remaining = (
                        deadline - time.monotonic()
                        if deadline is not None
                        else None
                    )
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError
                    wait_seconds = (
                        sample_interval_seconds
                        if remaining is None
                        else min(sample_interval_seconds, remaining)
                    )
                    done, _pending = await asyncio.wait(
                        {communicate_task},
                        timeout=wait_seconds,
                    )
                    if communicate_task in done:
                        stdout_bytes, stderr_bytes = communicate_task.result()
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError
                    maybe_awaitable = on_sample()
                    if maybe_awaitable is not None:
                        await maybe_awaitable
        except TimeoutError:
            proc.kill()
            stdout_bytes, stderr_bytes = await communicate_task
            return WorkerResult(
                ok=False,
                worker_type=spec.worker_type,
                task_id=spec.task_id,
                agent_id=spec.agent_id,
                trace_id=spec.trace_id,
                error=f"worker timed out after {timeout_seconds}s",
                pid=proc.pid,
                exit_code=proc.returncode,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                duration_secs=time.monotonic() - start,
                timed_out=True,
            )
        except asyncio.CancelledError:
            proc.kill()
            try:
                await communicate_task
            finally:
                raise
        except BaseException:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await communicate_task
            finally:
                raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        duration = time.monotonic() - start

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            if proc.returncode != 0:
                return WorkerResult(
                    ok=False,
                    worker_type=spec.worker_type,
                    task_id=spec.task_id,
                    agent_id=spec.agent_id,
                    trace_id=spec.trace_id,
                    error=stderr.strip() or f"worker exited with code {proc.returncode}",
                    pid=proc.pid,
                    exit_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_secs=duration,
                )
            return WorkerResult(
                ok=False,
                worker_type=spec.worker_type,
                task_id=spec.task_id,
                agent_id=spec.agent_id,
                trace_id=spec.trace_id,
                error=f"worker returned invalid JSON: {exc}",
                pid=proc.pid,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_secs=duration,
            )

        return WorkerResult(
            ok=bool(parsed.get("ok", False)),
            worker_type=str(parsed.get("worker_type", spec.worker_type)),
            task_id=str(parsed.get("task_id", spec.task_id)),
            agent_id=str(parsed.get("agent_id", spec.agent_id)),
            trace_id=parsed.get("trace_id", spec.trace_id),
            result=parsed.get("result"),
            error=parsed.get("error"),
            pid=proc.pid,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_secs=duration,
        )
