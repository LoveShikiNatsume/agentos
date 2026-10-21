from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable[..., Any]
    timeout_seconds: float | None = None
    exclusive: bool = False
    description: str = ""


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    output: Any = None
    error: str | None = None
    duration_secs: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "output_preview": str(self.output)[:200],
            "error": self.error,
            "duration_secs": self.duration_secs,
            "timed_out": self.timed_out,
        }


class ToolError(RuntimeError):
    def __init__(self, message: str, result: ToolResult | None = None) -> None:
        self.result = result
        super().__init__(message)


class ToolPermissionError(ToolError):
    """Raised when an agent calls a tool outside its whitelist."""


class ToolTimeoutError(ToolError):
    """Raised when a tool exceeds its tool-level timeout."""


class ToolConflictError(ToolError):
    """Raised when an exclusive tool is already in use."""


class ToolExecutionError(ToolError):
    """Raised when the registered tool handler fails."""


class ToolManager:
    def __init__(self, *, default_timeout_seconds: float | None = None) -> None:
        self.default_timeout_seconds = default_timeout_seconds
        self._tools: dict[str, ToolSpec] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name:
            raise ValueError("tool name must not be empty")
        self._tools[spec.name] = spec
        self._locks.setdefault(spec.name, asyncio.Lock())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    async def call(
        self,
        name: str,
        *,
        kwargs: dict[str, Any] | None = None,
        allowed_tools: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> ToolResult:
        kwargs = dict(kwargs or {})
        allowed = set(allowed_tools or [])
        start = time.monotonic()

        spec = self._tools.get(name)
        if spec is None:
            result = ToolResult(
                name=name,
                ok=False,
                error=f"tool not registered: {name}",
                duration_secs=time.monotonic() - start,
            )
            raise ToolPermissionError(result.error or "tool not registered", result)

        if name not in allowed:
            result = ToolResult(
                name=name,
                ok=False,
                error=f"tool not allowed: {name}",
                duration_secs=time.monotonic() - start,
            )
            raise ToolPermissionError(result.error or "tool not allowed", result)

        if spec.exclusive:
            lock = self._locks[name]
            if lock.locked():
                result = ToolResult(
                    name=name,
                    ok=False,
                    error=f"exclusive tool already running: {name}",
                    duration_secs=time.monotonic() - start,
                )
                raise ToolConflictError(result.error or "tool conflict", result)
            await lock.acquire()
            try:
                return await self._call_registered(spec, kwargs, start)
            finally:
                lock.release()

        return await self._call_registered(spec, kwargs, start)

    async def _call_registered(
        self,
        spec: ToolSpec,
        kwargs: dict[str, Any],
        start: float,
    ) -> ToolResult:
        timeout = (
            spec.timeout_seconds
            if spec.timeout_seconds is not None
            else self.default_timeout_seconds
        )
        try:
            if timeout is None:
                output = await self._invoke(spec.handler, kwargs)
            else:
                output = await asyncio.wait_for(
                    self._invoke(spec.handler, kwargs),
                    timeout=timeout,
                )
        except TimeoutError:
            result = ToolResult(
                name=spec.name,
                ok=False,
                error=f"tool timed out after {timeout}s",
                duration_secs=time.monotonic() - start,
                timed_out=True,
            )
            raise ToolTimeoutError(result.error or "tool timed out", result)
        except Exception as exc:
            result = ToolResult(
                name=spec.name,
                ok=False,
                error=str(exc),
                duration_secs=time.monotonic() - start,
            )
            raise ToolExecutionError(str(exc), result) from exc

        return ToolResult(
            name=spec.name,
            ok=True,
            output=output,
            duration_secs=time.monotonic() - start,
        )

    @staticmethod
    async def _invoke(handler: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        value = handler(**kwargs)
        if inspect.isawaitable(value):
            return await value
        return value
