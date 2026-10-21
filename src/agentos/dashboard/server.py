from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from agentos.dashboard.broadcast import EventBroadcaster
from agentos.events.log import EventLog
from agentos.events.types import RuntimeEvent


def preload_events(broadcaster: EventBroadcaster, log_path: Path) -> int:
    """Load historical events from a JSONL log into the broadcaster's history.

    Returns the number of events loaded.
    """
    if not log_path.exists():
        return 0
    log = EventLog(log_path)
    count = 0
    for event in log.read():
        broadcaster.publish(event)
        count += 1
    return count

_STATIC_DIR = Path(__file__).parent / "static"

_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/director.js": ("director.js", "application/javascript; charset=utf-8"),
    "/director.css": ("director.css", "text/css; charset=utf-8"),
}


class DashboardApp:
    """Lightweight async HTTP handler for the live dashboard."""

    def __init__(
        self,
        broadcaster: EventBroadcaster,
        workload_matrix_path: Path | None = None,
    ) -> None:
        self._broadcaster = broadcaster
        self._workload_matrix_path = workload_matrix_path

    async def handle_request(
        self, method: str, path: str
    ) -> tuple[int, dict[str, str], str]:
        # Strip query string so /?director=1 dispatches like / (Task 12 regression).
        path = path.split("?", 1)[0]
        if path in _STATIC_ROUTES:
            return self._serve_static(path)
        if path == "/api/snapshot":
            return self._serve_snapshot()
        if path == "/api/workload-matrix":
            return self._serve_workload_matrix()
        if path == "/api/cloud-benchmark":
            return self._serve_cloud_benchmark()
        return 404, {"Content-Type": "text/plain"}, "Not Found"

    def _serve_static(self, path: str) -> tuple[int, dict[str, str], str]:
        filename, content_type = _STATIC_ROUTES[path]
        body = (_STATIC_DIR / filename).read_text(encoding="utf-8")
        return 200, {"Content-Type": content_type}, body

    def _serve_snapshot(self) -> tuple[int, dict[str, str], str]:
        events = self._broadcaster.snapshot()
        payload = {"events": [e.to_dict() for e in events]}
        body = json.dumps(payload, ensure_ascii=False)
        return 200, {"Content-Type": "application/json"}, body

    def _serve_workload_matrix(self) -> tuple[int, dict[str, str], str]:
        if self._workload_matrix_path is None or not self._workload_matrix_path.exists():
            return 404, {"Content-Type": "application/json"}, json.dumps(
                {"error": "workload matrix not available"}
            )
        body = self._workload_matrix_path.read_text(encoding="utf-8")
        return 200, {"Content-Type": "application/json"}, body

    def _serve_cloud_benchmark(self) -> tuple[int, dict[str, str], str]:
        from agentos.dashboard.cloud_benchmark import cloud_benchmark_payload

        body = json.dumps(cloud_benchmark_payload(), ensure_ascii=False)
        return 200, {"Content-Type": "application/json"}, body

    async def sse_generator(self) -> "AsyncSSEStream":
        queue = self._broadcaster.subscribe()
        return AsyncSSEStream(queue, self._broadcaster)


class AsyncSSEStream:
    """Async iterator yielding SSE-formatted event strings."""

    def __init__(
        self,
        queue: asyncio.Queue[RuntimeEvent],
        broadcaster: EventBroadcaster,
    ) -> None:
        self._queue = queue
        self._broadcaster = broadcaster

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        event = await self._queue.get()
        data = json.dumps(event.to_dict(), ensure_ascii=False)
        return f"data: {data}\n\n"

    def close(self) -> None:
        self._broadcaster.unsubscribe(self._queue)


async def run_dashboard_server(
    broadcaster: EventBroadcaster,
    host: str = "127.0.0.1",
    port: int = 9400,
) -> None:
    """Start the async dashboard HTTP+SSE server."""
    # Resolve workload matrix path from env (may have been set by run_dashboard.py)
    wm_path_str = os.environ.get("DASHBOARD_WORKLOAD_MATRIX")
    wm_path = Path(wm_path_str) if wm_path_str else None

    app = DashboardApp(broadcaster, workload_matrix_path=wm_path)

    async def handle_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                writer.close()
                return
            parts = request_line.decode("utf-8", errors="replace").strip().split()
            if len(parts) < 2:
                writer.close()
                return
            method, path = parts[0], parts[1]

            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in (b"\r\n", b"\n", b""):
                    break

            if path == "/api/events":
                writer.write(b"HTTP/1.1 200 OK\r\n")
                writer.write(b"Content-Type: text/event-stream\r\n")
                writer.write(b"Cache-Control: no-cache\r\n")
                writer.write(b"Connection: keep-alive\r\n")
                writer.write(b"Access-Control-Allow-Origin: *\r\n")
                writer.write(b"\r\n")
                await writer.drain()

                stream = await app.sse_generator()
                try:
                    async for chunk in stream:
                        writer.write(chunk.encode("utf-8"))
                        await writer.drain()
                except (ConnectionError, asyncio.CancelledError):
                    pass
                finally:
                    stream.close()
            else:
                status, headers, body = await app.handle_request(method, path)
                body_bytes = body.encode("utf-8")
                status_text = "OK" if status == 200 else "Not Found"
                writer.write(f"HTTP/1.1 {status} {status_text}\r\n".encode())
                for key, value in headers.items():
                    writer.write(f"{key}: {value}\r\n".encode())
                writer.write(f"Content-Length: {len(body_bytes)}\r\n".encode())
                writer.write(b"\r\n")
                writer.write(body_bytes)
                await writer.drain()
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    server = await asyncio.start_server(handle_client, host, port)
    print(f"  Live dashboard: http://{host}:{port}/")
    async with server:
        await server.serve_forever()
