from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

from agentos.comm.bus import MessageBus
from agentos.comm.codec import JsonMessageCodec, MessageCodec, MessageCodecError
from agentos.core.types import Message


class MessageBusIPCError(RuntimeError):
    pass


class UnixSocketMessageBusServer:
    def __init__(
        self,
        bus: MessageBus,
        socket_path: str | Path,
        *,
        codec: MessageCodec | None = None,
    ) -> None:
        self._bus = bus
        self._socket_path = Path(socket_path)
        self._codec = codec if codec is not None else JsonMessageCodec()
        self._server: asyncio.AbstractServer | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._stopping = False
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "command_errors": 0,
            "connection_errors": 0,
        }

    async def start(self) -> None:
        if self._server is not None:
            return
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self._socket_path.exists():
            if not stat.S_ISSOCK(self._socket_path.stat().st_mode):
                raise MessageBusIPCError(
                    f"socket path exists and is not a socket: {self._socket_path}"
                )
            self._socket_path.unlink()
        self._stopping = False
        self._server = await asyncio.start_unix_server(
            self._accept_client,
            path=str(self._socket_path),
        )

    async def stop(self) -> None:
        self._stopping = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for writer in list(self._writers):
            writer.close()
        for writer in list(self._writers):
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                self._stats["connection_errors"] += 1

        current_task = asyncio.current_task()
        tasks = [task for task in self._client_tasks if task is not current_task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self._socket_path.exists():
            self._socket_path.unlink()

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _accept_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._stopping:
            writer.close()
            return

        task = asyncio.create_task(self._handle_client(reader, writer))
        self._client_tasks.add(task)
        task.add_done_callback(self._client_tasks.discard)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._writers.add(writer)
        try:
            line = await reader.readline()
            if not line:
                self._stats["connection_errors"] += 1
                return

            response = await self._response_for_line(line)
            writer.write(json.dumps(response, allow_nan=False).encode("utf-8") + b"\n")
            await writer.drain()
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError):
            self._stats["connection_errors"] += 1
        finally:
            self._writers.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                self._stats["connection_errors"] += 1

    async def _response_for_line(self, line: bytes) -> dict[str, Any]:
        try:
            command = json.loads(line.decode("utf-8"))
            if not isinstance(command, dict):
                raise _CommandError("command must be a JSON object")
            result = await self._dispatch(command)
            return {"ok": True, "result": result}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._stats["command_errors"] += 1
            return {"ok": False, "error": "invalid JSON command"}
        except (_CommandError, MessageCodecError) as exc:
            self._stats["command_errors"] += 1
            return {"ok": False, "error": str(exc)}

    async def _dispatch(self, command: dict[str, Any]) -> Any:
        op = _require_string(command, "op")
        if op == "ping":
            return "pong"
        if op == "send":
            message_data = _require_object(command, "message")
            message = self._codec.decode(message_data)
            await self._bus.send(message)
            self._stats["messages_sent"] += 1
            return "sent"
        if op == "receive":
            agent_id = _require_string(command, "agent_id")
            message = await self._bus.receive(agent_id)
            if message is not None:
                self._stats["messages_received"] += 1
            return self._codec.encode(message) if message is not None else None
        if op == "queue_depth":
            agent_id = command.get("agent_id")
            if agent_id is not None and not isinstance(agent_id, str):
                raise _CommandError("agent_id must be a string")
            return self._bus.queue_depth(agent_id)
        raise _CommandError(f"unknown op: {op}")


class UnixSocketMessageBusClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        codec: MessageCodec | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._socket_path = Path(socket_path)
        self._codec = codec if codec is not None else JsonMessageCodec()
        self._timeout = timeout

    async def ping(self) -> str:
        result = await self._request({"op": "ping"})
        if not isinstance(result, str):
            raise MessageBusIPCError("invalid ping response")
        return result

    async def send(self, message: Message) -> str:
        result = await self._request(
            {"op": "send", "message": self._codec.encode(message)}
        )
        if not isinstance(result, str):
            raise MessageBusIPCError("invalid send response")
        return result

    async def receive(self, agent_id: str) -> Message | None:
        result = await self._request({"op": "receive", "agent_id": agent_id})
        if result is None:
            return None
        if not isinstance(result, dict):
            raise MessageBusIPCError("invalid receive response")
        try:
            return self._codec.decode(result)
        except MessageCodecError as exc:
            raise MessageBusIPCError(str(exc)) from exc

    async def queue_depth(self, agent_id: str | None = None) -> int:
        command: dict[str, Any] = {"op": "queue_depth"}
        if agent_id is not None:
            command["agent_id"] = agent_id
        result = await self._request(command)
        if isinstance(result, bool) or not isinstance(result, int):
            raise MessageBusIPCError("invalid queue_depth response")
        return result

    async def _request(self, command: dict[str, Any]) -> Any:
        try:
            return await asyncio.wait_for(self._request_once(command), self._timeout)
        except asyncio.TimeoutError as exc:
            raise MessageBusIPCError("message bus IPC request timed out") from exc
        except (ConnectionError, OSError) as exc:
            raise MessageBusIPCError(f"message bus IPC connection failed: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise MessageBusIPCError(str(exc)) from exc

    async def _request_once(self, command: dict[str, Any]) -> Any:
        reader, writer = await asyncio.open_unix_connection(str(self._socket_path))
        try:
            payload = json.dumps(command, allow_nan=False).encode("utf-8") + b"\n"
            writer.write(payload)
            await writer.drain()

            line = await reader.readline()
            if not line:
                raise MessageBusIPCError("empty response from message bus IPC server")
            response = json.loads(line.decode("utf-8"))
            if not isinstance(response, dict):
                raise MessageBusIPCError("invalid response from message bus IPC server")
            ok = response.get("ok")
            if ok is True:
                return response.get("result")
            if ok is False:
                error = response.get("error", "message bus IPC command failed")
                raise MessageBusIPCError(str(error))
            raise MessageBusIPCError("invalid response from message bus IPC server")
        finally:
            writer.close()
            await writer.wait_closed()


class _CommandError(ValueError):
    pass


def _require_string(command: dict[str, Any], field: str) -> str:
    if field not in command:
        raise _CommandError(f"missing field: {field}")
    value = command[field]
    if not isinstance(value, str):
        raise _CommandError(f"{field} must be a string")
    return value


def _require_object(command: dict[str, Any], field: str) -> dict[str, Any]:
    if field not in command:
        raise _CommandError(f"missing field: {field}")
    value = command[field]
    if not isinstance(value, dict):
        raise _CommandError(f"{field} must be a JSON object")
    return value
