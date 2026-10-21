import asyncio
import json

import pytest

from agentos.comm.bus import MessageBus
from agentos.comm.ipc import (
    MessageBusIPCError,
    UnixSocketMessageBusClient,
    UnixSocketMessageBusServer,
)
from agentos.core.types import Message


def _message(
    *,
    from_agent: str = "planner",
    to_agent: str = "worker",
    content: dict | None = None,
    message_id: str = "msg-1",
    context_refs: list[str] | None = None,
) -> Message:
    return Message(
        from_agent=from_agent,
        to_agent=to_agent,
        content=content or {"task": "inspect"},
        message_id=message_id,
        context_refs=context_refs or [],
        timestamp=1.0,
    )


@pytest.fixture
async def ipc_pair(tmp_path):
    bus = MessageBus()
    socket_path = tmp_path / "ipc" / "messagebus.sock"
    server = UnixSocketMessageBusServer(bus, socket_path)
    await server.start()
    client = UnixSocketMessageBusClient(socket_path)

    try:
        yield bus, server, client, socket_path
    finally:
        await server.stop()


async def _raw_request(socket_path, command: dict) -> dict:
    return await _raw_line(socket_path, json.dumps(command).encode("utf-8") + b"\n")


async def _raw_line(socket_path, line: bytes) -> dict:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    writer.write(line)
    await writer.drain()
    response = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return json.loads(response.decode("utf-8"))


async def test_client_ping_reaches_server(ipc_pair):
    _bus, server, client, _socket_path = ipc_pair

    assert await client.ping() == "pong"
    assert server.stats()["messages_received"] == 0
    assert server.stats()["messages_sent"] == 0


async def test_client_send_writes_into_runtime_bus_history(ipc_pair):
    bus, server, client, _socket_path = ipc_pair
    message = _message(message_id="client-send", content={"body": "hello"})

    assert await client.send(message) == "sent"

    history = bus.get_history("worker")
    assert len(history) == 1
    assert history[0].message_id == "client-send"
    assert history[0].content == {"body": "hello"}
    assert server.stats()["messages_sent"] == 1
    assert server.stats()["messages_received"] == 0


async def test_runtime_bus_send_then_client_receive_returns_message_with_context_refs(
    ipc_pair,
):
    bus, server, client, _socket_path = ipc_pair
    message = _message(
        message_id="runtime-send",
        content={"body": "see refs"},
        context_refs=["ctx://one", "ctx://two"],
    )
    await bus.send(message)

    received = await client.receive("worker")

    assert received is not None
    assert received.message_id == "runtime-send"
    assert received.content == {"body": "see refs"}
    assert received.context_refs == ["ctx://one", "ctx://two"]
    assert server.stats()["messages_sent"] == 0
    assert server.stats()["messages_received"] == 1


async def test_empty_client_receive_does_not_increment_message_stats(ipc_pair):
    _bus, server, client, _socket_path = ipc_pair

    assert await client.receive("worker") is None

    assert server.stats()["messages_sent"] == 0
    assert server.stats()["messages_received"] == 0


async def test_queue_depth_reports_agent_and_total(ipc_pair):
    bus, server, client, _socket_path = ipc_pair
    await bus.send(_message(to_agent="worker", message_id="worker-1"))
    await bus.send(_message(to_agent="worker", message_id="worker-2"))
    await bus.send(_message(to_agent="reviewer", message_id="reviewer-1"))

    assert await client.queue_depth("worker") == 2
    assert await client.queue_depth("reviewer") == 1
    assert await client.queue_depth() == 3
    assert server.stats()["messages_sent"] == 0
    assert server.stats()["messages_received"] == 0


async def test_unknown_command_returns_error_and_server_still_handles_ping(ipc_pair):
    _bus, server, client, socket_path = ipc_pair

    response = await _raw_request(socket_path, {"op": "dance"})

    assert response["ok"] is False
    assert "unknown op" in response["error"]
    assert server.stats()["command_errors"] == 1
    assert server.stats()["messages_sent"] == 0
    assert server.stats()["messages_received"] == 0
    assert await client.ping() == "pong"


async def test_client_raises_ipc_error_on_server_command_error(ipc_pair):
    _bus, _server, client, _socket_path = ipc_pair

    with pytest.raises(MessageBusIPCError, match="missing field: agent_id"):
        await client._request({"op": "receive"})


async def test_stop_closes_idle_client_connection_and_removes_socket(tmp_path):
    bus = MessageBus()
    socket_path = tmp_path / "ipc" / "messagebus.sock"
    server = UnixSocketMessageBusServer(bus, socket_path)
    await server.start()
    reader, writer = await asyncio.open_unix_connection(socket_path)
    await asyncio.sleep(0)

    await server.stop()

    assert not socket_path.exists()
    writer.write(json.dumps({"op": "ping"}).encode("utf-8") + b"\n")
    try:
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), timeout=0.5)
    except (BrokenPipeError, ConnectionError, OSError, asyncio.TimeoutError):
        response = b""
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    if response:
        decoded = json.loads(response.decode("utf-8"))
        assert decoded != {"ok": True, "result": "pong"}
    assert bus.queue_depth() == 0


async def test_start_refuses_to_unlink_regular_file_socket_path(tmp_path):
    bus = MessageBus()
    socket_path = tmp_path / "messagebus.sock"
    socket_path.write_text("not a socket")
    server = UnixSocketMessageBusServer(bus, socket_path)

    with pytest.raises(MessageBusIPCError, match="exists and is not a socket"):
        await server.start()

    assert socket_path.read_text() == "not a socket"


async def test_malformed_json_command_returns_error_and_increments_command_errors(
    ipc_pair,
):
    _bus, server, _client, socket_path = ipc_pair

    response = await _raw_line(socket_path, b"{bad-json\n")

    assert response["ok"] is False
    assert response["error"] == "invalid JSON command"
    assert server.stats()["command_errors"] == 1
    assert server.stats()["messages_sent"] == 0
    assert server.stats()["messages_received"] == 0


async def test_client_connection_failure_raises_ipc_error(tmp_path):
    client = UnixSocketMessageBusClient(tmp_path / "missing.sock")

    with pytest.raises(MessageBusIPCError, match="connection failed"):
        await client.ping()


async def test_client_invalid_json_response_raises_ipc_error(tmp_path):
    socket_path = tmp_path / "bad-response.sock"

    async def handle_bad_response(_reader, writer):
        writer.write(b"not-json\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handle_bad_response, path=socket_path)
    try:
        client = UnixSocketMessageBusClient(socket_path)
        with pytest.raises(MessageBusIPCError):
            await client.ping()
    finally:
        server.close()
        await server.wait_closed()
        socket_path.unlink(missing_ok=True)
