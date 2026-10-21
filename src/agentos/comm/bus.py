from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from agentos.comm.transport import InMemoryMessageTransport, MessageTransport
from agentos.core.types import Message


class MessageBus:
    def __init__(self, transport: MessageTransport | None = None) -> None:
        self._transport = (
            transport if transport is not None else InMemoryMessageTransport()
        )
        self._history: dict[str, list[Message]] = defaultdict(list)
        self._all_history: list[Message] = []

    @property
    def transport(self) -> MessageTransport:
        return self._transport

    async def send(self, message: Message) -> None:
        await self._transport.send(message)
        self._history[message.to_agent].append(message)
        self._all_history.append(message)

    async def receive(self, agent_id: str) -> Message | None:
        return await self._transport.receive(agent_id)

    def queue_depth(self, agent_id: str | None = None) -> int:
        return self._transport.queue_depth(agent_id)

    def get_history(
        self,
        agent_id: str | None = None,
        *,
        task_id: str | None = None,
        message_type: str | None = None,
    ) -> list[Message]:
        return self._filtered_history(
            agent_id=agent_id,
            task_id=task_id,
            message_type=message_type,
        )

    def stats(
        self,
        agent_id: str | None = None,
        *,
        task_id: str | None = None,
        message_type: str | None = None,
    ) -> dict[str, Any]:
        history = self._filtered_history(
            agent_id=agent_id,
            task_id=task_id,
            message_type=message_type,
        )
        by_agent: dict[str, int] = defaultdict(int)
        by_type: dict[str, int] = defaultdict(int)
        context_ref_count = 0
        payload_bytes = 0

        for message in history:
            by_agent[message.to_agent] += 1
            by_type[_message_type(message)] += 1
            context_ref_count += len(_context_refs(message))
            payload_bytes += _payload_bytes(message)

        return {
            "message_count": len(history),
            "context_ref_count": context_ref_count,
            "payload_bytes": payload_bytes,
            "by_agent": dict(sorted(by_agent.items())),
            "by_type": dict(sorted(by_type.items())),
        }

    def payload_savings(
        self,
        agent_id: str | None = None,
        *,
        task_id: str | None = None,
        inline_message_type: str = "inline",
        ref_message_type: str = "context_ref",
    ) -> dict[str, Any]:
        inline_messages = self._filtered_history(
            agent_id=agent_id,
            task_id=task_id,
            message_type=inline_message_type,
        )
        ref_messages = self._filtered_history(
            agent_id=agent_id,
            task_id=task_id,
            message_type=ref_message_type,
        )
        inline_payload_bytes = sum(
            _payload_bytes(message) for message in inline_messages
        )
        ref_payload_bytes = sum(_payload_bytes(message) for message in ref_messages)
        bytes_saved = inline_payload_bytes - ref_payload_bytes
        payload_reduction_ratio = (
            bytes_saved / inline_payload_bytes if inline_payload_bytes else 0.0
        )

        return {
            "inline_message_type": inline_message_type,
            "ref_message_type": ref_message_type,
            "inline_message_count": len(inline_messages),
            "ref_message_count": len(ref_messages),
            "inline_payload_bytes": inline_payload_bytes,
            "ref_payload_bytes": ref_payload_bytes,
            "bytes_saved": bytes_saved,
            "payload_reduction_ratio": payload_reduction_ratio,
            "context_ref_count": sum(
                len(_context_refs(message)) for message in ref_messages
            ),
        }

    def _filtered_history(
        self,
        *,
        agent_id: str | None,
        task_id: str | None,
        message_type: str | None,
    ) -> list[Message]:
        history = self._all_history if agent_id is None else self._history[agent_id]
        return [
            message
            for message in history
            if (task_id is None or _task_id(message) == task_id)
            and (message_type is None or _message_type(message) == message_type)
        ]


def _task_id(message: Message) -> str | None:
    return getattr(message, "task_id", None)


def _message_type(message: Message) -> str:
    value = getattr(message, "message_type", None)
    return str(value) if value is not None else "generic"


def _context_refs(message: Message) -> list[str]:
    refs = getattr(message, "context_refs", []) or []
    if isinstance(refs, list):
        return [ref for ref in refs if isinstance(ref, str)]
    return []


def _payload_bytes(message: Message) -> int:
    return len(
        json.dumps(message.content, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
    )
