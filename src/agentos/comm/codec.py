from __future__ import annotations

import json
from typing import Any, Protocol

from agentos.core.types import Message


class MessageCodecError(ValueError):
    pass


class MessageCodec(Protocol):
    def encode(self, message: Message) -> dict[str, Any]:
        ...

    def decode(self, data: dict[str, Any]) -> Message:
        ...


class JsonMessageCodec:
    version = 1

    def encode(self, message: Message) -> dict[str, Any]:
        content = _require_json_object(message.content, "content")
        metadata = _require_json_object(message.metadata, "metadata")
        context_refs = _require_context_refs(message.context_refs)
        encoded = {
            "version": self.version,
            "message_id": message.message_id,
            "from_agent": message.from_agent,
            "to_agent": message.to_agent,
            "content": content,
            "message_type": message.message_type,
            "priority": message.priority,
            "task_id": message.task_id,
            "trace_id": message.trace_id,
            "context_refs": context_refs,
            "metadata": metadata,
            "timestamp": message.timestamp,
        }
        _ensure_json_serializable(encoded)
        return encoded

    def decode(self, data: dict[str, Any]) -> Message:
        if "version" in data:
            version = _require_integer(data["version"], "version")
            if version != self.version:
                raise MessageCodecError(f"unsupported message version: {version}")

        for field in ("from_agent", "to_agent", "content"):
            if field not in data:
                raise MessageCodecError(f"missing required field: {field}")

        kwargs: dict[str, Any] = {
            "from_agent": _require_string(data["from_agent"], "from_agent"),
            "to_agent": _require_string(data["to_agent"], "to_agent"),
            "content": _require_json_object(data["content"], "content"),
        }
        for field in ("message_id", "message_type"):
            if field in data:
                kwargs[field] = _require_string(data[field], field)
        for field in ("task_id", "trace_id"):
            if field in data:
                kwargs[field] = _require_optional_string(data[field], field)
        if "priority" in data:
            kwargs["priority"] = _require_integer(data["priority"], "priority")
        if "timestamp" in data:
            kwargs["timestamp"] = _require_number(data["timestamp"], "timestamp")
        if "context_refs" in data:
            kwargs["context_refs"] = _require_context_refs(data["context_refs"])
        if "metadata" in data:
            kwargs["metadata"] = _require_json_object(data["metadata"], "metadata")

        _ensure_json_serializable(kwargs)
        return Message(**kwargs)


def _require_json_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MessageCodecError(f"{field} must be a JSON object")
    copied = _copy_json_safe(value)
    if not isinstance(copied, dict):
        raise MessageCodecError(f"{field} must be a JSON object")
    return copied


def _require_context_refs(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(ref, str) for ref in value):
        raise MessageCodecError("context_refs must be a list of strings")
    copied = _copy_json_safe(value)
    if not isinstance(copied, list) or not all(isinstance(ref, str) for ref in copied):
        raise MessageCodecError("context_refs must be a list of strings")
    return copied


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise MessageCodecError(f"{field} must be a string")
    return value


def _require_optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field)


def _require_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MessageCodecError(f"{field} must be an integer")
    return value


def _require_number(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MessageCodecError(f"{field} must be a number")
    return value


def _ensure_json_serializable(value: Any) -> None:
    _copy_json_safe(value)


def _copy_json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise MessageCodecError("message contains non-JSON-serializable values") from exc
