import json
import math

import pytest

from agentos.comm.codec import JsonMessageCodec, MessageCodec, MessageCodecError
from agentos.core.types import Message


def test_comm_package_exports_codec_types():
    from agentos.comm import (
        JsonMessageCodec as ExportedJsonMessageCodec,
        MessageCodec as ExportedMessageCodec,
        MessageCodecError as ExportedMessageCodecError,
    )

    assert ExportedJsonMessageCodec is JsonMessageCodec
    assert ExportedMessageCodec is MessageCodec
    assert ExportedMessageCodecError is MessageCodecError


def test_full_message_round_trip_with_all_fields():
    codec: MessageCodec = JsonMessageCodec()
    message = Message(
        from_agent="planner",
        to_agent="worker",
        content={"steps": ["inspect", "patch"], "count": 2},
        message_type="task",
        priority=5,
        task_id="task-1",
        trace_id="trace-1",
        context_refs=["ctx://one", "ctx://two"],
        metadata={"source": "test", "attempt": 1},
        timestamp=123.45,
        message_id="msg-1",
    )

    encoded = codec.encode(message)
    wire = json.loads(json.dumps(encoded, ensure_ascii=False))
    decoded = codec.decode(wire)

    assert encoded == {
        "version": 1,
        "message_id": "msg-1",
        "from_agent": "planner",
        "to_agent": "worker",
        "content": {"steps": ["inspect", "patch"], "count": 2},
        "message_type": "task",
        "priority": 5,
        "task_id": "task-1",
        "trace_id": "trace-1",
        "context_refs": ["ctx://one", "ctx://two"],
        "metadata": {"source": "test", "attempt": 1},
        "timestamp": 123.45,
    }
    assert decoded == message


@pytest.mark.parametrize("field", ["from_agent", "to_agent", "content"])
def test_decode_missing_required_field_raises_codec_error(field: str):
    codec = JsonMessageCodec()
    data = {
        "from_agent": "planner",
        "to_agent": "worker",
        "content": {"ok": True},
    }
    del data[field]

    with pytest.raises(MessageCodecError, match=f"missing required field: {field}"):
        codec.decode(data)


def test_context_refs_are_preserved():
    codec = JsonMessageCodec()
    message = Message(
        from_agent="a",
        to_agent="b",
        content={"body": "see references"},
        context_refs=["ctx://alpha", "ctx://beta"],
    )

    decoded = codec.decode(codec.encode(message))

    assert decoded.context_refs == ["ctx://alpha", "ctx://beta"]


def test_optional_fields_default_on_decode():
    codec = JsonMessageCodec()

    decoded = codec.decode(
        {"from_agent": "planner", "to_agent": "worker", "content": {"ok": True}}
    )

    assert decoded.message_type == "generic"
    assert decoded.priority == 0
    assert decoded.task_id is None
    assert decoded.trace_id is None
    assert decoded.context_refs == []
    assert decoded.metadata == {}
    assert decoded.message_id.startswith("msg-")
    assert isinstance(decoded.timestamp, float)


def test_non_object_content_is_rejected():
    codec = JsonMessageCodec()

    with pytest.raises(MessageCodecError, match="content must be a JSON object"):
        codec.decode({"from_agent": "a", "to_agent": "b", "content": ["not-object"]})


def test_non_string_context_refs_are_rejected():
    codec = JsonMessageCodec()

    with pytest.raises(
        MessageCodecError, match="context_refs must be a list of strings"
    ):
        codec.decode(
            {
                "from_agent": "a",
                "to_agent": "b",
                "content": {},
                "context_refs": ["ctx://ok", 1],
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("from_agent", 123, "from_agent must be a string"),
        ("to_agent", 123, "to_agent must be a string"),
    ],
)
def test_invalid_agent_id_types_are_rejected(
    field: str, value: object, message: str
):
    codec = JsonMessageCodec()
    data = {"from_agent": "a", "to_agent": "b", "content": {}}
    data[field] = value

    with pytest.raises(MessageCodecError, match=message):
        codec.decode(data)


@pytest.mark.parametrize("value", ["high", True])
def test_invalid_priority_type_is_rejected(value: object):
    codec = JsonMessageCodec()

    with pytest.raises(MessageCodecError, match="priority must be an integer"):
        codec.decode(
            {"from_agent": "a", "to_agent": "b", "content": {}, "priority": value}
        )


def test_invalid_timestamp_type_is_rejected():
    codec = JsonMessageCodec()

    with pytest.raises(MessageCodecError, match="timestamp must be a number"):
        codec.decode(
            {"from_agent": "a", "to_agent": "b", "content": {}, "timestamp": True}
        )


def test_unsupported_version_is_rejected():
    codec = JsonMessageCodec()

    with pytest.raises(MessageCodecError, match="unsupported message version: 2"):
        codec.decode({"version": 2, "from_agent": "a", "to_agent": "b", "content": {}})


def test_bool_version_is_rejected():
    codec = JsonMessageCodec()

    with pytest.raises(MessageCodecError, match="version must be an integer"):
        codec.decode(
            {"version": True, "from_agent": "a", "to_agent": "b", "content": {}}
        )


def test_string_version_is_rejected():
    codec = JsonMessageCodec()

    with pytest.raises(MessageCodecError, match="version must be an integer"):
        codec.decode(
            {"version": "1", "from_agent": "a", "to_agent": "b", "content": {}}
        )


def test_encode_rejects_nan_content():
    codec = JsonMessageCodec()
    message = Message(from_agent="a", to_agent="b", content={"value": math.nan})

    with pytest.raises(MessageCodecError):
        codec.encode(message)


def test_encode_rejects_infinite_timestamp():
    codec = JsonMessageCodec()
    message = Message(from_agent="a", to_agent="b", content={}, timestamp=math.inf)

    with pytest.raises(MessageCodecError):
        codec.encode(message)


def test_encode_output_mutation_does_not_mutate_original_message():
    codec = JsonMessageCodec()
    message = Message(
        from_agent="a",
        to_agent="b",
        content={"nested": {"items": ["one"]}},
        context_refs=["ctx://one"],
        metadata={"nested": {"ok": True}},
    )

    encoded = codec.encode(message)
    encoded["content"]["nested"]["items"].append("two")
    encoded["context_refs"].append("ctx://two")
    encoded["metadata"]["nested"]["ok"] = False

    assert message.content == {"nested": {"items": ["one"]}}
    assert message.context_refs == ["ctx://one"]
    assert message.metadata == {"nested": {"ok": True}}


def test_decode_input_mutation_does_not_mutate_decoded_message():
    codec = JsonMessageCodec()
    data = {
        "from_agent": "a",
        "to_agent": "b",
        "content": {"nested": {"items": ["one"]}},
        "context_refs": ["ctx://one"],
        "metadata": {"nested": {"ok": True}},
    }

    decoded = codec.decode(data)
    data["content"]["nested"]["items"].append("two")
    data["context_refs"].append("ctx://two")
    data["metadata"]["nested"]["ok"] = False

    assert decoded.content == {"nested": {"items": ["one"]}}
    assert decoded.context_refs == ["ctx://one"]
    assert decoded.metadata == {"nested": {"ok": True}}
