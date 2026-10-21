"""Strict JSON scoring, with one protocol-wide enclosing fence allowance."""
import json
import re
from experiments.context_selection.scoring import normalize_answer as describe_answer
from experiments.context_selection.workflow import unique_object


def strip_fence(raw):
    match = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", raw, re.DOTALL | re.IGNORECASE)
    return (match.group(1), True) if match else (raw, False)


def parse_json(raw):
    text, stripped = strip_fence(raw)
    return json.loads(text, object_pairs_hook=unique_object), stripped


def _literal(value):
    """探测后登记：default 若为 JSON 数字或布尔，按 Python 字面量转成字符串（各组一致）。"""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    return value


def normalize_answer(raw, family):
    value = parse_json(raw)[0] if isinstance(raw, str) else raw
    if isinstance(value, dict) and "default" in value:
        value = {**value, "default": _literal(value["default"])}
    if family == "describe":
        return describe_answer(value)
    if family != "hop":
        raise ValueError("unknown question family")
    if not isinstance(value, dict) or set(value) != {"callee", "parameter_count", "parameter", "default"}:
        raise ValueError("hop answer requires exactly callee, parameter_count, parameter, default")
    if not isinstance(value["callee"], str) or not re.fullmatch(r"[^\W\d]\w*(?:\.[^\W\d]\w*)+", value["callee"].strip()):
        raise ValueError("callee must be a qualified identifier")
    if type(value["parameter_count"]) is not int or value["parameter_count"] < 0:
        raise ValueError("parameter_count must be a nonnegative integer")
    result = {**value, "callee": value["callee"].strip()}
    if value["parameter"] is None and value["default"] is None:
        return result
    normalized = describe_answer({"winner": value["callee"], "parameter": value["parameter"], "default": value["default"]})
    result.update(parameter=normalized["parameter"], default=normalized["default"])
    return result


def score_answer(raw, expected, family):
    try:
        return normalize_answer(raw, family) == normalize_answer(expected, family)
    except (ValueError, TypeError, RecursionError):
        return False
