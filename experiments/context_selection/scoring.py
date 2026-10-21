"""Exact answer scoring with strict JSON and Python-expression normalization."""
from __future__ import annotations

import ast
import json
import re
from typing import Any

_FIELDS = {"winner", "parameter", "default"}


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def normalize_answer(raw: str | dict[str, Any]) -> dict[str, str]:
    """Raise ValueError for malformed answers; do not extract JSON from prose.

    Identifier case is significant. Python default expressions are parsed and
    unparsed so whitespace, quoting and redundant parentheses do not matter.
    Expressions are never evaluated.
    """
    if isinstance(raw, str):
        try:
            value = json.loads(raw, object_pairs_hook=_object)
        except (ValueError, TypeError) as exc:
            raise ValueError("answer must be one JSON object") from exc
    else:
        value = raw
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("answer must have exactly winner, parameter and default")
    if any(not isinstance(value[key], str) for key in _FIELDS):
        raise ValueError("all answer values must be strings")
    result = {key: value[key].strip() for key in _FIELDS}
    if not re.fullmatch(r"[^\W\d]\w*(?:\.[^\W\d]\w*)+", result["winner"], re.UNICODE):
        raise ValueError("winner must be a qualified Python identifier")
    if not result["parameter"].isidentifier():
        raise ValueError("parameter must be a Python identifier")
    if not result["default"] or len(result["default"]) > 4000:
        raise ValueError("invalid default expression length")
    try:
        expression = ast.parse(result["default"], mode="eval")
        result["default"] = ast.unparse(expression.body)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("default must be a Python expression") from exc
    return result


def score_answer(raw: str | dict[str, Any], expected: str | dict[str, Any]) -> bool:
    try:
        return normalize_answer(raw) == normalize_answer(expected)
    except (TypeError, ValueError, RecursionError):
        return False
