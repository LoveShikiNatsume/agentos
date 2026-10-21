from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from enum import Enum
from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class RecoveryAction(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    FAIL = "fail"
    CONTEXT_RETRY = "context_retry"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    fallback_role: str | None = None
    reason: str = ""
    feedback: str = ""


class RecoveryPolicy(Protocol):
    def decide(self, error: BaseException, attempt: int) -> RecoveryDecision: ...


class RetryPolicy:
    def __init__(self, max_retries: int = 1) -> None:
        self.max_retries = max_retries

    def decide(self, error: BaseException, attempt: int) -> RecoveryDecision:
        if attempt <= self.max_retries:
            return RecoveryDecision(
                RecoveryAction.RETRY,
                reason=f"retry attempt {attempt}/{self.max_retries}",
            )
        return RecoveryDecision(RecoveryAction.FAIL, reason="retry exhausted")


class FallbackPolicy:
    def __init__(self, fallback_role: str) -> None:
        self.fallback_role = fallback_role

    def decide(self, error: BaseException, attempt: int) -> RecoveryDecision:
        if attempt == 1:
            return RecoveryDecision(
                RecoveryAction.FALLBACK,
                fallback_role=self.fallback_role,
                reason=f"fallback to {self.fallback_role}",
            )
        return RecoveryDecision(RecoveryAction.FAIL, reason="fallback failed")


class SkipPolicy:
    def decide(self, error: BaseException, attempt: int) -> RecoveryDecision:
        return RecoveryDecision(RecoveryAction.SKIP, reason="skip failed task")


class ContextRetryPolicy:
    def __init__(self, max_retries: int = 1) -> None:
        self.max_retries = max_retries

    def decide(self, error: BaseException, attempt: int) -> RecoveryDecision:
        if attempt <= self.max_retries:
            return RecoveryDecision(
                RecoveryAction.CONTEXT_RETRY,
                reason=f"context_retry attempt {attempt}/{self.max_retries}",
            )
        return RecoveryDecision(
            RecoveryAction.FAIL,
            reason="context_retry exhausted",
        )


@dataclass(frozen=True)
class ValidationResult:
    """An application-owned acceptance decision; Runtime does not judge reasoning."""

    passed: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or not isinstance(self.reason, str):
            raise TypeError("ValidationResult requires a bool and a string reason")


Validator = Callable[[Any], ValidationResult | Awaitable[ValidationResult]]


class SemanticValidationError(RuntimeError):
    failure_type = "semantic"


class RepetitionError(RuntimeError):
    failure_type = "repetition"


class ValidationExecutionError(RuntimeError):
    """A broken validator, distinct from a rejected agent answer."""

    failure_type = "validation_error"


def failure_type(error: BaseException) -> str:
    return getattr(error, "failure_type", "execution")


def normalized_result_hash(value: Any, *, normalize_text: bool = True) -> str:
    """Hash JSON-like values, with sorted keys, NFC and collapsed whitespace.

    Raw JSON objects/arrays are parsed first (duplicate keys/nonfinite values
    fall back to text). Case, numbers and list order remain significant.
    normalize_text=False keeps tool argument strings exact. Unsupported objects
    use repr, so applications with custom result types should return JSON data.
    """
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = child
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON value: {value}")

    if normalize_text and isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            value = json.loads(value, object_pairs_hook=unique_object,
                               parse_constant=invalid_constant)
        except (ValueError, TypeError, RecursionError):
            pass

    def normalize(item: Any) -> Any:
        if normalize_text and isinstance(item, str):
            return " ".join(unicodedata.normalize("NFC", item).split())
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item

    encoded = json.dumps(normalize(value), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SemanticRecoveryPolicy:
    """Bounded feedback retries with optional private-context snapshot restore.

    max_attempts includes the first attempt. Execution errors use ordinary
    RETRY/CONTEXT_RETRY semantics; semantic rejection adds validator feedback.
    Repetition and validator defects are terminal regardless of the budget.
    """

    def __init__(self, max_attempts: int = 2, *, restore_context: bool = True) -> None:
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.max_attempts = max_attempts
        self.restore_context = restore_context

    def decide(self, error: BaseException, attempt: int) -> RecoveryDecision:
        kind = failure_type(error)
        if kind in {"repetition", "validation_error"}:
            return RecoveryDecision(RecoveryAction.FAIL, reason=str(error))
        if attempt >= self.max_attempts:
            return RecoveryDecision(RecoveryAction.FAIL, reason="maximum attempts exhausted")
        action = RecoveryAction.CONTEXT_RETRY if self.restore_context else RecoveryAction.RETRY
        return RecoveryDecision(action, reason=f"retry after {kind} failure",
                                feedback=f"Previous attempt failed ({kind}): {error}")
