from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


class ContextPriority:
    """Named priority levels for prompt prefix ordering.

    Lower values appear earlier in the prefix. Segments with the same
    priority are sorted alphabetically by name for determinism.
    """

    SYSTEM_INSTRUCTION = 0
    TASK_BACKGROUND = 10
    REPO_SUMMARY = 20
    SHARED_CONTEXT = 30
    DEFAULT = 30
    AGENT_ROLE = 100


@dataclass(frozen=True)
class PrefixSegment:
    name: str
    content: str
    priority: int = ContextPriority.DEFAULT
    ordinal: int = 0

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (self.priority, self.ordinal, self.name)


@dataclass(frozen=True)
class PrefixSegmentFingerprint:
    name: str
    priority: int
    ordinal: int
    tokens: int
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "ordinal": self.ordinal,
            "tokens": self.tokens,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class PrefixOverlap:
    left_prefix_id: str
    right_prefix_id: str
    shared_prefix_chars: int
    shared_prefix_tokens: int
    left_tokens: int
    right_tokens: int
    left_overlap_ratio: float
    right_overlap_ratio: float
    shorter_overlap_ratio: float
    first_difference_index: int | None
    shared_segment_names: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_prefix_id": self.left_prefix_id,
            "right_prefix_id": self.right_prefix_id,
            "shared_prefix_chars": self.shared_prefix_chars,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "left_tokens": self.left_tokens,
            "right_tokens": self.right_tokens,
            "left_overlap_ratio": self.left_overlap_ratio,
            "right_overlap_ratio": self.right_overlap_ratio,
            "shorter_overlap_ratio": self.shorter_overlap_ratio,
            "first_difference_index": self.first_difference_index,
            "shared_segment_names": list(self.shared_segment_names),
            "reason": self.reason,
        }


class PrefixCanonicalizer:
    """Produces byte-identical prompt prefixes across agents for cache optimization."""

    @staticmethod
    def canonicalize(segments: list[PrefixSegment]) -> list[PrefixSegment]:
        return sorted(segments, key=lambda s: s.sort_key)

    @staticmethod
    def compute_prefix_id(segments: list[PrefixSegment]) -> str:
        canonical = PrefixCanonicalizer.canonicalize(segments)
        combined = "\n\n".join(seg.content for seg in canonical)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @staticmethod
    def build_prefix_content(segments: list[PrefixSegment]) -> str:
        canonical = PrefixCanonicalizer.canonicalize(segments)
        return "\n\n".join(seg.content for seg in canonical)

    @staticmethod
    def segment_fingerprints(
        segments: list[PrefixSegment],
    ) -> list[PrefixSegmentFingerprint]:
        canonical = PrefixCanonicalizer.canonicalize(segments)
        return [
            PrefixSegmentFingerprint(
                name=seg.name,
                priority=seg.priority,
                ordinal=seg.ordinal,
                tokens=PrefixCanonicalizer.estimate_tokens(seg.content),
                content_hash=hashlib.sha256(seg.content.encode()).hexdigest()[:16],
            )
            for seg in canonical
        ]

    @staticmethod
    def compute_overlap(
        left_segments: list[PrefixSegment],
        right_segments: list[PrefixSegment],
    ) -> PrefixOverlap:
        left = PrefixCanonicalizer.build_prefix_content(left_segments)
        right = PrefixCanonicalizer.build_prefix_content(right_segments)
        shared_chars = PrefixCanonicalizer.common_prefix_chars(left, right)
        shared_tokens = PrefixCanonicalizer.estimate_tokens(left[:shared_chars])
        left_tokens = PrefixCanonicalizer.estimate_tokens(left)
        right_tokens = PrefixCanonicalizer.estimate_tokens(right)

        left_ratio = shared_tokens / left_tokens if left_tokens else 0.0
        right_ratio = shared_tokens / right_tokens if right_tokens else 0.0
        shorter = min(left_tokens, right_tokens)
        shorter_ratio = shared_tokens / shorter if shorter else 0.0

        left_canonical = PrefixCanonicalizer.canonicalize(left_segments)
        right_canonical = PrefixCanonicalizer.canonicalize(right_segments)
        shared_names: list[str] = []
        for left_seg, right_seg in zip(left_canonical, right_canonical):
            if (
                left_seg.name != right_seg.name
                or left_seg.priority != right_seg.priority
                or left_seg.content != right_seg.content
            ):
                break
            shared_names.append(left_seg.name)

        if not left and not right:
            reason = "empty"
        elif left == right:
            reason = "identical"
        elif shared_chars == 0:
            reason = "no_shared_prefix"
        elif left.startswith(right) or right.startswith(left):
            reason = "prefix_extension"
        else:
            reason = "diverged"

        first_difference = None
        if left != right:
            first_difference = shared_chars

        return PrefixOverlap(
            left_prefix_id=PrefixCanonicalizer.compute_prefix_id(left_segments),
            right_prefix_id=PrefixCanonicalizer.compute_prefix_id(right_segments),
            shared_prefix_chars=shared_chars,
            shared_prefix_tokens=shared_tokens,
            left_tokens=left_tokens,
            right_tokens=right_tokens,
            left_overlap_ratio=left_ratio,
            right_overlap_ratio=right_ratio,
            shorter_overlap_ratio=shorter_ratio,
            first_difference_index=first_difference,
            shared_segment_names=tuple(shared_names),
            reason=reason,
        )

    @staticmethod
    def common_prefix_chars(left: str, right: str) -> int:
        shared = 0
        for left_char, right_char in zip(left, right):
            if left_char != right_char:
                break
            shared += 1
        return shared

    @staticmethod
    def estimate_tokens(content: str) -> int:
        if not content:
            return 0
        return max(1, len(content) // 4)
