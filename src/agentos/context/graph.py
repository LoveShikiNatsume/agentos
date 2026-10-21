from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from agentos.context.prefix import PrefixCanonicalizer, PrefixSegment


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _hash_parts(parts: Iterable[str]) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ContextSegment:
    """Content-addressed unit in a prompt layout path."""

    name: str
    content_hash: str
    tokens: int
    priority: int
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prefix_segment(cls, segment: PrefixSegment) -> "ContextSegment":
        return cls(
            name=segment.name,
            content_hash=_hash_text(segment.content),
            tokens=PrefixCanonicalizer.estimate_tokens(segment.content),
            priority=segment.priority,
            ordinal=segment.ordinal,
        )

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.priority}:{self.ordinal}:{self.content_hash}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrefixDivergence:
    left_layout_id: str
    right_layout_id: str
    shared_segment_names: tuple[str, ...]
    shared_tokens: int
    first_divergence_index: int | None
    left_only_segment_names: tuple[str, ...]
    right_only_segment_names: tuple[str, ...]
    reason: str

    @property
    def shared_segment_count(self) -> int:
        return len(self.shared_segment_names)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextLayout:
    """Merkle-style path for one agent's shared context prefix."""

    segments: tuple[ContextSegment, ...]
    layout_id: str
    cumulative_hashes: tuple[str, ...]

    @classmethod
    def from_prefix_segments(
        cls,
        segments: list[PrefixSegment],
    ) -> "ContextLayout":
        canonical = PrefixCanonicalizer.canonicalize(segments)
        graph_segments = tuple(
            ContextSegment.from_prefix_segment(seg) for seg in canonical
        )
        cumulative: list[str] = []
        current = ""
        for seg in graph_segments:
            current = _hash_parts((current, seg.identity))
            cumulative.append(current)
        layout_id = cumulative[-1] if cumulative else "__empty_layout__"
        return cls(
            segments=graph_segments,
            layout_id=layout_id,
            cumulative_hashes=tuple(cumulative),
        )

    @property
    def total_tokens(self) -> int:
        return sum(seg.tokens for seg in self.segments)

    @property
    def segment_names(self) -> tuple[str, ...]:
        return tuple(seg.name for seg in self.segments)

    def divergence(self, other: "ContextLayout") -> PrefixDivergence:
        shared_count = 0
        for left, right in zip(self.segments, other.segments):
            if left.identity != right.identity:
                break
            shared_count += 1

        shared_segments = self.segments[:shared_count]
        if self.layout_id == other.layout_id:
            first_divergence = None
            reason = "identical"
        elif shared_count == 0:
            first_divergence = 0
            reason = "no_shared_prefix"
        elif shared_count == min(len(self.segments), len(other.segments)):
            first_divergence = shared_count
            reason = "prefix_extension"
        else:
            first_divergence = shared_count
            reason = "diverged"

        return PrefixDivergence(
            left_layout_id=self.layout_id,
            right_layout_id=other.layout_id,
            shared_segment_names=tuple(seg.name for seg in shared_segments),
            shared_tokens=sum(seg.tokens for seg in shared_segments),
            first_divergence_index=first_divergence,
            left_only_segment_names=tuple(
                seg.name for seg in self.segments[shared_count:]
            ),
            right_only_segment_names=tuple(
                seg.name for seg in other.segments[shared_count:]
            ),
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "total_tokens": self.total_tokens,
            "segment_names": list(self.segment_names),
            "cumulative_hashes": list(self.cumulative_hashes),
            "segments": [seg.to_dict() for seg in self.segments],
        }


class PrefixGraph:
    """Tracks prompt-layout paths and computes shared-prefix structure."""

    def __init__(self) -> None:
        self._layouts: dict[str, ContextLayout] = {}

    def add_layout(self, key: str, layout: ContextLayout) -> None:
        self._layouts[key] = layout

    def get_layout(self, key: str) -> ContextLayout | None:
        return self._layouts.get(key)

    def divergence(self, left_key: str, right_key: str) -> PrefixDivergence:
        left = self._layouts[left_key]
        right = self._layouts[right_key]
        return left.divergence(right)

    def shared_prefix_matrix(self) -> dict[str, dict[str, Any]]:
        keys = sorted(self._layouts)
        matrix: dict[str, dict[str, Any]] = {}
        for i, left_key in enumerate(keys):
            for right_key in keys[i + 1:]:
                matrix[f"{left_key}->{right_key}"] = self.divergence(
                    left_key,
                    right_key,
                ).to_dict()
        return matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "layouts": {
                key: layout.to_dict()
                for key, layout in sorted(self._layouts.items())
            },
            "shared_prefix_matrix": self.shared_prefix_matrix(),
        }
