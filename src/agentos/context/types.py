from __future__ import annotations

import time
from collections.abc import Collection
from dataclasses import dataclass, field
from enum import Enum


class ContextScope(Enum):
    SHARED = "shared"
    PRIVATE = "private"
    TASK = "task"
    TOOL_RESULT = "tool_result"


class ContextPermission(Enum):
    READ = "read"
    WRITE = "write"
    SHARE = "share"


DEFAULT_CONTEXT_PRIORITY = 30


class ContextAccessDenied(PermissionError):
    """Raised when an agent reads a ContextRef it has no permission for."""


@dataclass(frozen=True)
class ContextRef:
    name: str
    owner_agent_id: str | None = None
    scope: ContextScope = ContextScope.SHARED
    content_hash: str | None = None
    permissions: frozenset[ContextPermission] = field(
        default_factory=lambda: frozenset(
            {
                ContextPermission.READ,
                ContextPermission.SHARE,
            }
        )
    )
    priority: int = DEFAULT_CONTEXT_PRIORITY

    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "permissions", frozenset(self.permissions))
        if not self.name:
            raise ValueError("ContextRef name cannot be empty")
        if self.name.startswith("ctx://"):
            raise ValueError("ContextRef name must not include ctx:// prefix")

    @property
    def uri(self) -> str:
        return f"ctx://{self.name}"

    @classmethod
    def from_uri(
        cls,
        uri: str,
        owner_agent_id: str | None = None,
        scope: ContextScope = ContextScope.SHARED,
        content_hash: str | None = None,
        permissions: Collection[ContextPermission] | None = None,
    ) -> "ContextRef":
        if not uri.startswith("ctx://"):
            raise ValueError("ContextRef URI must start with ctx://")
        name = uri.removeprefix("ctx://")
        if not name:
            raise ValueError("ContextRef name cannot be empty")
        return cls(
            name=name,
            owner_agent_id=owner_agent_id,
            scope=scope,
            content_hash=content_hash,
            permissions=permissions
            if permissions is not None
            else frozenset(
                {
                    ContextPermission.READ,
                    ContextPermission.SHARE,
                }
            ),
        )

    def can_read(self) -> bool:
        return ContextPermission.READ in self.permissions

    def can_write(self) -> bool:
        return ContextPermission.WRITE in self.permissions

    def can_share(self) -> bool:
        return ContextPermission.SHARE in self.permissions

    def __str__(self) -> str:
        return self.uri
