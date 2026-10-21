from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ContextDependency:
    ref_name: str
    read_files: frozenset[str] = field(default_factory=frozenset)
    write_files: frozenset[str] = field(default_factory=frozenset)
    symbols: frozenset[str] = field(default_factory=frozenset)
    producer_task_id: str | None = None
    consumer_task_ids: frozenset[str] = field(default_factory=frozenset)
    role: str = "context"
    weight: float = 1.0

    @classmethod
    def build(
        cls,
        ref_name: str,
        *,
        read_files: Iterable[str] = (),
        write_files: Iterable[str] = (),
        symbols: Iterable[str] = (),
        producer_task_id: str | None = None,
        consumer_task_ids: Iterable[str] = (),
        role: str = "context",
        weight: float = 1.0,
    ) -> "ContextDependency":
        return cls(
            ref_name=ref_name,
            read_files=frozenset(read_files),
            write_files=frozenset(write_files),
            symbols=frozenset(symbols),
            producer_task_id=producer_task_id,
            consumer_task_ids=frozenset(consumer_task_ids),
            role=role,
            weight=weight,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["read_files"] = sorted(self.read_files)
        data["write_files"] = sorted(self.write_files)
        data["symbols"] = sorted(self.symbols)
        data["consumer_task_ids"] = sorted(self.consumer_task_ids)
        return data


@dataclass(frozen=True)
class DependencyQuery:
    task_id: str | None = None
    read_files: frozenset[str] = field(default_factory=frozenset)
    write_files: frozenset[str] = field(default_factory=frozenset)
    symbols: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def build(
        cls,
        *,
        task_id: str | None = None,
        read_files: Iterable[str] = (),
        write_files: Iterable[str] = (),
        symbols: Iterable[str] = (),
    ) -> "DependencyQuery":
        return cls(
            task_id=task_id,
            read_files=frozenset(read_files),
            write_files=frozenset(write_files),
            symbols=frozenset(symbols),
        )


@dataclass
class DependencyMatch:
    ref_name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    dependency: ContextDependency | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_name": self.ref_name,
            "score": self.score,
            "reasons": list(self.reasons),
            "dependency": (
                self.dependency.to_dict()
                if self.dependency is not None
                else None
            ),
        }


class ContextDependencyGraph:
    """Indexes ContextRefs by coding-task dependencies."""

    def __init__(self) -> None:
        self._deps: dict[str, ContextDependency] = {}
        self._file_index: dict[str, set[str]] = {}
        self._symbol_index: dict[str, set[str]] = {}
        self._task_index: dict[str, set[str]] = {}

    def add(self, dependency: ContextDependency) -> None:
        self._deps[dependency.ref_name] = dependency
        for file_name in dependency.read_files | dependency.write_files:
            self._file_index.setdefault(file_name, set()).add(dependency.ref_name)
        for symbol in dependency.symbols:
            self._symbol_index.setdefault(symbol, set()).add(dependency.ref_name)
        if dependency.producer_task_id:
            self._task_index.setdefault(dependency.producer_task_id, set()).add(
                dependency.ref_name
            )
        for task_id in dependency.consumer_task_ids:
            self._task_index.setdefault(task_id, set()).add(dependency.ref_name)

    def add_ref(
        self,
        ref_name: str,
        *,
        read_files: Iterable[str] = (),
        write_files: Iterable[str] = (),
        symbols: Iterable[str] = (),
        producer_task_id: str | None = None,
        consumer_task_ids: Iterable[str] = (),
        role: str = "context",
        weight: float = 1.0,
    ) -> ContextDependency:
        dep = ContextDependency.build(
            ref_name,
            read_files=read_files,
            write_files=write_files,
            symbols=symbols,
            producer_task_id=producer_task_id,
            consumer_task_ids=consumer_task_ids,
            role=role,
            weight=weight,
        )
        self.add(dep)
        return dep

    def get(self, ref_name: str) -> ContextDependency | None:
        return self._deps.get(ref_name)

    def match(self, query: DependencyQuery) -> list[DependencyMatch]:
        candidates: set[str] = set()
        if query.task_id:
            candidates.update(self._task_index.get(query.task_id, set()))
        for file_name in query.read_files | query.write_files:
            candidates.update(self._file_index.get(file_name, set()))
        for symbol in query.symbols:
            candidates.update(self._symbol_index.get(symbol, set()))

        matches: list[DependencyMatch] = []
        for ref_name in candidates:
            dep = self._deps[ref_name]
            score, reasons = self._score(dep, query)
            if score > 0:
                matches.append(
                    DependencyMatch(
                        ref_name=ref_name,
                        score=score,
                        reasons=reasons,
                        dependency=dep,
                    )
                )
        matches.sort(key=lambda match: (-match.score, match.ref_name))
        return matches

    def relevant_refs(
        self,
        query: DependencyQuery,
        *,
        limit: int | None = None,
    ) -> list[str]:
        matches = self.match(query)
        if limit is not None:
            matches = matches[:limit]
        return [match.ref_name for match in matches]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependencies": {
                ref_name: dep.to_dict()
                for ref_name, dep in sorted(self._deps.items())
            },
            "files": {
                file_name: sorted(refs)
                for file_name, refs in sorted(self._file_index.items())
            },
            "symbols": {
                symbol: sorted(refs)
                for symbol, refs in sorted(self._symbol_index.items())
            },
            "tasks": {
                task_id: sorted(refs)
                for task_id, refs in sorted(self._task_index.items())
            },
        }

    @staticmethod
    def _score(
        dep: ContextDependency,
        query: DependencyQuery,
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []

        if query.task_id and (
            query.task_id == dep.producer_task_id
            or query.task_id in dep.consumer_task_ids
        ):
            score += 3.0
            reasons.append("task")

        read_overlap = query.read_files & (dep.read_files | dep.write_files)
        if read_overlap:
            score += 2.0 * len(read_overlap)
            reasons.append("read_file")

        write_overlap = query.write_files & (dep.read_files | dep.write_files)
        if write_overlap:
            score += 2.5 * len(write_overlap)
            reasons.append("write_file")

        symbol_overlap = query.symbols & dep.symbols
        if symbol_overlap:
            score += 1.5 * len(symbol_overlap)
            reasons.append("symbol")

        return score * dep.weight, reasons
