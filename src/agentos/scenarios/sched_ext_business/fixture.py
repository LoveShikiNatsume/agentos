"""Freeze and verify deterministic repository inputs for the business demo."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from agentos.project_layout import project_file, project_scope


FIXTURE_SCHEMA = "sched_ext_business_fixture.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _normalise_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts:
        raise ValueError(f"fixture path must be relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"fixture path escapes root: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class FixtureFile:
    path: str
    size: int
    sha256: str
    kind: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FixtureFile":
        path = _normalise_relative_path(str(value.get("path", "")))
        size = int(value.get("size", -1))
        sha256 = str(value.get("sha256", ""))
        kind = str(value.get("kind", "other"))
        if size < 0:
            raise ValueError(f"invalid fixture size for {path}")
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError(f"invalid fixture sha256 for {path}")
        return cls(path=path, size=size, sha256=sha256, kind=kind)


@dataclass(frozen=True)
class FixtureManifest:
    schema: str
    files: tuple[FixtureFile, ...]
    aggregate_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "files": [asdict(item) for item in self.files],
            "aggregate_sha256": self.aggregate_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FixtureManifest":
        schema = str(value.get("schema", ""))
        if schema != FIXTURE_SCHEMA:
            raise ValueError(f"unsupported fixture schema: {schema!r}")
        raw_files = value.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("fixture manifest must contain files")
        files = tuple(FixtureFile.from_dict(item) for item in raw_files)
        if len({item.path for item in files}) != len(files):
            raise ValueError("fixture manifest contains duplicate paths")
        if tuple(sorted(item.path for item in files)) != tuple(item.path for item in files):
            raise ValueError("fixture manifest files must be sorted")
        aggregate = str(value.get("aggregate_sha256", ""))
        expected = _aggregate_sha256(files)
        if aggregate != expected:
            raise ValueError("fixture manifest aggregate mismatch")
        return cls(schema=schema, files=files, aggregate_sha256=aggregate)


def _kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return "other"


def _aggregate_sha256(files: Iterable[FixtureFile]) -> str:
    records = [asdict(item) for item in files]
    return _sha256_bytes(_canonical_json(records))


def _checked_file(root: Path, relative_path: str) -> Path:
    relative = _normalise_relative_path(relative_path)
    root_resolved = root.resolve(strict=True)
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"fixture path must not be a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"fixture path escapes root: {relative}")
    if not resolved.is_file():
        raise ValueError(f"fixture path is not a regular file: {relative}")
    return resolved


def freeze_fixture(
    source_root: Path,
    fixture_root: Path,
    relative_paths: Iterable[str],
) -> FixtureManifest:
    """Copy selected real files into a new immutable-by-convention fixture."""

    source_root = Path(source_root)
    fixture_root = Path(fixture_root)
    paths = sorted({_normalise_relative_path(value) for value in relative_paths})
    if not paths:
        raise ValueError("at least one fixture path is required")
    if fixture_root.exists():
        raise FileExistsError(f"fixture root already exists: {fixture_root}")
    fixture_root.mkdir(parents=True)

    records: list[FixtureFile] = []
    for relative in paths:
        scope = project_scope(source_root)
        mapped = project_file(source_root, relative)
        source = _checked_file(scope, mapped.relative_to(scope).as_posix())
        destination = fixture_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        value = destination.read_bytes()
        records.append(
            FixtureFile(
                path=relative,
                size=len(value),
                sha256=_sha256_bytes(value),
                kind=_kind(relative),
            )
        )

    files = tuple(records)
    manifest = FixtureManifest(
        schema=FIXTURE_SCHEMA,
        files=files,
        aggregate_sha256=_aggregate_sha256(files),
    )
    (fixture_root / "fixture_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_fixture(
    fixture_root: Path,
    manifest: FixtureManifest | Mapping[str, Any],
) -> dict[str, bytes]:
    """Verify every frozen byte before a worker performs business analysis."""

    parsed = (
        manifest
        if isinstance(manifest, FixtureManifest)
        else FixtureManifest.from_dict(manifest)
    )
    root = Path(fixture_root)
    values: dict[str, bytes] = {}
    for item in parsed.files:
        path = _checked_file(root, item.path)
        value = path.read_bytes()
        if len(value) != item.size or _sha256_bytes(value) != item.sha256:
            raise ValueError(f"fixture hash mismatch: {item.path}")
        values[item.path] = value
    return values
