"""Shared artifact helpers for the P2 sched_ext business benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


BUSINESS_BUNDLE_SCHEMA = "sched_ext_business.v1"
FUNCTIONAL_MODE = "functional"

DEFAULT_FIXTURE_PATHS = (
    "README.md",
    "agentos/isolation/sched_ext_adapter.py",
    "agentos/runtime/engine.py",
    "agentos/scenarios/software_eng/pipeline.py",
    "agentos/scheduler/dag.py",
    "agentos/worker/entrypoint.py",
    "tests/test_runtime_worker.py",
    "tests/test_sched_ext_adapter.py",
)
FROZEN_FIXTURE_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "scenarios/sched_ext_business/fixture_v2_manifest.json"
)
FROZEN_FIXTURE_VERSION = 2


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path.name}")
        values.append(value)
    return values


def seal_directory(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (output_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def verify_seal(output_dir: Path) -> bool:
    seal = output_dir / "SHA256SUMS"
    if not seal.is_file():
        return False
    seen: set[str] = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            return False
        digest, name = parts
        if name in seen or "/" in name or name == "SHA256SUMS":
            return False
        path = output_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
        seen.add(name)
    expected = {path.name for path in output_dir.iterdir() if path.is_file()} - {"SHA256SUMS"}
    return seen == expected


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    fields = ("mode", "metric", "n", "mean", "p50", "p95", "p99", "ci95_low", "ci95_high")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode, mode_values in sorted(summary.items()):
            if not isinstance(mode_values, Mapping):
                continue
            for metric, stats in sorted(mode_values.items()):
                if not isinstance(stats, Mapping) or "n" not in stats:
                    continue
                writer.writerow(
                    {
                        "mode": mode,
                        "metric": metric,
                        **{name: stats.get(name) for name in fields[2:]},
                    }
                )
