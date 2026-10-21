"""Deterministic CPU-bound repository analysis workers for the P2 demo."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from agentos.scenarios.sched_ext_business.fixture import (
    FixtureManifest,
    verify_fixture,
)


BUSINESS_WORKER_TYPES = frozenset(
    {
        "artifact_hash",
        "dependency_analyze",
        "docs_scan",
        "patch_validate",
        "repo_index",
        "security_scan",
        "symbol_index",
        "test_select",
    }
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _text_files(files: Mapping[str, bytes]) -> dict[str, str]:
    return {
        path: value.decode("utf-8", errors="strict")
        for path, value in files.items()
    }


def _python_sources(files: Mapping[str, bytes]) -> dict[str, str]:
    return {
        path: value.decode("utf-8", errors="strict")
        for path, value in files.items()
        if path.endswith(".py")
    }


def _patch_validate(files: Mapping[str, bytes]) -> dict[str, object]:
    nodes = 0
    invalid: list[str] = []
    for path, source in _python_sources(files).items():
        try:
            tree = ast.parse(source, filename=path)
            compile(source, path, "exec", dont_inherit=True)
            nodes += sum(1 for _node in ast.walk(tree))
        except (SyntaxError, ValueError, TypeError):
            invalid.append(path)
    return {
        "python_files": len(_python_sources(files)),
        "ast_nodes": nodes,
        "invalid_files": sorted(invalid),
        "passed": not invalid,
    }


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        parts = [target.attr]
        value = target.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _security_scan(files: Mapping[str, bytes]) -> dict[str, object]:
    risky = {"eval", "exec", "os.system", "pickle.loads", "yaml.load"}
    findings: list[dict[str, object]] = []
    calls = 0
    for path, source in _python_sources(files).items():
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            calls += 1
            name = _call_name(node)
            shell_true = name.startswith("subprocess.") and any(
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            if name in risky or shell_true:
                findings.append(
                    {"path": path, "line": node.lineno, "rule": name or "shell=True"}
                )
    return {"calls_scanned": calls, "findings": findings, "finding_count": len(findings)}


def _test_select(files: Mapping[str, bytes]) -> dict[str, object]:
    python_paths = sorted(path for path in files if path.endswith(".py"))
    tests = sorted(path for path in python_paths if Path(path).name.startswith("test_"))
    modules = [path for path in python_paths if path not in tests]
    selected: dict[str, list[str]] = {}
    for module in modules:
        stem = Path(module).stem
        matches = [path for path in tests if stem in Path(path).stem]
        selected[module] = matches
    return {
        "module_count": len(modules),
        "test_count": len(tests),
        "selection": selected,
    }


def _dependency_analyze(files: Mapping[str, bytes]) -> dict[str, object]:
    edges: set[tuple[str, str]] = set()
    for path, source in _python_sources(files).items():
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                edges.update((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                edges.add((path, module))
    return {
        "python_files": len(_python_sources(files)),
        "edge_count": len(edges),
        "edges": [list(edge) for edge in sorted(edges)],
    }


def _repo_index(files: Mapping[str, bytes]) -> dict[str, object]:
    by_kind: dict[str, int] = {}
    total_lines = 0
    for path, value in files.items():
        suffix = Path(path).suffix.lower() or "<none>"
        by_kind[suffix] = by_kind.get(suffix, 0) + 1
        total_lines += len(value.splitlines())
    return {
        "file_count": len(files),
        "byte_count": sum(len(value) for value in files.values()),
        "line_count": total_lines,
        "files_by_suffix": dict(sorted(by_kind.items())),
    }


def _artifact_hash(files: Mapping[str, bytes]) -> dict[str, object]:
    digests = {
        path: hashlib.sha256(value).hexdigest()
        for path, value in sorted(files.items())
    }
    aggregate = hashlib.sha256(_canonical(digests)).hexdigest()
    return {"file_count": len(digests), "digests": digests, "aggregate": aggregate}


_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _docs_scan(files: Mapping[str, bytes]) -> dict[str, object]:
    markdown = {
        path: text
        for path, text in _text_files(files).items()
        if Path(path).suffix.lower() in {".md", ".markdown"}
    }
    known = set(files)
    links: list[dict[str, object]] = []
    for path, text in markdown.items():
        for target in _MARKDOWN_LINK.findall(text):
            if target.startswith(("http://", "https://", "#")):
                valid = True
            else:
                clean = target.split("#", 1)[0]
                resolved = (Path(path).parent / clean).as_posix()
                valid = resolved in known
            links.append({"source": path, "target": target, "valid": valid})
    return {
        "markdown_files": len(markdown),
        "link_count": len(links),
        "broken_count": sum(not item["valid"] for item in links),
        "links": links,
    }


def _symbol_index(files: Mapping[str, bytes]) -> dict[str, object]:
    symbols: list[dict[str, object]] = []
    for path, source in _python_sources(files).items():
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif isinstance(node, ast.AsyncFunctionDef):
                kind = "async_function"
            elif isinstance(node, ast.FunctionDef):
                kind = "function"
            else:
                continue
            symbols.append({"path": path, "line": node.lineno, "name": node.name, "kind": kind})
    symbols.sort(key=lambda item: (str(item["path"]), int(item["line"]), str(item["name"])))
    return {"symbol_count": len(symbols), "symbols": symbols}


_WORKERS: dict[str, Callable[[Mapping[str, bytes]], dict[str, object]]] = {
    "artifact_hash": _artifact_hash,
    "dependency_analyze": _dependency_analyze,
    "docs_scan": _docs_scan,
    "patch_validate": _patch_validate,
    "repo_index": _repo_index,
    "security_scan": _security_scan,
    "symbol_index": _symbol_index,
    "test_select": _test_select,
}


def run_business_worker(worker_type: str, payload: Mapping[str, Any]) -> dict[str, object]:
    """Run one deterministic business operation over a verified fixture."""

    if worker_type not in _WORKERS:
        raise ValueError(f"unsupported business worker_type: {worker_type}")
    fixture_root = Path(str(payload.get("fixture_root", "")))
    raw_manifest = payload.get("manifest")
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("business worker payload requires manifest")
    manifest = FixtureManifest.from_dict(raw_manifest)
    try:
        work_units = int(payload.get("work_units", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("work_units must be an integer") from exc
    if not 1 <= work_units <= 10_000:
        raise ValueError("work_units must be between 1 and 10000")
    files = verify_fixture(fixture_root, manifest)

    operation = _WORKERS[worker_type]
    result: dict[str, object] = {}
    workload_digest = hashlib.sha256()
    for unit in range(work_units):
        result = operation(files)
        workload_digest.update(unit.to_bytes(8, "little"))
        workload_digest.update(_canonical(result))

    checksum_payload = {
        "worker_type": worker_type,
        "input_sha256": manifest.aggregate_sha256,
        "work_units": work_units,
        "result": result,
    }
    return {
        **checksum_payload,
        "workload_checksum": workload_digest.hexdigest(),
        "checksum": hashlib.sha256(_canonical(checksum_payload)).hexdigest(),
    }
