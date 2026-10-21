#!/usr/bin/env python3
"""Run Stage 6 incremental regression or assemble the frozen release bundle."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.benchmarks.sched_ext_release import (  # noqa: E402
    OFFICIAL,
    SCHEMA_VERSION,
    aggregate_stage4,
    artifact_manifest,
    benchmark_csv,
    collect_lifecycle_events,
    load_official_evidence,
    read_json,
    render_report,
    sha256_file,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(command: list[str], log: Path) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def child_python() -> str:
    """Keep the active virtualenv executable instead of resolving its symlink."""
    return sys.executable


def _seal(root: Path) -> None:
    paths = [path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in sorted(paths)]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def _snapshot() -> dict[str, Any]:
    sysfs = Path("/sys/kernel/sched_ext")
    return {
        key: (sysfs / key).read_text(encoding="utf-8").strip()
        if (sysfs / key).exists() else None
        for key in ("state", "ops", "enable_seq", "switch_all")
    }


def regress(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    python = child_python()
    scripts = PROJECT_ROOT / "scripts"
    if not args.allow_sysrq:
        raise ValueError("Stage 6 regression requires --allow-sysrq for all 19 cases")
    environment = {
        "schema": SCHEMA_VERSION,
        "kernel_release": platform.release(),
        "python": sys.version,
        "cpu_topology": {"controller": 0, "workers": [1, 2], "pressure": 3},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sysfs_before": _snapshot(),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "scheduler": args.scheduler_binary.resolve(),
                "simple": args.simple_binary.resolve(),
                "worker": args.worker_binary.resolve(),
            }.items()
        },
    }
    _write_json(output / "environment.json", environment)
    try:
        _run(
            [python, str(scripts / "preflight_sched_ext.py"), "--output", str(output / "preflight.json")],
            output / "preflight.log",
        )
        runtime = output / "runtime"
        _run(
            [python, str(scripts / "verify_sched_ext_runtime.py"),
             "--scheduler-binary", str(args.scheduler_binary), "--output-dir", str(runtime),
             "--workers", "10"],
            output / "runtime.console.log",
        )
        effects = output / "policy-effects"
        _run(
            [python, str(scripts / "verify_sched_ext_policy_effects.py"),
             "--scheduler-binary", str(args.scheduler_binary),
             "--simple-binary", str(args.simple_binary),
             "--worker-binary", str(args.worker_binary), "--output-dir", str(effects)],
            output / "policy-effects.console.log",
        )
        ablation = output / "ablation-3x"
        _run(
            [python, str(scripts / "run_sched_ext_stage4_ablation.py"),
             "--scheduler-binary", str(args.scheduler_binary),
             "--simple-binary", str(args.simple_binary),
             "--worker-binary", str(args.worker_binary),
             "--policy-effects-summary", str(effects / "summary.json"),
             "--output-dir", str(ablation), "--iterations", "3", "--warmups", "1",
             "--duration-ms", "12000"],
            output / "ablation.console.log",
        )
        reliability = output / "reliability-1x"
        command = [
            python, str(scripts / "verify_sched_ext_reliability.py"),
            "--scheduler-binary", str(args.scheduler_binary),
            "--output-dir", str(reliability), "--iterations", "1",
        ]
        if args.allow_sysrq:
            command.append("--allow-sysrq")
        _run(command, output / "reliability.console.log")
        summaries = {
            "runtime": read_json(runtime / "summary.json"),
            "policy_effects": read_json(effects / "summary.json"),
            "ablation": read_json(ablation / "summary.json"),
            "reliability": read_json(reliability / "summary.json"),
        }
        accepted = (
            all(summary.get("accepted") is True for summary in summaries.values())
            and summaries["reliability"].get("passed") == 19
            and summaries["reliability"].get("trials") == 19
            and summaries["ablation"].get("iterations") == 3
        )
        result = {"schema": SCHEMA_VERSION, "accepted": accepted, "summaries": summaries}
    except Exception as exc:
        result = {
            "schema": SCHEMA_VERSION, "accepted": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    result["sysfs_after"] = _snapshot()
    _write_json(output / "summary.json", result)
    _seal(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args], text=True, capture_output=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _copy_official(evidence_root: Path, destination: Path) -> None:
    directories: set[str] = set()
    for relative in OFFICIAL.values():
        directories.add(relative.split("/summary.json", 1)[0].split("/result.json", 1)[0].split("/preflight.json", 1)[0])
    for relative in sorted(directories):
        source = evidence_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)


def _pytest_record(log: Path, exit_code: int) -> dict[str, Any]:
    return {
        "log": log.name, "exit_code": exit_code, "sha256": sha256_file(log),
        "passed": exit_code == 0,
    }


def assemble(args: argparse.Namespace) -> int:
    evidence = args.evidence_root.resolve()
    regression = args.regression_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    official = load_official_evidence(evidence)
    benchmark, raw = aggregate_stage4(evidence)
    regression_summary = read_json(regression / "summary.json")
    stage5_dir = evidence / OFFICIAL["stage5_formal"].removesuffix("/summary.json")
    events = collect_lifecycle_events(stage5_dir, evidence)

    _write_json(output / "environment.json", {
        "schema": SCHEMA_VERSION, "assembled_on": platform.platform(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_evidence_root": str(evidence), "regression_root": str(regression),
    })
    _write_json(output / "takeover.json", {
        "stage2": official["stage2"], "stage3": official["stage3"],
    })
    _write_json(output / "fallback.json", official["stage5_formal"])
    _write_json(output / "benchmark_raw.json", {"schema": SCHEMA_VERSION, "trials": raw})
    _write_json(output / "benchmark_summary.json", benchmark)
    (output / "benchmark_summary.csv").write_text(benchmark_csv(benchmark), encoding="utf-8")
    (output / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    logs = output / "logs"
    logs.mkdir()
    debian_log = logs / "debian-pytest.log"
    openeuler_log = logs / "openeuler-pytest.log"
    shutil.copy2(args.debian_pytest_log.resolve(), debian_log)
    shutil.copy2(args.openeuler_pytest_log.resolve(), openeuler_log)
    pytest_results = {
        "debian": _pytest_record(debian_log, args.debian_pytest_exit_code),
        "openeuler": _pytest_record(openeuler_log, args.openeuler_pytest_exit_code),
    }
    _write_json(output / "pytest_results.json", pytest_results)
    report = render_report(official, benchmark)
    (output / "report.md").write_text(report, encoding="utf-8")
    regression_environment = read_json(regression / "environment.json")
    build_manifest = {
        "schema": SCHEMA_VERSION,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "regression_accepted": regression_summary.get("accepted") is True,
        "kernel_release": regression_environment.get("kernel_release"),
        "cpu_topology": regression_environment.get("cpu_topology"),
        "native_artifacts": regression_environment.get("artifacts", {}),
        "pytest": pytest_results,
    }
    _write_json(output / "build_manifest.json", build_manifest)

    defense = output / "defense_bundle"
    evidence_out = defense / "evidence"
    _copy_official(evidence, evidence_out / "stage1-5")
    shutil.copytree(regression, evidence_out / "stage6-regression")
    (defense / "docs").mkdir(parents=True)
    shutil.copy2(output / "report.md", defense / "docs/report.md")
    readme = """# AgentOS sched_ext Stage 6 Defense Bundle

本包证明 Linux sched_ext partial mode 仅接管显式绑定的 Agent Worker，并提供异常回退证据。

允许表述：真实 partial takeover、三级 DSQ、策略消费、自动回退。

禁止表述：替换 Linux 全部调度器、接管 RT、达到生产级。

只读复验：`python scripts/verify_sched_ext_stage6_release.py --bundle <stage6_release> --frozen --check-only --json`
"""
    (defense / "README.md").write_text(readme, encoding="utf-8")
    manifest_paths = [
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS", "strict_verification.json"}
    ]
    _write_json(defense / "manifest.json", {
        "schema": SCHEMA_VERSION, "git": build_manifest["git"],
        "artifact_root": "release",
        "artifacts": artifact_manifest(output, manifest_paths),
    })
    _write_json(output / "strict_verification.json", {
        "passed": False, "reason": "run strict verifier after assembly",
    })
    _seal(output)

    result = {
        "output_dir": str(output), "archive": None,
        "next_step": "run strict verifier to create the frozen archive",
        "schema": SCHEMA_VERSION,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    regress_parser = sub.add_parser("regress")
    regress_parser.add_argument("--scheduler-binary", type=Path, required=True)
    regress_parser.add_argument("--simple-binary", type=Path, required=True)
    regress_parser.add_argument("--worker-binary", type=Path, required=True)
    regress_parser.add_argument("--output-dir", type=Path, required=True)
    regress_parser.add_argument("--allow-sysrq", action="store_true")
    regress_parser.set_defaults(func=regress)
    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--evidence-root", type=Path, required=True)
    assemble_parser.add_argument("--regression-root", type=Path, required=True)
    assemble_parser.add_argument("--debian-pytest-log", type=Path, required=True)
    assemble_parser.add_argument("--debian-pytest-exit-code", type=int, required=True)
    assemble_parser.add_argument("--openeuler-pytest-log", type=Path, required=True)
    assemble_parser.add_argument("--openeuler-pytest-exit-code", type=int, required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    assemble_parser.set_defaults(func=assemble)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
