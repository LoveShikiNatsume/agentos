#!/usr/bin/env python3
"""Run and verify the real AgentOS sched_ext Stage 2 skeleton."""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import json
import os
import platform
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.project_layout import project_file

from scripts.preflight_sched_ext import probe_sched_ext


STAT_NAMES = (
    "select_cpu",
    "direct_local",
    "enqueue",
    "dispatch",
    "running",
    "stopping",
    "unknown_policy",
)
REQUIRED_BATCH_STATS = tuple(name for name in STAT_NAMES if name != "direct_local")
ERROR_PATTERNS = (
    "runnable task stall",
    "watchdog timeout",
    "verification error",
    "verifier error",
    "failed to load bpf",
    "sched_ext: error",
    "sched_ext: bpf error",
    "unexpected watchdog stall",
    "runnable task loss",
    "vruntime overflow",
)


def sched_ext_enabled(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    try:
        lines = (proc_root / str(pid) / "sched").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return None
    for line in lines:
        if line.strip().startswith("ext.enabled"):
            try:
                return int(line.rsplit(":", 1)[1].strip())
            except (IndexError, ValueError):
                return None
    return None


def kernel_log_errors(lines: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(pattern in lowered for pattern in ERROR_PATTERNS):
            errors.append(line.rstrip())
    return errors


def acceptance_passed(summary: dict[str, Any]) -> bool:
    workers = summary.get("workers", {})
    controls = summary.get("controls", {})
    stats = summary.get("stats_delta", {})
    return bool(
        summary.get("preflight_accepted") is True
        and summary.get("scheduler_name") == "agentos_scx"
        and summary.get("partial_mode") is True
        and workers.get("attempted") == 100
        and workers.get("completed") == 100
        and workers.get("failed") == 0
        and workers.get("ext_enabled") == 100
        and workers.get("valid_checksums") == 100
        and summary.get("affinity_violations") == 0
        and controls.get("verifier_ext_enabled") == 0
        and controls.get("loader_ext_enabled") == 0
        and controls.get("unmanaged_ext_enabled") == 0
        and controls.get("cfs_worker_ext_enabled") == 0
        and all(
            type(stats.get(name)) is int and stats[name] > 0
            for name in REQUIRED_BATCH_STATS
        )
        and summary.get("direct_probe_observed") is True
        and summary.get("direct_probe_completed") == summary.get("direct_probe_attempts")
        and summary.get("loader_exit_code") == 0
        and summary.get("sigint_loader_exit_code") == 0
        and type(summary.get("duplicate_loader_exit_code")) is int
        and summary["duplicate_loader_exit_code"] != 0
        and type(summary.get("nonroot_loader_exit_code")) is int
        and summary["nonroot_loader_exit_code"] != 0
        and summary.get("nonroot_guard_verified") is True
        and summary.get("final_sched_ext_state") == "disabled"
        and summary.get("systemd_failed_units") == 0
        and summary.get("kernel_log_errors") == []
        and summary.get("failure_reason") is None
    )


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_json_line(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    if process.stdout is None:
        raise RuntimeError("loader stdout is unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        events = selector.select(timeout)
        if not events:
            raise TimeoutError("timed out waiting for loader JSON")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        raise RuntimeError(f"loader exited before ready: rc={process.poll()}")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid loader JSON: {line!r}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("loader event must be a JSON object")
    return payload


def _start_loader(binary: Path) -> tuple[subprocess.Popen[str], list[dict[str, Any]]]:
    process = subprocess.Popen(
        [str(binary), "--stats-interval-ms", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        events = [_read_json_line(process, 10)]
        if events[0].get("event") != "ready":
            raise RuntimeError(f"loader did not become ready: {events[0]}")
        events.append(_read_json_line(process, 3))
        if events[1].get("event") != "stats":
            raise RuntimeError(f"loader did not emit initial stats: {events[1]}")
        return process, events
    except Exception:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=2)
        raise


def _stop_loader(
    process: subprocess.Popen[str],
    events: list[dict[str, Any]],
    signo: signal.Signals,
) -> tuple[int, str, str]:
    process.send_signal(signo)
    stdout, stderr = process.communicate(timeout=10)
    for line in stdout.splitlines():
        if line.strip():
            events.append(json.loads(line))
    return process.returncode, stdout, stderr


def _worker_result(binary: Path, duration_ms: int, cpu: int | None) -> dict[str, Any]:
    command = [str(binary), "--duration-ms", str(duration_ms)]
    if cpu is not None:
        command.extend(["--cpu", str(cpu)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"completed": False, "parse_error": completed.stdout}
    payload["process_return_code"] = completed.returncode
    payload["stderr"] = completed.stderr
    return payload


def _nonroot_identity(environ: Mapping[str, str] | None = None) -> tuple[int, int]:
    environ = os.environ if environ is None else environ
    try:
        uid = int(environ.get("SUDO_UID", "65534"))
        gid = int(environ.get("SUDO_GID", "65534"))
    except ValueError:
        return 65534, 65534
    if uid <= 0 or gid < 0:
        return 65534, 65534
    return uid, gid


def _drop_root_privileges(uid: int, gid: int) -> None:
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)


def _snapshot() -> dict[str, Any]:
    return {
        "state": _read(Path("/sys/kernel/sched_ext/state")),
        "enable_seq": _read(Path("/sys/kernel/sched_ext/enable_seq")),
        "switch_all": _read(Path("/sys/kernel/sched_ext/switch_all")),
        "ops": _read(Path("/sys/kernel/sched_ext/root/ops")),
    }


def _stats(events: list[dict[str, Any]], event_name: str, *, last: bool) -> dict[str, int]:
    selected = [event for event in events if event.get("event") == event_name]
    if not selected:
        return {}
    event = selected[-1] if last else selected[0]
    stats = event.get("stats", {})
    return {name: int(stats.get(name, 0)) for name in STAT_NAMES}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for payload in payloads:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _hash_artifacts(output_dir: Path, paths: Iterable[Path]) -> None:
    hashes: dict[str, str] = {}
    for path in paths:
        try:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    _write_json(output_dir / "artifact-sha256.json", hashes)


def _seal(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.name}\n")
    (output_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def run_verification(
    *,
    scheduler_binary: Path,
    worker_binary: Path,
    output_dir: Path,
    unmanaged_pid: int,
    workers: int = 100,
    concurrency: int = 10,
    duration_ms: int = 100,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "accepted": False,
        "kernel_release": platform.release(),
        "scheduler_name": None,
        "partial_mode": None,
        "workers": {
            "attempted": workers,
            "completed": 0,
            "failed": workers,
            "ext_enabled": 0,
            "valid_checksums": 0,
        },
        "controls": {},
        "affinity_violations": None,
        "stats_before": {},
        "stats_after": {},
        "stats_delta": {},
        "direct_probe_attempts": 0,
        "direct_probe_completed": 0,
        "direct_probe_observed": False,
        "direct_probe_stats": {},
        "loader_exit_code": None,
        "sigint_loader_exit_code": None,
        "duplicate_loader_exit_code": None,
        "nonroot_loader_exit_code": None,
        "nonroot_guard_verified": False,
        "final_sched_ext_state": None,
        "systemd_failed_units": None,
        "kernel_log_errors": [],
        "failure_reason": None,
    }
    loader: subprocess.Popen[str] | None = None
    control: subprocess.Popen[str] | None = None
    loader_events: list[dict[str, Any]] = []
    worker_results: list[dict[str, Any]] = []
    started_at = int(time.time())
    loader_stderr = ""

    try:
        if os.geteuid() != 0:
            raise PermissionError("real sched_ext verification must run as root")
        preflight = probe_sched_ext()
        _write_json(output_dir / "preflight.json", preflight)
        summary["preflight_accepted"] = preflight.get("accepted") is True
        if not summary["preflight_accepted"]:
            raise RuntimeError(f"preflight rejected host: {preflight.get('failures')}")
        if workers != 100:
            raise ValueError("Stage 2 strict verification requires exactly 100 workers")
        if concurrency <= 0 or concurrency > workers:
            raise ValueError("invalid worker concurrency")
        if not scheduler_binary.is_file() or not worker_binary.is_file():
            raise FileNotFoundError("scheduler or worker binary is missing")

        nonroot_uid, nonroot_gid = _nonroot_identity()
        nonroot = subprocess.run(
            [str(scheduler_binary), "--quiet"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            preexec_fn=functools.partial(
                _drop_root_privileges, nonroot_uid, nonroot_gid
            ),
        )
        summary["nonroot_loader_exit_code"] = nonroot.returncode
        summary["nonroot_guard_verified"] = (
            nonroot.returncode != 0 and "must run as root" in nonroot.stdout
        )
        (output_dir / "nonroot-loader.log").write_text(
            nonroot.stdout + nonroot.stderr, encoding="utf-8"
        )

        _write_json(output_dir / "sysfs-before.json", _snapshot())
        loader, loader_events = _start_loader(scheduler_binary)
        during = _snapshot()
        _write_json(output_dir / "sysfs-enabled.json", during)
        summary["scheduler_name"] = during.get("ops")
        summary["partial_mode"] = during.get("switch_all") == "0"
        summary["controls"] = {
            "verifier_ext_enabled": sched_ext_enabled(os.getpid()),
            "loader_ext_enabled": sched_ext_enabled(loader.pid),
            "unmanaged_ext_enabled": sched_ext_enabled(unmanaged_pid),
        }

        duplicate = subprocess.run(
            [str(scheduler_binary), "--quiet"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        summary["duplicate_loader_exit_code"] = duplicate.returncode
        (output_dir / "duplicate-loader.log").write_text(
            duplicate.stdout + duplicate.stderr, encoding="utf-8"
        )

        control = subprocess.Popen(
            [sys.executable, "-c", "while True: pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.05)
        summary["controls"]["cfs_worker_ext_enabled"] = sched_ext_enabled(control.pid)

        affinity_cpu = min(os.sched_getaffinity(0))
        cpu_args: list[int | None] = [None] * workers
        cpu_args[-1] = affinity_cpu
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_worker_result, worker_binary, duration_ms, cpu)
                for cpu in cpu_args
            ]
            worker_results = [future.result() for future in futures]

        completed = sum(
            result.get("completed") is True and result.get("process_return_code") == 0
            for result in worker_results
        )
        ext_enabled = sum(result.get("ext_enabled_after") == 1 for result in worker_results)
        valid_checksums = sum(
            type(result.get("checksum")) is int and result["checksum"] > 0
            for result in worker_results
        )
        affinity_violations = sum(
            int(result.get("affinity_violations", 1)) for result in worker_results
        )
        summary["workers"] = {
            "attempted": workers,
            "completed": completed,
            "failed": workers - completed,
            "ext_enabled": ext_enabled,
            "valid_checksums": valid_checksums,
        }
        summary["affinity_violations"] = affinity_violations
        _write_jsonl(output_dir / "workers.jsonl", worker_results)

        time.sleep(0.2)
        summary["stats_before"] = _stats(loader_events, "stats", last=False)
        loader_rc, remaining_stdout, loader_stderr = _stop_loader(
            loader, loader_events, signal.SIGTERM
        )
        loader = None
        summary["loader_exit_code"] = loader_rc
        summary["stats_after"] = _stats(loader_events, "final_stats", last=True)
        summary["stats_delta"] = {
            name: summary["stats_after"].get(name, 0)
            - summary["stats_before"].get(name, 0)
            for name in STAT_NAMES
        }
        _write_jsonl(output_dir / "loader-events.jsonl", loader_events)
        (output_dir / "loader-remaining-stdout.log").write_text(
            remaining_stdout, encoding="utf-8"
        )
        (output_dir / "loader-stderr.log").write_text(loader_stderr, encoding="utf-8")

        if control is not None and control.poll() is None:
            control.terminate()
            try:
                control.wait(timeout=2)
            except subprocess.TimeoutExpired:
                control.kill()
                control.wait(timeout=2)
        control = None

        sigint_loader, sigint_events = _start_loader(scheduler_binary)
        time.sleep(0.25)
        direct_probe_results = []
        for _ in range(4):
            direct_probe_results.append(
                _worker_result(worker_binary, duration_ms, None)
            )
            time.sleep(0.1)
        sigint_rc, sigint_stdout, sigint_stderr = _stop_loader(
            sigint_loader, sigint_events, signal.SIGINT
        )
        summary["sigint_loader_exit_code"] = sigint_rc
        summary["direct_probe_attempts"] = len(direct_probe_results)
        summary["direct_probe_completed"] = sum(
            result.get("completed") is True
            and result.get("process_return_code") == 0
            for result in direct_probe_results
        )
        summary["direct_probe_stats"] = _stats(
            sigint_events, "final_stats", last=True
        )
        summary["direct_probe_observed"] = (
            summary["direct_probe_stats"].get("direct_local", 0) > 0
        )
        _write_jsonl(output_dir / "direct-probe-workers.jsonl", direct_probe_results)
        _write_jsonl(output_dir / "sigint-loader-events.jsonl", sigint_events)
        (output_dir / "sigint-loader-stdout.log").write_text(
            sigint_stdout, encoding="utf-8"
        )
        (output_dir / "sigint-loader-stderr.log").write_text(
            sigint_stderr, encoding="utf-8"
        )

        final_snapshot = _snapshot()
        _write_json(output_dir / "sysfs-after.json", final_snapshot)
        summary["final_sched_ext_state"] = final_snapshot.get("state")

        systemd = subprocess.run(
            ["systemctl", "--failed", "--no-legend", "--plain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        failed_units = [line for line in systemd.stdout.splitlines() if line.strip()]
        summary["systemd_failed_units"] = len(failed_units)
        (output_dir / "systemd-failed.log").write_text(
            systemd.stdout + systemd.stderr, encoding="utf-8"
        )

        journal = subprocess.run(
            [
                "journalctl",
                "-k",
                "-b",
                f"--since=@{started_at}",
                "--no-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        (output_dir / "kernel-window.log").write_text(
            journal.stdout + journal.stderr, encoding="utf-8"
        )
        if journal.returncode:
            raise RuntimeError(f"journalctl failed with code {journal.returncode}")
        summary["kernel_log_errors"] = kernel_log_errors(journal.stdout.splitlines())
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if control is not None and control.poll() is None:
            control.terminate()
            try:
                control.wait(timeout=2)
            except subprocess.TimeoutExpired:
                control.kill()
                control.wait(timeout=2)
        if loader is not None and loader.poll() is None:
            loader.terminate()
            try:
                stdout, stderr = loader.communicate(timeout=5)
                (output_dir / "loader-cleanup.log").write_text(
                    stdout + stderr, encoding="utf-8"
                )
            except subprocess.TimeoutExpired:
                loader.kill()
                loader.communicate(timeout=2)
        if summary["final_sched_ext_state"] is None:
            summary["final_sched_ext_state"] = _read(
                Path("/sys/kernel/sched_ext/state")
            )

    summary["accepted"] = acceptance_passed(summary)
    if not summary["accepted"] and summary["failure_reason"] is None:
        non_growing = [
            name
            for name in REQUIRED_BATCH_STATS
            if summary["stats_delta"].get(name, 0) <= 0
        ]
        detail = f"; non-growing stats: {','.join(non_growing)}" if non_growing else ""
        if not summary["direct_probe_observed"]:
            detail += "; direct-local probe not observed"
        summary["failure_reason"] = f"acceptance criteria not met{detail}"
    _write_json(output_dir / "summary.json", summary)
    _hash_artifacts(
        output_dir,
        (
            scheduler_binary,
            worker_binary,
            PROJECT_ROOT / "native/sched_ext/build/obj/agentos_scx.bpf.o",
            PROJECT_ROOT / "native/sched_ext/build/include/agentos_scx.bpf.skel.h",
            PROJECT_ROOT / "native/sched_ext/agentos_scx.bpf.c",
            PROJECT_ROOT / "native/sched_ext/agentos_scx.c",
            PROJECT_ROOT / "native/sched_ext/agentos_scx_worker.c",
            PROJECT_ROOT / "native/sched_ext/include/agentos_scx.h",
            PROJECT_ROOT / "native/sched_ext/Makefile",
            PROJECT_ROOT / "native/sched_ext/README.md",
            PROJECT_ROOT / "scripts/verify_real_sched_ext.py",
            project_file(PROJECT_ROOT, "tests/test_verify_real_sched_ext.py"),
        ),
    )
    _seal(output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-bin",
        type=Path,
        default=PROJECT_ROOT / "native/sched_ext/build/bin/agentos_scx",
    )
    parser.add_argument(
        "--worker-bin",
        type=Path,
        default=PROJECT_ROOT / "native/sched_ext/build/bin/agentos_scx_worker",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs/member_a_sched_ext_validation/stage2_skeleton",
    )
    parser.add_argument("--unmanaged-pid", type=int, required=True)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--duration-ms", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_verification(
        scheduler_binary=args.scheduler_bin,
        worker_binary=args.worker_bin,
        output_dir=args.output_dir,
        unmanaged_pid=args.unmanaged_pid,
        workers=args.workers,
        concurrency=args.concurrency,
        duration_ms=args.duration_ms,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
