"""Trusted, single-instance SWE-bench Docker evaluation of archived run artifacts.

No model entry point is used here. Official data and test patches stay outside
candidate workspaces. Existing run records are read-only inputs.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agentos.execution.records import inventory

DEFAULTS = {"type": "swebench", "split": "test", "timeout": 1800, "wall_timeout": 3600,
            "max_workers": 1, "namespace": "swebench", "instance_image_tag": "latest",
            "env_image_tag": "latest", "cache_level": "instance"}
FIELDS = set(DEFAULTS) | {"python", "dataset_name", "instance_id", "base_commit", "base_repo", "repo"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def trusted_env(excluded_names=()):
    # Evaluation uses the chosen official environment, never the candidate's
    # Python modules or the model service credential environment.
    return {k: v for k, v in os.environ.items()
            if k not in {"PYTHONPATH", "PYTHONHOME", *excluded_names}
            and not any(part in k.upper() for part in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))}


def git(root: Path, *args, input_data=None) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], input=input_data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(f"git {args[0]} failed: {result.stderr.decode(errors='replace')[:600]}")
    return result.stdout


def safe_relative(name: str) -> Path:
    value = PurePosixPath(name)
    if not name or value.is_absolute() or ".." in value.parts or ".git" in value.parts or str(value) != name or "\x00" in name:
        raise ValueError(f"unsafe artifact path: {name!r}")
    return Path(*value.parts)


def materialize_base(spec: dict, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    archive = git(Path(spec["base_repo"]), "archive", "--format=tar", spec["base_commit"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            target = destination / safe_relative(member.name.rstrip("/"))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.extractfile(member).read())
                target.chmod(member.mode & 0o777)
            else:
                raise ValueError("SWE-bench snapshot adapter does not support linked/special baseline files")


@dataclass(frozen=True)
class FrozenSpec:
    config_json: bytes
    dataset_json: bytes
    base_files: tuple[tuple[str, str], ...]

    @property
    def config(self):
        return json.loads(self.config_json)


def prepare_spec(spec: dict, *, relative_to: Path | None = None, snapshot: Path | None = None, credential_env: str | None = None) -> FrozenSpec:
    """Validate/freeze trusted scoring inputs before model execution, if any."""
    source = json.loads(json.dumps(spec))
    if set(source) - FIELDS:
        raise ValueError(f"unsupported SWE-bench scoring keys: {sorted(set(source) - FIELDS)}")
    config = {**DEFAULTS, **source}
    if config["type"] != "swebench":
        raise ValueError("expected scorer type=swebench")
    for key in ("python", "dataset_name", "instance_id", "base_commit", "base_repo"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"SWE-bench scoring requires {key}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["instance_id"]):
        raise ValueError("invalid instance_id")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", config["base_commit"]):
        raise ValueError("base_commit must be the full 40-character commit ID")
    for key in ("timeout", "wall_timeout", "max_workers"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["cache_level"] not in {"none", "base", "env", "instance"}:
        raise ValueError("invalid cache_level")
    for key in ("split", "instance_image_tag", "env_image_tag"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if config["namespace"] is not None and not isinstance(config["namespace"], str):
        raise ValueError("namespace must be a string or null")
    relative_to = (relative_to or Path.cwd()).resolve()
    for key in ("python", "base_repo"):
        # Do not resolve the venv Python symlink to the system interpreter.
        config[key] = os.path.abspath(relative_to / Path(config[key]).expanduser())
    if not os.access(config["python"], os.X_OK):
        raise ValueError("official SWE-bench Python is not executable")
    resolved = git(Path(config["base_repo"]), "rev-parse", "--verify", config["base_commit"] + "^{commit}").decode().strip()
    if resolved != config["base_commit"].lower():
        raise ValueError("base repository does not contain the specified exact commit")
    config["base_commit"] = resolved
    dataset = config["dataset_name"]
    candidate = (relative_to / Path(dataset).expanduser()).absolute()
    if candidate.is_file():
        raw = candidate.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if candidate.suffix == ".jsonl" else json.loads(raw)
        config["dataset_name"] = str(candidate)
        config["dataset_source_sha256"] = digest(raw)
    else:
        if dataset.endswith((".json", ".jsonl")):
            raise ValueError(f"dataset file does not exist: {candidate}")
        code = ("import json,sys; from swebench.harness.utils import load_swebench_dataset; "
                "print(json.dumps(load_swebench_dataset(sys.argv[1],sys.argv[2],[sys.argv[3]])))")
        loaded = subprocess.run([config["python"], "-c", code, dataset, config["split"], config["instance_id"]],
                                capture_output=True, timeout=300, env=trusted_env([credential_env] if credential_env else []))
        if loaded.returncode:
            raise ValueError("official dataset loading failed; no model or evaluation was started")
        rows = json.loads(loaded.stdout)
    if not isinstance(rows, list):
        raise ValueError("official dataset JSON must be a list of instance rows")
    selected = [row for row in rows if row.get("instance_id") == config["instance_id"]]
    if len(selected) != 1 or selected[0].get("base_commit") != config["base_commit"]:
        raise ValueError("dataset instance/base_commit does not match the scoring specification")
    if config.get("repo") and selected[0].get("repo") != config["repo"]:
        raise ValueError("dataset repository mismatch")
    config["repo"] = selected[0].get("repo")
    dataset_json = json.dumps(selected, ensure_ascii=False).encode()
    config["frozen_dataset_sha256"] = digest(dataset_json)
    with tempfile.TemporaryDirectory(prefix="agentctl-swe-base-") as temporary:
        base = Path(temporary)
        materialize_base(config, base)
        files = inventory(base)
    if snapshot is not None and inventory(snapshot) != files:
        raise ValueError("frozen input snapshot does not exactly match the specified baseline commit")
    return FrozenSpec(json.dumps(config, sort_keys=True).encode(), dataset_json, tuple(sorted(files.items())))


def _index_tree(root):
    git(root, "init", "-q")
    git(root, "-c", "core.autocrlf=false", "add", "-f", "-A", "--", ".")
    return git(root, "write-tree").decode().strip()


def _diff(root, tree):
    git(root, "-c", "core.autocrlf=false", "add", "-f", "-A", "--", ".")
    return git(root, "diff", "--cached", "--binary", "--full-index", "--no-renames", "--no-ext-diff", tree, "--")


def snapshot_patch(before: Path, after: Path) -> bytes:
    """Canonical Git patch even when neither directory contains .git."""
    import shutil
    with tempfile.TemporaryDirectory(prefix="agentctl-swe-diff-") as temporary:
        root = Path(temporary)
        original = inventory(before)
        final = inventory(after)
        for name in original:
            target = root / safe_relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(before / name, target)
        tree = _index_tree(root)
        for name in original.keys() - final.keys():
            (root / name).unlink()
        for name in final:
            target = root / safe_relative(name)
            if target.is_dir():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(after / name, target)
        return _diff(root, tree)


def bind_archived_run(run: Path, frozen: FrozenSpec):
    """Reconstruct from archived files; the live candidate checkout is ignored."""
    import shutil
    run = run.resolve()
    raw = {name: (run / name).read_bytes() for name in ("summary.json", "start.json", "artifacts.json", "task.json")}
    summary, start, artifacts, task = (json.loads(raw[name]) for name in ("summary.json", "start.json", "artifacts.json", "task.json"))
    if summary.get("schema") != "agentos.run.v1" or summary.get("run_id") != task.get("run_id"):
        raise ValueError("original run identity/schema mismatch")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", summary["run_id"]):
        raise ValueError("unsafe source run_id")
    base_files = dict(frozen.base_files)
    provenance = start.get("provenance") or {}
    head = provenance.get("verified_base_commit") or (provenance.get("git") or start.get("git") or {}).get("head")
    if head != frozen.config["base_commit"] or start.get("files") != base_files:
        raise ValueError("original run starting content/commit does not match scoring baseline")
    changes = artifacts.get("changes")
    if not isinstance(changes, list) or changes != summary.get("artifacts"):
        raise ValueError("archived artifact manifest disagrees with original summary")
    blobs, expected = {}, dict(base_files)
    changed_paths = set()
    for change in changes:
        name = change["path"]
        safe_relative(name)
        if name in changed_paths or change.get("before_sha256") != base_files.get(name):
            raise ValueError("duplicate or incorrect artifact baseline")
        changed_paths.add(name)
        after_hash = change.get("after_sha256")
        if after_hash == change.get("before_sha256"):
            raise ValueError("artifact change contains no content change")
        if after_hash is None:
            expected.pop(name, None)
        else:
            path = run / "artifacts/files" / name
            if path.is_symlink() or not path.resolve().is_relative_to((run / "artifacts/files").resolve()):
                raise ValueError("archived artifact escapes run")
            content = path.read_bytes()
            if digest(content) != after_hash:
                raise ValueError(f"archived artifact hash mismatch: {name}")
            blobs[name] = content
            expected[name] = after_hash
    with tempfile.TemporaryDirectory(prefix="agentctl-swe-artifacts-") as temporary:
        root = Path(temporary)
        materialize_base(frozen.config, root)
        tree = _index_tree(root)
        for change in changes:
            if change["after_sha256"] is None:
                (root / change["path"]).unlink()
        for name, content in blobs.items():
            target = root / name
            if target.is_dir():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        canonical = _diff(root, tree)
        if inventory(root) != expected:
            raise ValueError("reconstructed artifact content mismatch")
    patch, source = canonical, "canonical_diff_of_archived_artifacts_and_verified_baseline"
    archived = run / "model.patch"
    if archived.exists():
        patch = archived.read_bytes()
        with tempfile.TemporaryDirectory(prefix="agentctl-swe-verify-patch-") as temporary:
            root = Path(temporary)
            materialize_base(frozen.config, root)
            tree = _index_tree(root)
            if patch.strip():
                git(root, "apply", "--binary", "-", input_data=patch)
            _diff(root, tree)
            names = {name.decode() for name in git(root, "diff", "--cached", "--name-only", "-z", "--no-renames", tree, "--").split(b"\x00") if name}
            if inventory(root) != expected or names != changed_paths:
                raise ValueError("model.patch does not reproduce this run's archived artifacts")
        raw["model.patch"] = patch
        source = "verified_archived_model.patch"
    patch.decode("utf-8")  # prediction JSON must preserve patch bytes exactly
    for optional in ("config.json", "calls.jsonl", "events.jsonl"):
        if (run / optional).is_file():
            raw[optional] = (run / optional).read_bytes()
    binding = {"source_run_id": summary["run_id"], "source_run": str(run),
               "instance_id": frozen.config["instance_id"], "base_commit": frozen.config["base_commit"],
               "patch_sha256": digest(patch), "patch_source": source, "canonical_patch_sha256": digest(canonical),
               "source_records_sha256": {name: digest(data) for name, data in raw.items()},
               "artifact_changes": changes}
    return summary, raw["summary.json"], patch, binding


def parse_outcome(instance_id, *, instance_report=None, log="", aggregate=None, process=None, empty=False):
    """Instance evidence wins; neither exit 0 nor generic error_ids imply pass."""
    process = process or {}
    common = {"official": True, "harness_returncode": process.get("returncode"),
              "harness_timed_out": bool(process.get("timed_out"))}
    def result(outcome, status, passed, reason):
        return {**common, "outcome": outcome, "status": status, "passed": passed, "reason": reason}
    if empty:
        return result("not_evaluated", "not_evaluated", None, "empty_patch")
    report = instance_report.get(instance_id) if isinstance(instance_report, dict) else None
    if isinstance(report, dict) and report.get("patch_exists") is False:
        return result("not_evaluated", "not_evaluated", None, "official_report_patch_missing")
    if isinstance(report, dict) and isinstance(report.get("resolved"), bool):
        if report.get("patch_successfully_applied") is not False:
            return result("resolved" if report["resolved"] else "unresolved", "completed", report["resolved"], "official_instance_report")
    if re.search(r"(?m)^(?:.*? - (?:INFO|ERROR) - )?>>>>> Patch Apply Failed(?::|$)", log):
        return result("patch_apply_error", "error", False, "official_patch_apply_failure_marker")
    lowered = log.lower()
    environment_markers = ("buildimageerror", "docker.errors", "container failed", "error building image", "failed to build", "connection refused", "no space left on device", "image not found", "pull access denied")
    if any(marker in lowered for marker in environment_markers):
        return result("environment_error", "error", None, "official_environment_failure_evidence")
    if process.get("timed_out") or "test timed out" in lowered or "timeout error:" in lowered:
        return result("evaluation_error", "error", None, "evaluation_timeout")
    if aggregate and instance_id in aggregate.get("empty_patch_ids", []):
        return result("not_evaluated", "not_evaluated", None, "official_empty_patch")
    return result("evaluation_error", "error", None, "missing_or_invalid_instance_report")


async def run_official(argv, directory, timeout, *, credential_env=None):
    with (directory / "stdout.txt").open("xb") as out, (directory / "stderr.txt").open("xb") as err:
        process = await asyncio.create_subprocess_exec(*argv, cwd=directory, env=trusted_env([credential_env] if credential_env else []),
            stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except TimeoutError:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        return {"returncode": process.returncode, "timed_out": timed_out}


async def evaluate_run(run: Path, spec: dict | FrozenSpec, *, output: Path | str = ".agentos/evaluations", relative_to=None):
    original_config = json.loads((Path(run) / "summary.json").read_text()).get("config", {})
    credential_env = original_config.get("api_key_env")
    frozen = spec if isinstance(spec, FrozenSpec) else prepare_spec(spec, relative_to=relative_to, credential_env=credential_env)
    source, source_bytes, patch, binding = bind_archived_run(Path(run), frozen)
    root = Path(output).expanduser().resolve()
    if root.is_relative_to(Path(run).resolve()):
        raise ValueError("evaluation output must be outside the original run")
    evaluation_id = "eval-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    bundle = root / evaluation_id
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "source-summary.json").write_bytes(source_bytes)
    (bundle / "prediction.patch").write_bytes(patch)
    config = frozen.config
    write_json(bundle / "spec.json", config)
    write_json(bundle / "binding.json", binding)
    official = bundle / "official"
    official.mkdir()
    (official / "dataset.json").write_bytes(frozen.dataset_json)
    label = "agentctl-" + source["run_id"]
    prediction = {"instance_id": config["instance_id"], "model_name_or_path": label, "model_patch": patch.decode()}
    (official / "predictions.jsonl").write_text(json.dumps(prediction, ensure_ascii=False) + "\n")
    argv = [config["python"], "-m", "swebench.harness.run_evaluation",
            "--dataset_name", str(official / "dataset.json"), "--split", config["split"],
            "--predictions_path", str(official / "predictions.jsonl"), "--instance_ids", config["instance_id"],
            "--run_id", evaluation_id, "--timeout", str(config["timeout"]), "--max_workers", str(config["max_workers"]),
            "--namespace", config["namespace"] or "none", "--instance_image_tag", config["instance_image_tag"],
            "--env_image_tag", config["env_image_tag"], "--cache_level", config["cache_level"], "--clean", "false",
            "--rewrite_reports", "false"]
    write_json(bundle / "invocation.json", {"argv": argv, "cwd": str(official), "official_run_id": evaluation_id,
               "trust_boundary": "trusted official Docker evaluator; not the agent command sandbox"})
    start = time.monotonic()
    process = {}
    if patch.strip():
        try:
            process = await run_official(argv, official, config["wall_timeout"], credential_env=credential_env)
        except OSError as exc:
            process = {"returncode": None, "timed_out": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        process = {"returncode": None, "timed_out": False, "not_started": "empty_patch"}
    # Installed 4.1.0 uses run_evaluation; newer documentation uses evaluation.
    log_dirs = [official / "logs" / name / evaluation_id / label / config["instance_id"]
                for name in ("run_evaluation", "evaluation")]
    log_dir = next((path for path in log_dirs if (path / "report.json").exists()),
                   next((path for path in log_dirs if path.exists()), log_dirs[0]))
    report_file = log_dir / "report.json"
    aggregate_files = [official / f"{label}.{evaluation_id}.json", *[
        official / "logs" / name / evaluation_id / "results.json" for name in ("run_evaluation", "evaluation")]]
    def read_json(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None
    log = (log_dir / "run_instance.log").read_text(errors="replace") if (log_dir / "run_instance.log").exists() else ""
    stderr = (official / "stderr.txt").read_text(errors="replace") if (official / "stderr.txt").exists() else ""
    aggregate = next((data for path in aggregate_files if (data := read_json(path)) is not None), None)
    score = parse_outcome(config["instance_id"], instance_report=read_json(report_file), log=log + "\n" + stderr,
                          aggregate=aggregate, process=process, empty=not patch.strip())
    if process.get("error"):
        score.update(outcome="environment_error", status="error", passed=None, reason=process["error"])
    result = {"schema": "agentos.evaluation.v1", "evaluation_id": evaluation_id, "output": str(bundle),
              **binding, "source_summary_sha256": digest(source_bytes), "score": score,
              "evaluation_seconds": time.monotonic() - start, "process": process,
              "official_instance_report": str(report_file.relative_to(bundle)) if report_file.exists() else None,
              "official_aggregate_reports": [str(path.relative_to(bundle)) for path in aggregate_files if path.exists()]}
    write_json(bundle / "evaluation.json", result)
    return result
