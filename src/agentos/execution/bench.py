"""Manifest-driven isolated runs and scoring of those exact workspaces."""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from agentos.execution.config import FIELDS, resolve_config
from agentos.execution.records import EXCLUDED, git_state, inventory
from agentos.execution.service import execute
from agentos.execution.swebench import prepare_spec, evaluate_run, digest
from agentos.execution.tools import run_command
from agentos.core.resources import ResourceQuota
from agentos.worker.process import WorkerSpec
from agentos.worker.sandbox import BubblewrapCommandWorker

SCORERS = {}


def register_scorer(name, scorer):
    if name in SCORERS:
        raise ValueError(f"scorer already registered: {name}")
    SCORERS[name] = scorer


async def command_scorer(spec, cwd, output, *, run_config=None):
    if run_config and run_config["command_sandbox"] == "bubblewrap":
        worker = BubblewrapCommandWorker(cwd, read_only_paths=run_config["command_read_only_paths"])
        completed = await worker.run(WorkerSpec(
            "command", "evaluation", "evaluator", output.name,
            payload={"argv": spec["argv"]},
            resource_quota=ResourceQuota(allow_filesystem_write=True, allow_network=run_config["command_network"]),
        ), timeout_seconds=spec.get("timeout", 120))
        result = {"returncode": completed.exit_code, "stdout": completed.stdout, "stderr": completed.stderr,
                  "worker_pid": completed.pid, "command_backend": "bubblewrap", **completed.result}
        if completed.timed_out:
            return {"status": "error", "passed": None, "error": "evaluation command timed out", "result": result}
    else:
        env = dict(os.environ)
        if run_config:
            env.pop(run_config["api_key_env"], None)
        result = await run_command(spec["argv"], cwd, spec.get("timeout", 120), env=env)
        result["command_backend"] = "host"
    return {"status": "completed", "passed": result["returncode"] == 0, "result": result}


async def file_equals_scorer(spec, cwd, output):
    target = (cwd / spec["path"]).resolve()
    if not target.is_relative_to(cwd.resolve()):
        raise ValueError("score path escapes workspace")
    exists = target.is_file()
    return {"status": "completed", "passed": exists and target.read_text() == spec["expected"],
            "path": spec["path"], "exists": exists}


async def swebench_scorer(spec, cwd, output, *, frozen=None, evaluation_root=None):
    # Official Docker evaluation is deliberately outside the agent command sandbox.
    return await evaluate_run(output, frozen or prepare_spec(spec),
                              output=evaluation_root or Path(output).parent / "evaluations")


register_scorer("swebench", swebench_scorer)
register_scorer("command", command_scorer)
register_scorer("file_equals", file_equals_scorer)


async def run_suite(suite: str, *, compare="native,runtime", output=".agentos/batches", overrides=None, backend_factory=None, listener=None):
    manifest_path = Path(suite).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"suite {suite!r} is not configured; supply a JSON manifest path (built-in finals/swe-verified-30 datasets are not registered)")
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest.get("tasks")
    if manifest.get("schema") != "agentos.bench.v1" or not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest requires schema agentos.bench.v1 and nonempty tasks")
    arms = compare.split(",")
    combined = set(arms) == {"baseline", "optimized"} and len(arms) == 2
    if not combined and (not arms or len(set(arms)) != len(arms) or any(p not in {"native", "runtime"} for p in arms)):
        raise ValueError("compare must be native/runtime profiles or the two arms baseline,optimized")
    profiles = ["runtime"] if combined else arms
    seen = set()
    prepared = []
    credentials = set()
    for task in tasks:
        tid = task.get("id")
        if not isinstance(tid, str) or not tid or tid in seen or not all(c.isalnum() or c in "-_" for c in tid):
            raise ValueError("manifest tasks require unique safe ids")
        seen.add(tid)
        if not isinstance(task.get("task"), str) or not task["task"].strip():
            raise ValueError(f"missing natural language task: {tid}")
        source = (manifest_path.parent / task["cwd"]).resolve()
        if not source.is_dir():
            raise ValueError(f"missing task workspace: {source}")
        revision = task.get("revision")
        state = git_state(source)
        if revision and (state["head"] != revision or state["status"]):
            raise ValueError(f"{tid}: revision protocol requires clean workspace at exact full commit {revision}")
        spec = task.get("score")
        if spec and spec.get("type") not in SCORERS:
            raise ValueError(f"unsupported scorer: {spec.get('type')}")
        settings = {**manifest.get("config", {}), **task.get("config", {}),
                    **{k: v for k, v in (overrides or {}).items() if v is not None}}
        if combined:
            settings["profile"] = "runtime"
            settings.setdefault("max_parallel", 2)
        resolved = resolve_config(source, settings)
        common_settings = {key: resolved[key] for key in FIELDS if key in resolved}
        credential = os.environ.get(resolved["api_key_env"], "")
        if credential:
            credentials.add(credential)
        frozen = prepare_spec(spec, relative_to=manifest_path.parent, snapshot=source, credential_env=resolved["api_key_env"]) if spec and spec["type"] == "swebench" else None
        prepared.append((task, source, state, frozen, common_settings))
    batch_id = "batch-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    batch = Path(output).expanduser().resolve() / batch_id
    # Never recursively copy a batch directory into itself.
    if any(batch.is_relative_to(source) and not any(p in EXCLUDED for p in batch.relative_to(source).parts) for _, source, _, _, _ in prepared):
        raise ValueError("batch output must be outside source workspaces or under an excluded .agentos/runs directory")
    batch.mkdir(parents=True, exist_ok=False)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    for credential in credentials:
        manifest_text = manifest_text.replace(json.dumps(credential, ensure_ascii=False)[1:-1], "[REDACTED]")
    (batch / "manifest.json").write_text(manifest_text)
    def copy_input(source_file, destination):
        if credentials:
            content = Path(source_file).read_bytes()
            if any(credential.encode() in content for credential in credentials):
                raise ValueError("input contains the configured model credential; remove it from source files before snapshotting")
        return shutil.copy2(source_file, destination)
    comparison = {"comparison": "combined_context_prefix_cache" if combined else "execution_profiles",
                  "arms": arms, "profiles": profiles,
                  "order_policy": "alternate_first_arm_by_task_index" if combined else "declared_order",
                  "tasks": [{"task_id": task["id"],
                             "arm_order": list(reversed(arms)) if combined and index % 2 else list(arms),
                             "frozen_common_config": common}
                            for index, (task, _, _, _, common) in enumerate(prepared)]}
    (batch / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    snapshots = []
    for task, source, state, frozen, common_settings in prepared:
        snapshot = batch / "snapshots" / task["id"]
        shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns(*EXCLUDED), symlinks=True, copy_function=copy_input)
        # Reject linked inputs: copies must not mutate the original via symlinks.
        if any(p.is_symlink() for p in snapshot.rglob("*")):
            raise ValueError(f"{task['id']}: snapshot contains symlinks; materialize inputs first")
        starting_files = inventory(snapshot)
        if frozen and starting_files != dict(frozen.base_files):
            raise ValueError("snapshot changed after baseline verification")
        snapshots.append((snapshot, starting_files))
    rows = []
    for task_index, (task, source, state, frozen, common_settings) in enumerate(prepared):
        snapshot, starting_files = snapshots[task_index]
        arm_order = comparison["tasks"][task_index]["arm_order"]
        for arm_index, arm in enumerate(arm_order):
            if inventory(snapshot) != starting_files:
                raise ValueError("frozen snapshot was modified during the batch")
            profile = "runtime" if combined else arm
            mode = {"baseline": "off", "optimized": "on"}[arm] if combined else common_settings["context_optimization"]
            workspace = batch / "workspaces" / task["id"] / arm
            shutil.copytree(snapshot, workspace)
            spec = task.get("score")
            settings = {**common_settings, "profile": profile, "context_optimization": mode,
                        "output": str(batch / "runs")}
            # Freeze trusted policy before the candidate can edit project config.
            scoring_config = resolve_config(workspace, settings)
            async def score(cwd, output_path, spec=spec, scoring_config=scoring_config):
                scorer = SCORERS[spec["type"]]
                if scorer is command_scorer:
                    return await scorer(spec, cwd, output_path, run_config=scoring_config)
                return await scorer(spec, cwd, output_path)
            order = {"task_id": task["id"], "task_index": task_index, "arm": arm, "profile": profile,
                     "context_optimization": mode, "position_in_task": arm_index, "sequence": len(rows)}
            with (batch / "execution-order.jsonl").open("a") as stream:
                stream.write(json.dumps({**order, "state": "started", "timestamp": time.time()}) + "\n")
            result = await execute(task["task"], cwd=workspace, overrides=settings,
                                   backend=backend_factory() if backend_factory else None,
                                   listener=listener, scorer=score if spec and not frozen else None,
                                   provenance={"source": str(source), "git": state, "files": starting_files,
                                               "protocol": "frozen directory snapshot, exclusions documented",
                                               "verified_base_commit": frozen.config["base_commit"] if frozen else None,
                                               "arm": arm, "comparison": comparison["comparison"], "execution_order": order})
            row = {**order, "run_id": result["run_id"], "output": result["output"]}
            with (batch / "execution-order.jsonl").open("a") as stream:
                stream.write(json.dumps({**row, "state": "model_execution_finished", "timestamp": time.time()}) + "\n")
            if frozen:
                evaluation = await swebench_scorer(spec, workspace, Path(result["output"]),
                                                  frozen=frozen, evaluation_root=batch / "evaluations")
                row["evaluation"] = str(Path(evaluation["output"]).relative_to(batch))
            rows.append(row)
            (batch / "batch.json").write_text(json.dumps({"schema": "agentos.batch.v1", "batch_id": batch_id,
                  "output": str(batch), "profiles": profiles, "arms": arms, "comparison": comparison["comparison"],
                  "order_policy": comparison["order_policy"], "runs": rows}, ensure_ascii=False, indent=2))
    return json.loads((batch / "batch.json").read_text())


def _evaluation_record(path):
    evaluation = json.loads((path / "evaluation.json").read_text())
    raw = (path / "source-summary.json").read_bytes()
    source = json.loads(raw)
    if digest(raw) != evaluation["source_summary_sha256"] or source["run_id"] != evaluation["source_run_id"]:
        raise ValueError("evaluation source summary binding mismatch")
    return source, evaluation


def report(batch_path: str, *, root=".agentos/batches") -> str:
    path = Path(batch_path)
    if not path.exists():
        path = Path(root) / batch_path
    stream = io.StringIO()
    fields = ["batch_id", "task_id", "profile", "run_id", "workflow_status", "score_status", "passed",
              "execution_seconds", "evaluation_seconds", "prompt_tokens", "completion_tokens",
              "usage_complete", "missing_usage_calls", "failed_attempts", "retries", "cached_tokens", "backend_injected_for_test",
              "evaluation_id", "instance_id", "base_commit", "patch_sha256", "outcome", "source_run_id",
              "arm", "context_optimization", "execution_sequence", "cache_hit_tokens", "cache_miss_tokens",
              "cache_hit_source", "cache_miss_source"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    if (path / "evaluation.json").exists():
        source, evaluation = _evaluation_record(path)
        entries = [("", evaluation["instance_id"], source, evaluation, {})]
    else:
        data = json.loads((path / "batch.json").read_text())
        entries = []
        for item in data["runs"]:
            raw = (path / "runs" / item["run_id"] / "summary.json").read_bytes()
            source, evaluation = json.loads(raw), None
            if item.get("evaluation"):
                linked_source, evaluation = _evaluation_record(path / item["evaluation"])
                if linked_source != source or digest(raw) != evaluation["source_summary_sha256"]:
                    raise ValueError("batch evaluation does not match original run")
            entries.append((data["batch_id"], item["task_id"], source, evaluation, item))
    for batch_id, task_id, run, evaluation, item in entries:
        row = {k: run.get(k) for k in fields}
        row.update({k: v for k, v in run["usage"].items() if k in fields})
        score = evaluation["score"] if evaluation else run["score"]
        row.update(batch_id=batch_id, task_id=task_id, profile=run["config"]["profile"],
                   score_status=score["status"], passed=score.get("passed"), outcome=score.get("outcome"),
                   arm=item.get("arm", run.get("arm", run["config"]["profile"])),
                   context_optimization=run["config"].get("context_optimization", "legacy_unlabelled"),
                   execution_sequence=item.get("sequence"))
        if evaluation:
            row.update({key: evaluation[key] for key in ("evaluation_id", "instance_id", "base_commit", "patch_sha256", "source_run_id", "evaluation_seconds")})
        writer.writerow(row)
    return stream.getvalue()
