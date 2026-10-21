"""Immutable manifest loading and explicit backend configuration."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.context_selection.common import (
    add_model_arguments, config_from_args, digest, load, questions_for, write_new,
)

HERE = Path(__file__).resolve().parent
PROTOCOL_VERSION = "semantic-recovery-v1"
ARMS = ("dirty", "snapshot", "no_retry")
INJECTION_TYPES = ("execution", "semantic")


def materialize(spec, source):
    if spec.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported semantic-recovery protocol")
    if digest(source) != spec["source_manifest_sha256"]:
        raise ValueError("source manifest hash mismatch")
    assignments = spec["injections"]
    ids = [row["question_id"] for row in assignments]
    questions = source["questions"]
    if len(ids) != len(set(ids)) or set(ids) != {q["id"] for q in questions}:
        raise ValueError("injection manifest must cover each source question exactly once")
    by_id = {row["question_id"]: row for row in assignments}
    result = deepcopy(source)
    result["protocol_version"] = PROTOCOL_VERSION
    result["manifest_sha256"] = digest(spec)
    result["injection_seed"] = spec["seed"]
    for q in result["questions"]:
        injection = by_id[q["id"]]
        if (injection["type"] not in INJECTION_TYPES
                or isinstance(injection["after_private_segments"], bool)
                or not isinstance(injection["after_private_segments"], int)
                or not 1 <= injection["after_private_segments"] <= len(q["required_refs"])
                or injection.get("attempt") != 1):
            raise ValueError("invalid injection assignment")
        if not isinstance(injection.get("wrong_fact"), dict) or not isinstance(injection.get("wrong_answer"), dict):
            raise ValueError("manifest must freeze injected wrong fact and answer")
        q["injection"] = deepcopy(injection)
    return result


def load_manifest(path):
    path = Path(path).resolve()
    spec = load(path)
    source_path = (path.parent / spec["source_manifest"]).resolve()
    return materialize(spec, load(source_path))
