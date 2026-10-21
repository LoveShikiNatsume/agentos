from __future__ import annotations

import json

from scripts import verify_real_bpf_map as verifier


def _accepted_result() -> dict[str, object]:
    return {
        "available": True,
        "map_created": True,
        "schema_verified": True,
        "updates": 1,
        "lookups": 1,
        "readback_verified": True,
        "dumps": 1,
        "cleanup_failed": 0,
        "skip_reason": None,
        "failure_reason": None,
    }


def test_acceptance_requires_every_real_bpf_map_check() -> None:
    result = _accepted_result()

    assert verifier.acceptance_passed(result)

    for key, bad_value in (
        ("available", False),
        ("map_created", False),
        ("schema_verified", False),
        ("updates", 0),
        ("lookups", 0),
        ("readback_verified", False),
        ("dumps", 0),
        ("cleanup_failed", 1),
    ):
        candidate = dict(result)
        candidate[key] = bad_value
        assert not verifier.acceptance_passed(candidate), key


def test_unavailable_environment_emits_machine_readable_skip(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        verifier,
        "probe_environment",
        lambda **_kwargs: {
            "available": False,
            "bpftool_version": None,
            "bpffs_mounted": False,
            "reason": "bpftool not found",
        },
    )

    exit_code = verifier.main(["--bpffs-root", str(tmp_path / "bpf")])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["available"] is False
    assert payload["skip_reason"] == "bpftool not found"
    assert payload["failure_reason"] is None


def test_json_output_writes_parent_directories(tmp_path) -> None:
    output = tmp_path / "nested" / "evidence.json"

    verifier._out({"available": False}, str(output))

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "available": False
    }
