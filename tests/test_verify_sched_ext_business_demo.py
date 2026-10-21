from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_sched_ext_business_demo import run_functional_demo
from scripts.verify_sched_ext_business_demo import verify_bundle


def _write_source(root: Path) -> list[str]:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/tool.py").write_text(
        "import json\n\ndef inspect(value):\n    return json.dumps(value)\n",
        encoding="utf-8",
    )
    (root / "tests/test_tool.py").write_text(
        "from src.tool import inspect\n\ndef test_inspect():\n    assert inspect({}) == '{}'\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Fixture\n\n[tool](src/tool.py)\n", encoding="utf-8")
    return ["README.md", "src/tool.py", "tests/test_tool.py"]


def _bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    paths = _write_source(source)
    output = tmp_path / "bundle"
    result = run_functional_demo(
        source_root=source,
        output_dir=output,
        fixture_paths=paths,
        rounds=1,
        seed=20260719,
        work_units=1,
    )
    assert result["accepted"] is True
    return output


def _reseal(bundle: Path) -> None:
    lines = []
    for path in sorted(bundle.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.name}\n")
    (bundle / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def test_functional_bundle_passes_read_only_verifier(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    before = {
        path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.rglob("*")
        if path.is_file()
    }

    result = verify_bundle(bundle)

    after = {
        path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert result["passed"] is True
    assert result["functional_only"] is True
    assert result["sched_ext_performance_claim_allowed"] is False
    assert result["checks"] == "12/12"
    assert before == after


def test_verifier_rejects_fixture_drift(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "fixture/src/tool.py").write_text("changed\n", encoding="utf-8")

    result = verify_bundle(bundle)

    assert result["passed"] is False
    assert "fixture_verified" in result["failed_checks"]


def test_verifier_rejects_checksum_mismatch_and_missing_trial(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    trial = json.loads((bundle / "trials.jsonl").read_text(encoding="utf-8"))
    trial["task_results"]["artifact_hash"]["checksum"] = "0" * 64
    (bundle / "trials.jsonl").write_text(json.dumps(trial) + "\n", encoding="utf-8")
    _reseal(bundle)

    mismatch = verify_bundle(bundle)
    assert mismatch["passed"] is False
    assert "metrics_recomputed" in mismatch["failed_checks"]

    (bundle / "trials.jsonl").write_text("", encoding="utf-8")
    _reseal(bundle)
    missing = verify_bundle(bundle)
    assert missing["passed"] is False
    assert "trial_set_complete" in missing["failed_checks"]


def test_verifier_rejects_residual_resources(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    trial = json.loads((bundle / "trials.jsonl").read_text(encoding="utf-8"))
    trial["cleanup"]["workers"] = 1
    (bundle / "trials.jsonl").write_text(json.dumps(trial) + "\n", encoding="utf-8")
    _reseal(bundle)

    result = verify_bundle(bundle)

    assert result["passed"] is False
    assert "no_residual_resources" in result["failed_checks"]
