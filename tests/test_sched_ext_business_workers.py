from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentos.scenarios.sched_ext_business.fixture import freeze_fixture
from agentos.scenarios.sched_ext_business.workers import (
    BUSINESS_WORKER_TYPES,
    run_business_worker,
)
from agentos.worker.process import SubprocessWorkerManager, WorkerSpec


@pytest.fixture
def business_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "pkg" / "core.py").write_text(
        "import hashlib\n\n"
        "def digest(value: bytes) -> str:\n"
        "    return hashlib.sha256(value).hexdigest()\n",
        encoding="utf-8",
    )
    (source / "pkg" / "service.py").write_text(
        "from pkg.core import digest\n\n"
        "class Service:\n"
        "    async def run(self, value: bytes) -> str:\n"
        "        return digest(value)\n",
        encoding="utf-8",
    )
    (source / "tests" / "test_service.py").write_text(
        "from pkg.service import Service\n\n"
        "def test_service_exists():\n"
        "    assert Service\n",
        encoding="utf-8",
    )
    (source / "README.md").write_text(
        "# Fixture\n\nSee [service](pkg/service.py).\n",
        encoding="utf-8",
    )
    fixture_root = tmp_path / "fixture"
    manifest = freeze_fixture(
        source,
        fixture_root,
        [
            "README.md",
            "pkg/core.py",
            "pkg/service.py",
            "tests/test_service.py",
        ],
    )
    return fixture_root, manifest.to_dict()


@pytest.mark.parametrize("worker_type", sorted(BUSINESS_WORKER_TYPES))
def test_business_workers_are_deterministic(
    business_fixture: tuple[Path, dict[str, object]],
    worker_type: str,
) -> None:
    fixture_root, manifest = business_fixture
    payload = {
        "fixture_root": str(fixture_root),
        "manifest": manifest,
        "work_units": 2,
    }

    first = run_business_worker(worker_type, payload)
    second = run_business_worker(worker_type, payload)

    assert first == second
    assert first["worker_type"] == worker_type
    assert first["work_units"] == 2
    assert first["input_sha256"] == manifest["aggregate_sha256"]
    assert len(first["checksum"]) == 64
    assert first["result"]


def test_business_worker_rejects_fixture_content_drift(
    business_fixture: tuple[Path, dict[str, object]],
) -> None:
    fixture_root, manifest = business_fixture
    (fixture_root / "pkg" / "core.py").write_text("raise SystemExit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture hash mismatch"):
        run_business_worker(
            "artifact_hash",
            {
                "fixture_root": str(fixture_root),
                "manifest": manifest,
                "work_units": 1,
            },
        )


def test_freeze_fixture_rejects_symlink_escape(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    (source / "escape.py").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        freeze_fixture(source, tmp_path / "fixture", ["escape.py"])


@pytest.mark.asyncio
async def test_business_worker_runs_through_subprocess_entrypoint(
    business_fixture: tuple[Path, dict[str, object]],
) -> None:
    fixture_root, manifest = business_fixture
    result = await SubprocessWorkerManager().run(
        WorkerSpec(
            worker_type="patch_validate",
            task_id="patch-validation",
            agent_id="agent-patch",
            payload={
                "fixture_root": str(fixture_root),
                "manifest": manifest,
                "work_units": 1,
            },
        ),
        timeout_seconds=15.0,
    )

    assert result.ok is True
    assert result.result["worker_type"] == "patch_validate"
    assert result.result["runtime_observation"]["policy"] == 0
    assert result.result["runtime_observation"]["ext_enabled"] in {-1, 0}
    assert result.result["runtime_observation"]["affinity"]
    assert json.loads(result.stdout)["result"]["checksum"] == result.result["checksum"]
