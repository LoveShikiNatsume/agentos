from __future__ import annotations

from pathlib import Path

from agentos.scenarios.sched_ext_business.fixture import freeze_fixture
from agentos.scenarios.sched_ext_business.pipeline import (
    BUSINESS_TASK_IDS,
    build_business_dag,
)


def test_business_dag_freezes_expected_sla_and_worker_mapping(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("value = 1\n", encoding="utf-8")
    fixture_root = tmp_path / "fixture"
    manifest = freeze_fixture(source, fixture_root, ["code.py"])

    dag = build_business_dag(fixture_root, manifest, work_units=3)

    assert set(dag.tasks) == {"issue_triage", *BUSINESS_TASK_IDS}
    for task_id in BUSINESS_TASK_IDS:
        task = dag.tasks[task_id]
        assert task.dependencies == ["issue_triage"]
        assert task.metadata["execution"] == "subprocess"
        assert task.metadata["worker_type"] != "sleep"
        assert task.metadata["worker_payload"]["work_units"] == 3
        assert task.metadata["worker_payload"]["manifest"] == manifest.to_dict()

    assert dag.tasks["patch_validation"].sla_class == "critical"
    assert dag.tasks["security_gate"].sla_class == "critical"
    assert dag.tasks["unit_test_select"].sla_class is None
    assert dag.tasks["dependency_analysis"].sla_class is None
    for task_id in {
        "repository_index",
        "artifact_hash",
        "documentation_scan",
        "symbol_index",
    }:
        assert dag.tasks[task_id].sla_class == "batch"


def test_business_dag_accepts_per_worker_work_units(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("value = 1\n", encoding="utf-8")
    fixture_root = tmp_path / "fixture"
    manifest = freeze_fixture(source, fixture_root, ["code.py"])

    units = {task_id: index + 1 for index, task_id in enumerate(BUSINESS_TASK_IDS)}
    dag = build_business_dag(fixture_root, manifest, work_units=units)

    for task_id, value in units.items():
        assert dag.tasks[task_id].metadata["worker_payload"]["work_units"] == value


def test_agentos_normal_forces_normal_kernel_queue(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("value = 1\n", encoding="utf-8")
    fixture_root = tmp_path / "fixture"
    manifest = freeze_fixture(source, fixture_root, ["code.py"])

    dag = build_business_dag(
        fixture_root,
        manifest,
        work_units=1,
        kernel_queue_mode="normal",
    )

    for task_id in BUSINESS_TASK_IDS:
        task = dag.tasks[task_id]
        assert task.sla_class is None
        assert task.metadata["expected_queue_class"] == "normal"
        assert task.metadata["business_sla_class"] in {"critical", "normal", "batch"}


def test_migrated_submission_fixture_retains_logical_inputs(tmp_path):
    from pathlib import Path
    from agentos.benchmarks.sched_ext_business import DEFAULT_FIXTURE_PATHS
    from agentos.scenarios.sched_ext_business.fixture import freeze_fixture, verify_fixture

    repository = Path(__file__).resolve().parents[1]
    frozen = tmp_path / "submission-fixture"
    manifest = freeze_fixture(repository / "src", frozen, DEFAULT_FIXTURE_PATHS)
    values = verify_fixture(frozen, manifest)
    assert set(values) == set(DEFAULT_FIXTURE_PATHS)
    assert values["tests/test_sched_ext_adapter.py"] == (
        repository / "tests/test_sched_ext_adapter.py"
    ).read_bytes()
    assert values["agentos/runtime/engine.py"] == (
        repository / "src/agentos/runtime/engine.py"
    ).read_bytes()
