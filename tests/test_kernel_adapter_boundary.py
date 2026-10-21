from __future__ import annotations

import inspect

from agentos.scheduler.dag import DAGScheduler


def test_dag_scheduler_constructor_has_no_kernel_adapter_parameter() -> None:
    parameters = inspect.signature(DAGScheduler).parameters

    assert "bpf_map_adapter" not in parameters
    assert "kernel_adapter" not in parameters


def test_dag_scheduler_module_has_no_kernel_map_adapter_dependency() -> None:
    import agentos.scheduler.dag as dag_module

    source = inspect.getsource(dag_module)

    assert "MockBPFMapAdapter" not in source
    assert "BpftoolPinnedMapAdapter" not in source
    assert "CgroupV2Manager" not in source
    assert "task_record_from_decision" not in source
    assert "agentos.isolation.bpf" not in source
    assert "agentos.isolation.cgroup" not in source
