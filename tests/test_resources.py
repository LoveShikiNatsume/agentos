from agentos.core.resources import ResourceQuota, ResourceUsage


def test_resource_quota_defaults_are_unlimited():
    quota = ResourceQuota()
    assert quota.cpu_shares is None
    assert quota.cpu_max_percent is None
    assert quota.memory_max_mb is None
    assert quota.pids_max is None
    assert quota.timeout_seconds is None
    assert quota.allowed_tools == []
    assert quota.allow_network is False
    assert quota.allow_filesystem_write is False


def test_resource_quota_detects_limits():
    unlimited = ResourceQuota()
    limited = ResourceQuota(memory_max_mb=512)
    assert not unlimited.has_limits()
    assert limited.has_limits()


def test_resource_quota_serializes_to_dict():
    quota = ResourceQuota(
        cpu_max_percent=50.0,
        memory_max_mb=256,
        pids_max=32,
        timeout_seconds=10.0,
        allowed_tools=["pytest", "python"],
        allow_network=True,
    )
    assert quota.to_dict() == {
        "cpu_shares": None,
        "cpu_max_percent": 50.0,
        "memory_max_mb": 256,
        "pids_max": 32,
        "io_read_bps": None,
        "io_write_bps": None,
        "timeout_seconds": 10.0,
        "allowed_tools": ["pytest", "python"],
        "allow_network": True,
        "allow_filesystem_write": False,
    }


def test_resource_usage_records_runtime_measurements():
    usage = ResourceUsage(
        cpu_percent=12.5,
        memory_rss_mb=128.0,
        pids_current=3,
        io_read_bytes=4096,
        io_write_bytes=8192,
    )
    assert usage.to_dict()["memory_rss_mb"] == 128.0
    assert usage.to_dict()["io_write_bytes"] == 8192
