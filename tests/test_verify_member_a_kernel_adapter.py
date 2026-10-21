"""Tests for verify_member_a_kernel_adapter.py verifier."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = (Path(__file__).parent.parent / "src")


def _make_summary(*, pressure_count=5, bpf_count=3, cgroup_count=0) -> dict:
    """Build a minimal valid summary.json for testing."""
    adapter_mode = "cgroup_v2+mock_bpf" if cgroup_count > 0 else "user_space+mock_bpf"
    return {
        "metadata": {"backend": "mock", "workloads": ["ai_inference"]},
        "kernel_adapter_evidence": {
            "schema_version": "kernel_adapter_evidence.v1",
            "adapter_mode": adapter_mode,
            "event_counts": {
                "cgroup_quota_applied": cgroup_count,
                "cgroup_unavailable": 0,
                "cgroup_cleanup_failed": 0,
                "pressure_sampled": pressure_count,
                "pressure_system": pressure_count,
                "pressure_agent_cgroup": 0,
                "bpf_map_updated": bpf_count,
                "kernel_probe": 1,
                "scheduler_decision": 2,
            },
            "cgroup_quota_applied_count": cgroup_count,
            "pressure_sample_count": pressure_count,
            "bpf_map_update_count": bpf_count,
            "worst_pressure_level": "normal",
            "pressure_level_counts": {"normal": pressure_count},
            "quota_paths": [],
            "bpf_modes": ["mock"],
            "bpf_record_schema_versions": ["bpf.task_record.v2"],
            "probe": {
                "sched_ext": "unavailable",
                "bpf_map": "mock",
                "fallback_mode": "mock_bpf_map",
                "reason": "test",
            },
            "fallback_active": True,
            "fallback_reasons": ["test"],
            "cgroup_control_observed": cgroup_count > 0,
        },
    }


class TestVerifierPass:
    """Test verifier passes with valid summary."""

    def test_valid_summary_passes(self, tmp_path):
        summary = _make_summary()
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "RESULT: PASS" in result.stdout

    def test_valid_summary_with_cgroup_passes(self, tmp_path):
        summary = _make_summary(cgroup_count=2)
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout


class TestVerifierWarn:
    """Test verifier warns when cgroup is 0 in default mode."""

    def test_cgroup_zero_warns_but_passes(self, tmp_path):
        summary = _make_summary(cgroup_count=0)
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "RESULT: PASS" in result.stdout
        assert "WARN" in result.stdout


class TestVerifierStrictCgroup:
    """Test strict-cgroup mode."""

    def test_strict_cgroup_zero_fails(self, tmp_path):
        summary = _make_summary(cgroup_count=0)
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path), "--strict-cgroup"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_strict_cgroup_nonzero_passes(self, tmp_path):
        summary = _make_summary(cgroup_count=3)
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path), "--strict-cgroup"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "RESULT: PASS" in result.stdout


class TestVerifierFail:
    """Test verifier fails on bad input."""

    def test_missing_file_fails(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(tmp_path / "nonexistent.json")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_corrupt_json_fails(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text("not json {{{", encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_missing_evidence_fails(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout


class TestVerifierJson:
    """Test --json output."""

    def test_json_output_is_valid(self, tmp_path):
        summary = _make_summary()
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path), "--json"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["passed"] is True
        assert output["fail_count"] == 0
        assert "results" in output

    def test_json_output_shows_failures(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text("bad json", encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path), "--json"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert output["passed"] is False
        assert output["fail_count"] > 0


# ── Phase 5: Enhanced verifier tests ──


def _make_summary_with_adapter_mode(mode: str = "user_space+mock_bpf") -> dict:
    """Build a summary with a composite adapter_mode."""
    s = _make_summary()
    s["kernel_adapter_evidence"]["adapter_mode"] = mode
    return s


class TestVerifierProbeRequired:
    """probe missing or empty must FAIL (not WARN), exit code 1."""

    def test_empty_probe_fails(self, tmp_path):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["probe"] = {}
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, "empty probe must exit 1"
        assert "FAIL" in result.stdout

    def test_missing_probe_key_fails(self, tmp_path):
        summary = _make_summary()
        del summary["kernel_adapter_evidence"]["probe"]
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, "missing probe must exit 1"
        assert "FAIL" in result.stdout

    def test_probe_missing_reason_fails(self, tmp_path):
        summary = _make_summary()
        # Remove 'reason' field
        summary["kernel_adapter_evidence"]["probe"] = {
            "sched_ext": "unavailable",
            "bpf_map": "mock",
            "fallback_mode": "mock_bpf_map",
        }
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_empty_probe_json_output_still_valid(self, tmp_path):
        """--json must output valid JSON even when probe empty."""
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["probe"] = {}
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path), "--json"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        output = json.loads(result.stdout)  # must not raise
        assert output["passed"] is False


class TestVerifierErrorHandling:
    """Verifier must give clean FAIL on bad inputs, no traceback."""

    def test_nonexistent_file_no_traceback(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(tmp_path / "no_such_file.json")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr
        assert "FAIL" in result.stdout

    def test_invalid_json_no_traceback(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text("{{{bad json", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr

    def test_json_root_not_dict_fails(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_kernel_adapter_evidence_not_dict_fails(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text(json.dumps({"kernel_adapter_evidence": "wrong"}), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_event_counts_not_dict_fails(self, tmp_path):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["event_counts"] = "not_a_dict"
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_count_value_string_fails(self, tmp_path):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["pressure_sample_count"] = "five"
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_count_value_negative_fails(self, tmp_path):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["pressure_sample_count"] = -1
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_count_value_none_fails(self, tmp_path):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["bpf_map_update_count"] = None
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_missing_file_json_mode_valid_json(self, tmp_path):
        """--json must output valid JSON even when file does not exist."""
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(tmp_path / "none.json"), "--json"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert output["passed"] is False

    def test_invalid_json_json_mode_valid_json(self, tmp_path):
        """--json must output valid JSON even on parse error."""
        path = tmp_path / "summary.json"
        path.write_text("<<<not json>>>", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path), "--json"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert output["passed"] is False


class TestVerifierCgroupCleanupFailed:
    """cgroup_cleanup_failed > 0 must FAIL in both default and strict mode."""

    def _make_summary_with_cleanup_fail(self) -> dict:
        s = _make_summary()
        s["kernel_adapter_evidence"]["event_counts"]["cgroup_cleanup_failed"] = 1
        return s

    def test_cleanup_fail_default_mode_fails(self, tmp_path):
        summary = self._make_summary_with_cleanup_fail()
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_cleanup_fail_strict_mode_fails(self, tmp_path):
        summary = self._make_summary_with_cleanup_fail()
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path), "--strict-cgroup"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout


class TestVerifierMisleadingStrings:
    """Summary must not contain misleading kernel-takeover strings."""

    def _summary_with_text(self, misleading: str) -> dict:
        s = _make_summary()
        s["misleading_note"] = misleading
        return s

    def test_kernel_takeover_fails(self, tmp_path):
        summary = self._summary_with_text("kernel takeover enabled")
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_native_scheduler_enabled_fails(self, tmp_path):
        summary = self._summary_with_text("native scheduler enabled")
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_ebpf_scheduling_active_fails(self, tmp_path):
        summary = self._summary_with_text("eBPF scheduling active")
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_clean_summary_no_misleading(self, tmp_path):
        summary = _make_summary()
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestVerifierCompositeAdapterMode:
    """Verifier must accept composite adapter modes."""

    @pytest.mark.parametrize("mode", [
        "user_space+mock_bpf",
        "cgroup_v2+mock_bpf",
        "user_space+pinned_probe",
        "cgroup_v2+pinned_probe",
        "user_space+unavailable",
        "cgroup_v2+unavailable",
    ])
    def test_composite_adapter_mode_passes(self, tmp_path, mode):
        summary = _make_summary_with_adapter_mode(mode)
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"mode={mode} should pass: {result.stdout}"

    def test_old_mock_mode_fails(self, tmp_path):
        """Old 'mock' only mode should now be rejected."""
        summary = _make_summary_with_adapter_mode("mock")
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
             "--summary", str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout


# ── Phase 5 final review: strict probe content validation ──


def _run_verifier(tmp_path, summary: dict, extra_args: list[str] | None = None):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    cmd = [sys.executable, "scripts/verify_member_a_kernel_adapter.py",
           "--summary", str(path)] + (extra_args or [])
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)


class TestVerifierProbeContentStrict:
    """All probe field values must be non-empty strings."""

    @pytest.mark.parametrize("field,bad_value", [
        ("sched_ext", ""),
        ("sched_ext", "   "),
        ("sched_ext", None),
        ("sched_ext", 0),
        ("sched_ext", False),
        ("sched_ext", []),
        ("bpf_map", ""),
        ("bpf_map", "   "),
        ("bpf_map", None),
        ("fallback_mode", ""),
        ("fallback_mode", "   "),
        ("fallback_mode", 42),
        ("reason", ""),
        ("reason", "   "),
        ("reason", {}),
    ])
    def test_empty_or_invalid_probe_field_fails(self, tmp_path, field, bad_value):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["probe"][field] = bad_value
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 1, (
            f"field={field!r} value={bad_value!r} should fail but got exit 0\n{result.stdout}"
        )
        assert "FAIL" in result.stdout
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr

    def test_valid_probe_passes(self, tmp_path):
        summary = _make_summary()
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 0, result.stdout

    def test_invalid_probe_json_mode_no_traceback(self, tmp_path):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["probe"]["reason"] = ""
        result = _run_verifier(tmp_path, summary, ["--json"])
        assert result.returncode == 1
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr
        out = json.loads(result.stdout)
        assert out["passed"] is False

    def test_fail_output_names_bad_field(self, tmp_path):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["probe"]["sched_ext"] = ""
        result = _run_verifier(tmp_path, summary)
        assert "sched_ext" in result.stdout


# ── Phase 5 final review: strict count validation ──


_STANDARD_EVENT_COUNTS = [
    "cgroup_quota_applied", "cgroup_unavailable", "cgroup_cleanup_failed",
    "pressure_sampled", "pressure_system", "pressure_agent_cgroup",
    "bpf_map_updated", "kernel_probe", "scheduler_decision",
]
_TOP_LEVEL_ALIASES = [
    ("cgroup_quota_applied_count", "cgroup_quota_applied"),
    ("pressure_sample_count", "pressure_sampled"),
    ("bpf_map_update_count", "bpf_map_updated"),
]


class TestVerifierCountStrict:
    """Counts must be non-negative integers; bool rejected; aliases must match."""

    # --- top-level alias bad types ---
    @pytest.mark.parametrize("field,bad_value", [
        ("cgroup_quota_applied_count", -1),
        ("cgroup_quota_applied_count", "zero"),
        ("cgroup_quota_applied_count", None),
        ("cgroup_quota_applied_count", True),
        ("cgroup_quota_applied_count", 1.5),
        ("pressure_sample_count", -3),
        ("pressure_sample_count", "five"),
        ("pressure_sample_count", None),
        ("pressure_sample_count", False),
        ("bpf_map_update_count", -1),
        ("bpf_map_update_count", "x"),
        ("bpf_map_update_count", None),
        ("bpf_map_update_count", True),
    ])
    def test_invalid_top_level_count_fails(self, tmp_path, field, bad_value):
        summary = _make_summary()
        summary["kernel_adapter_evidence"][field] = bad_value
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 1, (
            f"{field}={bad_value!r} should fail but got exit 0\n{result.stdout}"
        )
        assert "FAIL" in result.stdout

    # --- event_counts bad types ---
    @pytest.mark.parametrize("ec_field,bad_value", [
        ("pressure_sampled", -1),
        ("pressure_sampled", "bad"),
        ("pressure_sampled", None),
        ("pressure_sampled", True),
        ("bpf_map_updated", -2),
        ("bpf_map_updated", False),
        ("cgroup_quota_applied", -1),
        ("cgroup_quota_applied", "none"),
        ("cgroup_unavailable", -5),
        ("cgroup_unavailable", []),
        ("cgroup_cleanup_failed", "x"),
        ("pressure_system", None),
        ("pressure_agent_cgroup", True),
        ("kernel_probe", -1),
        ("scheduler_decision", {}),
    ])
    def test_invalid_event_count_fails(self, tmp_path, ec_field, bad_value):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["event_counts"][ec_field] = bad_value
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 1, (
            f"event_counts.{ec_field}={bad_value!r} should fail but got exit 0\n{result.stdout}"
        )
        assert "FAIL" in result.stdout

    # --- missing event_counts fields ---
    @pytest.mark.parametrize("ec_field", _STANDARD_EVENT_COUNTS)
    def test_missing_event_count_field_fails(self, tmp_path, ec_field):
        summary = _make_summary()
        del summary["kernel_adapter_evidence"]["event_counts"][ec_field]
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 1, (
            f"missing event_counts.{ec_field} should fail\n{result.stdout}"
        )
        assert "FAIL" in result.stdout

    # --- top-level alias mismatch ---
    @pytest.mark.parametrize("top_field,ec_field", _TOP_LEVEL_ALIASES)
    def test_alias_mismatch_fails(self, tmp_path, top_field, ec_field):
        summary = _make_summary()
        # Make top-level differ from event_counts by +1
        orig = summary["kernel_adapter_evidence"]["event_counts"][ec_field]
        summary["kernel_adapter_evidence"][top_field] = orig + 1
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 1, (
            f"{top_field} mismatch should fail\n{result.stdout}"
        )
        assert "FAIL" in result.stdout

    # --- valid zero is ok for quota (WARN not FAIL) ---
    def test_valid_zero_quota_warns_not_fails(self, tmp_path):
        summary = _make_summary(cgroup_count=0)
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 0
        assert "WARN" in result.stdout

    # --- negative quota must FAIL not WARN ---
    def test_negative_quota_fails_not_warns(self, tmp_path):
        summary = _make_summary()
        summary["kernel_adapter_evidence"]["cgroup_quota_applied_count"] = -1
        summary["kernel_adapter_evidence"]["event_counts"]["cgroup_quota_applied"] = -1
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    # --- valid positive int passes ---
    def test_valid_positive_counts_pass(self, tmp_path):
        summary = _make_summary(pressure_count=10, bpf_count=5, cgroup_count=0)
        result = _run_verifier(tmp_path, summary)
        assert result.returncode == 0
