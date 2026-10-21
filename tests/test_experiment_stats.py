import math
import statistics

import pytest

from agentos.benchmarks.experiment_stats import describe, percentile


def test_percentile_uses_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_describe_uses_sample_stddev_and_student_t_ci():
    result = describe(list(range(1, 11)))

    assert result["n"] == 10
    assert result["mean"] == 5.5
    assert result["stddev"] == pytest.approx(
        statistics.stdev(range(1, 11))
    )
    margin = 2.262 * result["stddev"] / math.sqrt(10)
    assert result["ci95_low"] == pytest.approx(5.5 - margin, rel=1e-4)
    assert result["ci95_high"] == pytest.approx(5.5 + margin, rel=1e-4)
    assert result["minimum"] == 1
    assert result["maximum"] == 10


def test_describe_contains_percentiles():
    result = describe([1.0, 2.0, 3.0, 4.0])

    assert result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert result["n"] == 4
    assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]


def test_statistics_reject_empty_nonfinite_and_invalid_quantiles():
    with pytest.raises(ValueError, match="at least one"):
        describe([])
    with pytest.raises(ValueError, match="finite"):
        describe([1.0, float("nan")])
    with pytest.raises(ValueError, match="quantile"):
        percentile([1.0], 1.1)
