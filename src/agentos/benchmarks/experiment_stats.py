from __future__ import annotations

import math
import statistics
from collections.abc import Iterable


_T95 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _finite(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("at least one value is required")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("all values must be finite")
    return result


def percentile(values: Iterable[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(_finite(values))
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def student_t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if degrees_of_freedom < len(_T95):
        return _T95[degrees_of_freedom]
    return 1.96


def describe(values: Iterable[float]) -> dict[str, float | int]:
    samples = _finite(values)
    n = len(samples)
    mean = statistics.fmean(samples)
    stddev = statistics.stdev(samples) if n > 1 else 0.0
    margin = (
        student_t_critical_95(n - 1) * stddev / math.sqrt(n)
        if n > 1
        else 0.0
    )
    return {
        "n": n,
        "mean": mean,
        "stddev": stddev,
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "minimum": min(samples),
        "maximum": max(samples),
    }
