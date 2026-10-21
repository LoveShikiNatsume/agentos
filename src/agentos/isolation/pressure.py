"""PSI (Pressure Stall Information) parser and pressure level utilities.

Parses Linux /proc/pressure/* files and computes pressure levels for
CPU, memory, and IO resources.  Used by the kernel adapter to feed
pressure data into the scheduler feedback loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PSI file parsing
# ---------------------------------------------------------------------------

def parse_psi(text: str) -> dict[str, dict[str, float | int]]:
    """Parse a Linux PSI file (e.g. /proc/pressure/cpu).

    Returns a dict keyed by ``"some"`` and/or ``"full"``.  Each value is a
    dict with keys ``avg10``, ``avg60``, ``avg300``, ``total``.

    Unknown lines, blank lines, and malformed numeric values are silently
    skipped so the caller never sees an exception for partial / corrupt
    files.
    """
    result: dict[str, dict[str, float | int]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected format: "some avg10=X avg60=Y avg300=Z total=T"
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        if kind not in ("some", "full"):
            continue
        entry: dict[str, float | int] = {}
        for token in parts[1:]:
            if "=" not in token:
                continue
            key, _, value_str = token.partition("=")
            try:
                value = float(value_str) if "." in value_str else int(value_str)
            except (ValueError, TypeError):
                continue
            entry[key] = value
        if entry:
            result[kind] = entry
    return result


def read_psi_avg10_with_status(path: str | Path) -> tuple[float, bool]:
    """Read ``some.avg10`` from a PSI file, returning (value, available).

    Returns ``(0.0, False)`` when the file does not exist, is not readable,
    or has no valid ``some.avg10`` field.  Returns ``(value, True)`` when
    the file is valid — even if the value is ``0.0`` (legitimate zero).
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, TypeError):
        return 0.0, False
    parsed = parse_psi(text)
    some = parsed.get("some", {})
    raw = some.get("avg10")
    if raw is None:
        return 0.0, False
    try:
        return float(raw), True
    except (TypeError, ValueError):
        return 0.0, False


def read_psi_avg10(path: str | Path) -> float:
    """Read ``some.avg10`` from a PSI file.

    Returns ``0.0`` when the file does not exist, is not readable, or
    contains corrupt data.
    """
    value, _available = read_psi_avg10_with_status(path)
    return value


# ---------------------------------------------------------------------------
# Pressure level thresholds
# ---------------------------------------------------------------------------

def pressure_level_from_psi(
    cpu: float,
    memory: float,
    io: float,
) -> str:
    """Derive a pressure level from three PSI ``some.avg10`` values.

    Thresholds (applied to the maximum of the three):
    - ``>= 50`` → ``"overloaded"``
    - ``>= 20`` → ``"critical"``
    - ``>= 10`` → ``"warning"``
    - otherwise → ``"normal"``
    """
    peak = max(cpu, memory, io)
    if peak >= 50:
        return "overloaded"
    if peak >= 20:
        return "critical"
    if peak >= 10:
        return "warning"
    return "normal"


_LEVEL_SEVERITY: dict[str, int] = {
    "normal": 0,
    "warning": 1,
    "critical": 2,
    "overloaded": 3,
}


def max_pressure_level(*levels: str) -> str:
    """Return the most severe pressure level among the arguments.

    Raises ``ValueError`` if any level is not one of the four valid values.
    """
    best = "normal"
    best_sev = 0
    for level in levels:
        sev = _LEVEL_SEVERITY.get(level)
        if sev is None:
            raise ValueError(f"invalid pressure level: {level!r}")
        if sev > best_sev:
            best = level
            best_sev = sev
    return best
