"""Tests for the frame-capture recorder's concat-list assembly."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "record_demo_frames",
    (Path(__file__).parent.parent / "src") / "scripts" / "record_demo_frames.py",
)
rdf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rdf)


def test_concat_list_durations_follow_timestamps(tmp_path):
    f1, f2, f3 = (tmp_path / n for n in ("a.png", "b.png", "c.png"))
    frames = [(10.0, f1), (10.5, f2), (12.0, f3)]
    out = tmp_path / "concat.txt"
    rdf.write_concat_list(frames, out, tail=2.0)
    lines = out.read_text().splitlines()
    assert lines[0] == "ffconcat version 1.0"
    assert lines[1] == f"file '{f1.as_posix()}'"
    assert lines[2] == "duration 0.500000"
    assert lines[4] == "duration 1.500000"
    assert lines[6] == "duration 2.000000"          # tail on last frame
    assert lines[7] == f"file '{f3.as_posix()}'"    # repeated last frame


def test_concat_list_clamps_zero_gaps(tmp_path):
    f1, f2 = tmp_path / "a.png", tmp_path / "b.png"
    frames = [(5.0, f1), (5.0, f2)]                  # identical timestamps
    out = tmp_path / "concat.txt"
    rdf.write_concat_list(frames, out)
    durations = [float(l.split()[1]) for l in out.read_text().splitlines()
                 if l.startswith("duration")]
    assert all(d >= 1 / 60 for d in durations)


def test_find_ffmpeg_available():
    assert rdf.find_ffmpeg() is not None
