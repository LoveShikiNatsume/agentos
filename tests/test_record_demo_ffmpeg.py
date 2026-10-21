"""find_ffmpeg falls back to imageio-ffmpeg when ffmpeg is not on PATH."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "record_demo",
    (Path(__file__).parent.parent / "src") / "scripts" / "record_demo.py",
)
record_demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(record_demo)


def test_prefers_path_ffmpeg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert record_demo.find_ffmpeg() == "/usr/bin/ffmpeg"


def test_falls_back_to_imageio(monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    exe = record_demo.find_ffmpeg()
    assert exe is not None and "imageio_ffmpeg" in exe
