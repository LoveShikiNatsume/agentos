"""Locate logical project inputs after the submission directory migration."""
from __future__ import annotations

from pathlib import Path, PurePosixPath


def _split_layout(root: Path) -> bool:
    return (
        root.name == "src"
        and (root / "pyproject.toml").is_file()
        and (root / "agentos").is_dir()
        and (root.parent / "tests").is_dir()
        and (root.parent / "docs").is_dir()
    )


def project_scope(source_root: Path) -> Path:
    root = Path(source_root).resolve()
    return root.parent if _split_layout(root) else root


def project_file(source_root: Path, relative_path: str) -> Path:
    """Keep existing logical fixture names while locating their moved files."""
    root = Path(source_root).resolve()
    relative = PurePosixPath(relative_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("project input must stay within its declared repository")
    if not _split_layout(root):
        return root / relative
    repo = root.parent
    if relative.as_posix() == "README.md":
        return repo / "README.md"
    mappings = (
        ("runs/demo_video", repo / "demo/replay"),
        ("docs", repo / "docs/technical"),
        ("tests", repo / "tests"),
        ("examples", repo / "demo/examples"),
        ("runs", repo / "tests/测试及验证材料/evidence"),
    )
    for prefix, destination in mappings:
        prefix_path = PurePosixPath(prefix)
        if relative == prefix_path or prefix_path in relative.parents:
            return destination / relative.relative_to(prefix_path)
    return root / relative
