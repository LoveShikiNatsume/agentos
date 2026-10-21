"""Bounded, read-only workspace search; literal content queries, no subprocess."""
import fnmatch
import json
import os
import stat
import time
from pathlib import Path

from agentos.execution.records import EXCLUDED

MAX_OUTPUT_BYTES = 32768
MAX_FILE_BYTES = 1024 * 1024
MAX_SCAN_BYTES = 32 * 1024 * 1024
MAX_VISITED_FILES = 5000
MAX_SCAN_SECONDS = 2.0
MAX_EXCERPT_CHARACTERS = 300


def search_files(root: Path, query: str, path: str = ".", file_glob: str = "*",
                 max_results: int = 50, case_sensitive: bool = True):
    root = root.resolve()
    location = (root / path).resolve()
    if not location.is_relative_to(root):
        raise ValueError("search path escapes working directory")
    if not location.exists():
        raise ValueError("search path does not exist")
    if not isinstance(query, str) or not 1 <= len(query) <= 512:
        raise ValueError("query must be a nonempty literal string of at most 512 characters")
    if not isinstance(file_glob, str) or not 1 <= len(file_glob) <= 256:
        raise ValueError("file_glob must contain 1 to 256 characters")
    if not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    matches, reasons = [], []
    skipped = {"symlinks": 0, "large_files": 0, "binary_or_non_utf8": 0, "unreadable": 0, "special_files": 0}
    visited = searched = scanned_bytes = 0
    started = time.monotonic()
    needle = query if case_sensitive else query.casefold()

    def candidates():
        if location.is_file():
            yield location
            return
        for base, dirs, files in os.walk(location, followlinks=False):
            if time.monotonic() - started > MAX_SCAN_SECONDS:
                reasons.append("scan_time_limit")
                return
            kept = []
            for name in sorted(dirs):
                directory = Path(base) / name
                if directory.is_symlink():
                    skipped["symlinks"] += 1
                elif name not in EXCLUDED:
                    kept.append(name)
            dirs[:] = kept
            for name in sorted(files):
                yield Path(base) / name

    for candidate in candidates():
        if time.monotonic() - started > MAX_SCAN_SECONDS:
            reasons.append("scan_time_limit")
            break
        if visited >= MAX_VISITED_FILES:
            reasons.append("visited_file_limit")
            break
        visited += 1
        relative = str(candidate.relative_to(root))
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            skipped["symlinks"] += 1
            continue
        if not fnmatch.fnmatchcase(relative, file_glob):
            continue
        try:
            info = candidate.stat()
            if not stat.S_ISREG(info.st_mode):
                skipped["special_files"] += 1
                continue
            if info.st_size > MAX_FILE_BYTES:
                skipped["large_files"] += 1
                continue
            # O_NOFOLLOW also rejects a final-component link swapped after the
            # check. Inter-task workspace writes are controlled by the harness.
            descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    skipped["special_files"] += 1
                    continue
                content = stream.read(MAX_FILE_BYTES + 1)
        except OSError:
            skipped["unreadable"] += 1
            continue
        if len(content) > MAX_FILE_BYTES:
            skipped["large_files"] += 1
            continue
        if scanned_bytes + len(content) > MAX_SCAN_BYTES:
            reasons.append("scan_byte_limit")
            break
        scanned_bytes += len(content)
        try:
            if b"\x00" in content:
                raise UnicodeError("binary data")
            text = content.decode("utf-8")
        except UnicodeError:
            skipped["binary_or_non_utf8"] += 1
            continue
        searched += 1
        for number, line in enumerate(text.splitlines(), 1):
            position = (line if case_sensitive else line.casefold()).find(needle)
            if position < 0:
                continue
            if not case_sensitive:
                folded_offset = 0
                for original_offset, character in enumerate(line):
                    if folded_offset + len(character.casefold()) > position:
                        position = original_offset
                        break
                    folded_offset += len(character.casefold())
            start = max(0, position - MAX_EXCERPT_CHARACTERS // 3)
            excerpt = line[start:start + MAX_EXCERPT_CHARACTERS]
            match = {"path": relative, "line": number, "text": excerpt,
                     "excerpt_truncated": len(excerpt) != len(line)}
            if len(json.dumps(matches + [match], ensure_ascii=False).encode()) > MAX_OUTPUT_BYTES - 1024:
                reasons.append("output_byte_limit")
                break
            matches.append(match)
            if len(matches) >= max_results:
                reasons.append("result_limit_reached")
                break
        if reasons:
            break
    return {"matches": matches, "truncated": bool(reasons), "truncation_reasons": reasons,
            "visited_files": visited, "searched_files": searched, "scanned_bytes": scanned_bytes,
            "skipped": skipped, "limits": {"max_results": max_results, "output_bytes": MAX_OUTPUT_BYTES,
                "file_bytes": MAX_FILE_BYTES, "scan_bytes": MAX_SCAN_BYTES,
                "visited_files": MAX_VISITED_FILES, "scan_seconds": MAX_SCAN_SECONDS,
                "excerpt_characters": MAX_EXCERPT_CHARACTERS}}
