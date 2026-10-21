from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from agentos.events.log import EventLog
from agentos.events.types import RuntimeEvent

EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".agentos", "runs", "node_modules"}


class Records(EventLog):
    """One append-only event stream, including Runtime events, for both profiles."""
    def __init__(self, output: Path, run_id: str, secrets=(), listener=None):
        super().__init__(output / "events.jsonl")
        self.output, self.run_id, self.listener = output, run_id, listener
        self.secrets = tuple(s for s in secrets if s)

    def clean(self, value):
        raw = json.dumps(value, ensure_ascii=False, default=str)
        for secret in self.secrets:
            raw = raw.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
        return json.loads(raw)

    def write(self, name, value):
        (self.output / name).write_text(json.dumps(self.clean(value), ensure_ascii=False, indent=2) + "\n")

    def append(self, event):
        data = self.clean(event.to_dict())
        data["data"]["run_id"] = self.run_id
        safe = RuntimeEvent.from_dict(data)
        super().append(safe)
        if self.listener:
            self.listener(safe.to_dict())

    def emit(self, kind, task_id=None, **data):
        self.append(RuntimeEvent(event_type=kind, task_id=task_id,
                                 agent_id=f"agent-{task_id}" if task_id else None,
                                 trace_id=self.run_id, data=data))


def inventory(root: Path, excluded_path: Path | None = None) -> dict:
    result = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not Path(base, d).is_symlink()
                         and (excluded_path is None or Path(base, d).resolve() != excluded_path))
        for name in sorted(files):
            path = Path(base, name)
            if path.is_symlink() or not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def git_state(root: Path) -> dict:
    def run(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else None
    return {"head": (run("rev-parse", "HEAD") or "").strip() or None,
            "status": run("status", "--porcelain"), "diff": run("diff", "--binary", "HEAD")}


def usage_summary(calls: list[dict]) -> dict:
    successful = [c for c in calls if c["status"] == "completed"]
    usable = [c for c in successful if isinstance(c.get("usage"), dict)
              and isinstance(c["usage"].get("prompt_tokens"), int)
              and isinstance(c["usage"].get("completion_tokens"), int)]
    complete = bool(successful) and len(usable) == len(calls)
    cached = []
    misses = []
    for call in successful:
        raw = call.get("usage")
        raw = raw if isinstance(raw, dict) else {}
        details = raw.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}
        value = raw.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            cached.append(value)
        reported_miss = raw.get("prompt_cache_miss_tokens")
        if isinstance(reported_miss, int) and not isinstance(reported_miss, bool) and reported_miss >= 0:
            misses.append(reported_miss)
    return {"attempts": len(calls), "completed_calls": len(successful),
            "failed_attempts": sum(c["status"] == "failed" for c in calls),
            "retries": sum(c["attempt"] > 1 for c in calls),
            "usage_complete": complete, "missing_usage_calls": len(successful) - len(usable),
            "usage_unknown_attempts": len(calls) - len(usable),
            "cached_tokens": sum(cached) if cached and len(cached) == len(calls) else None,
            "cache_hit_tokens": sum(cached) if cached and len(cached) == len(calls) else None,
            "cache_miss_tokens": sum(misses) if misses and len(misses) == len(calls) else None,
            "cache_hit_source": "provider_usage" if cached and len(cached) == len(calls) else "incomplete_or_unavailable",
            "cache_miss_source": "provider_usage" if misses and len(misses) == len(calls) else "incomplete_or_unavailable",
            "known_cache_miss_tokens": sum(misses), "missing_cache_miss_usage_calls": len(calls) - len(misses),
            "known_cached_tokens": sum(cached), "missing_cache_usage_calls": len(calls) - len(cached),
            "prompt_tokens": sum(c["usage"]["prompt_tokens"] for c in usable) if complete else None,
            "completion_tokens": sum(c["usage"]["completion_tokens"] for c in usable) if complete else None,
            "known_prompt_tokens": sum(c["usage"]["prompt_tokens"] for c in usable),
            "known_completion_tokens": sum(c["usage"]["completion_tokens"] for c in usable),
            "failed_attempt_usage_unknown": any(c["status"] == "failed" for c in calls)}
