"""Configuration for the shared CLI/App execution service (no implicit backend)."""
from __future__ import annotations

import os
import math
import re
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

DEFAULTS = dict(harness="langgraph", profile="runtime", max_parallel=1, max_steps=24, max_tasks=6,
                max_tokens=None, temperature=0.0, retries=1, timeout=120.0,
                command_timeout=60.0, run_timeout=1800.0, thinking=None, reasoning_effort=None, response_format=None, api_key_env="OPENAI_API_KEY")
DEFAULTS.update(context_optimization="legacy", command_sandbox="none", command_read_only_paths=(), command_network=False)
FIELDS = set(DEFAULTS) | {"model", "base_url", "output"}


def resolve_config(cwd: Path, overrides: dict | None = None, *, user_path: Path | None = None) -> dict:
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f"working directory does not exist: {cwd}")
    user_path = user_path or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "agentos/config.toml"
    config = dict(DEFAULTS)
    sources = []
    for path in (user_path, cwd / ".agentos.toml"):
        if path.exists():
            data = tomllib.loads(path.read_text())
            unknown = set(data) - FIELDS
            if unknown:
                raise ValueError(f"unsupported configuration keys in {path}: {sorted(unknown)}; use api_key_env for credentials")
            config.update(data)
            sources.append(str(path))
    unknown = set(overrides or {}) - FIELDS
    if unknown:
        raise ValueError(f"unsupported configuration overrides: {sorted(unknown)}")
    config.update(overrides or {})
    config.update(cwd=str(cwd), config_sources=sources)
    if config["profile"] not in {"native", "runtime"}:
        raise ValueError("profile must be native or runtime")
    if not config.get("model") or not config.get("base_url"):
        raise ValueError("real model configuration required: set model and base_url in .agentos.toml or --model/--base-url; no mock fallback")
    for key in ("model", "base_url", "harness", "api_key_env"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config["api_key_env"]):
        raise ValueError("api_key_env must name an environment variable, not contain a credential")
    url = urlsplit(config["base_url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query or fragment")
    for key in ("max_steps", "max_tasks", "max_parallel"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_tokens"] == "provider_default":
        config["max_tokens"] = None
    if config["max_tokens"] is not None and (not isinstance(config["max_tokens"], int) or isinstance(config["max_tokens"], bool) or config["max_tokens"] < 1):
        raise ValueError("max_tokens must be positive or provider_default/None")
    if not isinstance(config["retries"], int) or isinstance(config["retries"], bool) or not 0 <= config["retries"] <= 10:
        raise ValueError("retries must be an integer from 0 to 10")
    for key in ("timeout", "command_timeout", "run_timeout"):
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not isinstance(config["temperature"], (int, float)) or not math.isfinite(config["temperature"]) or not 0 <= config["temperature"] <= 2:
        raise ValueError("temperature must be a finite number from 0 to 2")
    if config["reasoning_effort"] not in {None, "none", "low", "high", "max"}:
        raise ValueError("reasoning_effort must be none, low, high, max, or unset")
    if config["response_format"] not in {None, "text", "json_object"}:
        raise ValueError("response_format must be text, json_object, or unset")
    if config["thinking"] not in {None, "enabled", "disabled"}:
        raise ValueError("thinking must be enabled, disabled, or unset")
    if config["thinking"] == "enabled" and config["reasoning_effort"] == "none":
        raise ValueError("enabled thinking conflicts with reasoning_effort=none")
    if config["context_optimization"] not in {"legacy", "off", "on"}:
        raise ValueError("context_optimization must be legacy, off, or on")
    if config["profile"] != "runtime" and config["context_optimization"] != "legacy":
        raise ValueError("the combined context optimization switch requires profile=runtime")
    if config["command_sandbox"] not in {"none", "bubblewrap"}:
        raise ValueError("command_sandbox must be none or bubblewrap")
    if not isinstance(config["command_network"], bool):
        raise ValueError("command_network must be a boolean")
    mounts = config["command_read_only_paths"]
    if not isinstance(mounts, (list, tuple)) or not all(isinstance(p, str) and p for p in mounts):
        raise ValueError("command_read_only_paths must be a list of absolute dependency paths")
    if any(not Path(p).expanduser().is_absolute() for p in mounts):
        raise ValueError("dependency mounts must be absolute so batch workspaces retain the same dependencies")
    config["command_read_only_paths"] = [str(Path(p).expanduser().resolve(strict=True)) for p in mounts]
    config["context_window"] = "provider_default"
    config["generation_limit"] = config["max_tokens"] if config["max_tokens"] is not None else "provider_default"
    return config
