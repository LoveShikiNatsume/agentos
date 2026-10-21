#!/usr/bin/env python3
"""Validate the deterministic offline finals-demo output bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "summary.json",
    "report.md",
    "requests.jsonl",
    "events.jsonl",
    "dashboard.html",
)
EXPECTED_CASES = {"all_shared", "selected", "isolation", "retry"}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{number} 不是 JSON 对象")
        rows.append(value)
    return rows


def validate(output: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (output / name).is_file():
            errors.append(f"缺少 {name}")
    if errors:
        return errors

    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"summary.json 无法解析：{exc}"]

    if summary.get("schema") != "agentos_finals_demo.v1":
        errors.append("summary.schema 不是 agentos_finals_demo.v1")
    if summary.get("passed") is not True:
        errors.append("summary.passed 不为 true")
    if summary.get("backend") != "mock":
        errors.append("演示后端不是 mock")
    if summary.get("synthetic_inputs") is not True:
        errors.append("未标记为合成输入")
    if summary.get("performance_benchmark") is not False:
        errors.append("性能基准标记不为 false")
    if summary.get("kernel_scheduler_enabled") is not False:
        errors.append("内核调度器标记不为 false")

    try:
        requests = read_jsonl(output / "requests.jsonl")
        events = read_jsonl(output / "events.jsonl")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"JSONL 无法解析：{exc}")
    else:
        if len(requests) != 6:
            errors.append(f"请求记录数量为 {len(requests)}，预期为 6")
        if any(item.get("model") != "mock" for item in requests):
            errors.append("请求记录中存在非 mock 模型标识")
        cases = {item.get("data", {}).get("demo_case") for item in events}
        missing_cases = EXPECTED_CASES - cases
        if missing_cases:
            errors.append(f"事件流缺少演示案例：{sorted(missing_cases)}")

    if (output / "dashboard.html").stat().st_size <= 1000:
        errors.append("dashboard.html 内容过短")
    if "单分支" not in (output / "report.md").read_text(encoding="utf-8"):
        errors.append("report.md 未包含故障隔离说明")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 AgentOS 离线演示输出")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.output)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[PASS] 离线演示输出包含完整且边界明确的验证材料")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
