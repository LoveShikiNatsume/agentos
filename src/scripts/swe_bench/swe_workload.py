"""SWE-bench task -> 6-agent software pipeline with role-based context scoping.

The honest, attributable token-saving mechanism lives here:

  native  (naive framework usage): EVERY agent is handed the full repository
          code, every time.
  runtime (framework + our Runtime): only the *code-consuming* agents
          (localizer / analyzer / patcher) receive the raw repo; the
          *synthesis* agents (tester / reviewer / summarizer) work from the
          distilled upstream findings instead of re-reading the whole repo.

So the token delta is exactly "raw repo code NOT re-sent to the 3 synthesis
agents" — transparent and defensible, not a tuned eviction budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SYSTEM = "You are a senior software engineer working on a real GitHub issue. Be concise and precise."

ROLE_ORDER = ["localizer", "analyzer", "patcher", "tester", "reviewer", "summarizer"]

DEPS: dict[str, list[str]] = {
    "localizer": [],
    "analyzer": ["localizer"],
    "patcher": ["analyzer"],
    "tester": ["patcher"],
    "reviewer": ["patcher"],
    "summarizer": ["tester", "reviewer"],
}

CODE_ROLES = {"localizer", "analyzer", "patcher"}   # need the raw repository
SYNTH_ROLES = {"tester", "reviewer", "summarizer"}  # work from distilled findings
FAULT_ROLE = "patcher"                               # where we inject a real crash

ROLE_INSTRUCTION: dict[str, str] = {
    "localizer": "List the files/functions most relevant to fixing this issue.",
    "analyzer": "Explain the root cause of this issue based on the code.",
    "patcher": "Propose a minimal code change (short diff) that resolves the issue.",
    "tester": "Describe the tests that would verify the fix.",
    "reviewer": "Review the proposed fix for correctness and side effects.",
    "summarizer": "Summarize localization, fix, tests and review into a short report.",
}


@dataclass
class SweTask:
    instance_id: str
    repo: str
    issue: str
    code: str
    est_context_tokens: int


def parse_task(raw: dict) -> SweTask:
    text = raw.get("shared_context", "") or ""
    m_issue = re.search(r"<issue>(.*?)</issue>", text, re.S)
    issue = (m_issue.group(1).strip() if m_issue
             else raw.get("problem_statement", "") or text[:2000])
    m_code = re.search(r"<code>(.*?)</code>", text, re.S)
    code = m_code.group(1).strip() if m_code else text
    return SweTask(
        instance_id=raw["instance_id"], repo=raw["repo"],
        issue=issue, code=code,
        est_context_tokens=raw.get("est_context_tokens", len(text) // 4),
    )


def build_user_prompt(role: str, mode: str, task: SweTask, findings: str) -> str:
    """Assemble the user prompt for one agent under a given mode.

    mode == 'native'  -> always include full repo code (naive).
    mode == 'runtime' -> include repo code only for CODE_ROLES; synthesis agents
                         get the distilled findings instead.
    """
    parts = [f"# Issue\n{task.issue}"]

    include_code = (mode == "native") or (role in CODE_ROLES)
    if include_code:
        parts.append(f"# Repository code\n{task.code}")

    # Everyone past the first agent sees upstream findings (normal in both arms).
    if findings and role != "localizer":
        parts.append(f"# Upstream findings\n{findings}")

    parts.append(f"# Your task as the {role}\n{ROLE_INSTRUCTION[role]}")
    return "\n\n".join(parts)


def messages_for(role: str, mode: str, task: SweTask, findings: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(role, mode, task, findings)},
    ]
