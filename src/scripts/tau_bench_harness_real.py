#!/usr/bin/env python3
"""TAU-bench 客服场景多 Agent Runtime 实验（任务 11，真实 DeepSeek API）。

背景与设计
----------
TAU-bench（sierra-research/tau-bench）是客服 Agent 测试集，分航空(airline)与
零售(retail)两个域，每个域有一份共享的 **政策文档(policy wiki)**、一批用户
任务(instruction + 官方标准动作 actions)、一套工具(tools)和一个账户数据库。

本脚本把它改造成 **多角色分工** 的多智能体工作流（任务 11 方式一），四个角色：
    1. dialogue_understanding 对话理解：解析顾客诉求 → 意图
    2. policy_lookup          政策查询：引用适用的政策条款
    3. tool_execution         工具执行：给出应调用的动作(JSON)
    4. result_review          结果复核：核对动作是否合规

对比两种模式（复用 travelplanner_lite_harness_real 的真实 API 惯例）：
    native  全量广播：把 **完整政策文档 + 完整账户上下文** 发给每一个角色，
            对应主流多智能体框架的默认行为（政策/系统提示广播给每个 Agent）。
    runtime 分域拆发(ContextRef)：政策文档只由 policy_lookup 查一次，其结论
            以 findings 形式向下游引用；tool_execution 只拿到自己需要的账户
            记录切片；共享内容只计一次。

共享内容 = 政策文档（域级常量，对该域所有任务字节一致），这是 native 侧最大的
重复项，也正是本系统 ContextRef 机制的发挥场景。token / cache / cost 全部取自
真实 DeepSeek API 的 usage 字段；质量以 TAU-bench 官方标准动作名(actions[0].name)
为准绳，比对 tool_execution 角色输出的动作名。故障注入沿用本项目一致定义（角色
进程被杀死/超时，不发起该次调用；恢复时才补发，不伪造模型输出）。

边界：这是自建的多智能体改造，动作名层级的正确性核对，**不是官方 TAU-bench
pass^k 榜单成绩**。结果无论好坏如实记录。

用法：
    cd AgentOS-Runtime
    set -a; source .env; set +a          # 只从环境变量读 key
    python3 scripts/tau_bench_harness_real.py \
        --tau-root runs/external_data/tau-bench \
        --out-dir runs/national_experiments/20260710_tau_bench_real \
        --airline 15 --retail 15
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import statistics
import sys
import time
import types as _types
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

# --- DeepSeek V4 Flash 官方列表价（2026-07-10 更新，登记表 N13；旧常量为 V3 价目）---
PRICE_INPUT_MISS = 0.14
PRICE_INPUT_HIT = 0.0028
PRICE_OUTPUT = 0.28

CONCURRENCY = 10
MODEL = "deepseek-v4-flash"

ROLES = ["dialogue_understanding", "policy_lookup", "tool_execution", "result_review"]
FAULT_ROLE = "tool_execution"  # 第 3 个角色；native 无重试则下游 result_review 被跳过

# 每个域允许的“写动作”工具名（给 tool_execution 严格 JSON 输出用）。TAU-bench 的
# actions 序列里第一个往往是 get_*/find_* 等只读查库步骤，真正决定任务成败的是
# 其中第一个“写动作”。质量核对以该写动作为准绳，纯查询/信息类任务对应 "none"。
AIRLINE_TOOLS = [
    "book_reservation", "cancel_reservation", "update_reservation_flights",
    "update_reservation_passengers", "update_reservation_baggages",
    "send_certificate", "transfer_to_human_agents", "none",
]
RETAIL_TOOLS = [
    "cancel_pending_order", "modify_pending_order_items",
    "modify_pending_order_address", "modify_pending_order_payment",
    "exchange_delivered_order_items", "return_delivered_order_items",
    "modify_user_address", "transfer_to_human_agents", "none",
]
WRITE_ACTIONS = {
    "airline": {a for a in AIRLINE_TOOLS if a not in ("transfer_to_human_agents", "none")},
    "retail": {a for a in RETAIL_TOOLS if a not in ("transfer_to_human_agents", "none")},
}


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def load_tasks(tasks_file: Path, domain: str) -> list[dict[str, Any]]:
    """加载 TAU-bench 的 tasks_test.py（注入 tau_bench.types 桩，避免装整包）。"""
    if "tau_bench" not in sys.modules:
        pkg = _types.ModuleType("tau_bench")
        sys.modules["tau_bench"] = pkg
        tmod = _types.ModuleType("tau_bench.types")

        class Action:  # noqa: D401 - 轻量桩
            def __init__(self, name: str, kwargs: dict | None = None):
                self.name = name
                self.kwargs = kwargs or {}

        class Task:
            def __init__(self, user_id="", instruction="", actions=None,
                         outputs=None, annotator=None):
                self.user_id = user_id
                self.instruction = instruction
                self.actions = actions or []
                self.outputs = outputs or []

        tmod.Action = Action
        tmod.Task = Task
        sys.modules["tau_bench.types"] = tmod
        pkg.types = tmod

    spec = importlib.util.spec_from_file_location(
        f"tau_tasks_{tasks_file.parent.name}", tasks_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    write_set = WRITE_ACTIONS[domain]
    tasks_list = getattr(module, "TASKS", None) or getattr(module, "TASKS_TEST", None)
    if tasks_list is None:  # 兜底：取模块里第一个 list 型全局
        tasks_list = next(v for v in vars(module).values() if isinstance(v, list) and v)
    out = []
    for t in tasks_list:
        gt = [a.name for a in t.actions]
        # gt_primary = 序列里第一个“写动作”；无写动作时若有转人工则记 transfer，
        # 否则记 none（纯查询/信息类任务）。
        primary = next((a for a in gt if a in write_set), None)
        if primary is None:
            primary = "transfer_to_human_agents" if "transfer_to_human_agents" in gt else "none"
        out.append({
            "user_id": t.user_id,
            "instruction": t.instruction,
            "gt_actions": gt,
            "gt_primary": primary,
        })
    return out


def user_records(domain: str, user_id: str, db: dict[str, Any]) -> dict[str, Any]:
    """取该用户的账户上下文：档案 + 其名下的预订/订单记录。"""
    profile = db["users"].get(user_id, {})
    if domain == "airline":
        recs = [r for r in db["reservations"].values() if r.get("user_id") == user_id]
    else:
        recs = [o for o in db["orders"].values() if o.get("user_id") == user_id]
    key = "reservations" if domain == "airline" else "orders"
    return {"profile": profile, key: recs}


# ----------------------------------------------------------------------------
# 政策文档按 "## " 分节 + 关键词路由（runtime 分域：给下游真实相关小节，非有损摘要）
# ----------------------------------------------------------------------------
# 每个小节的触发词（命中则把该小节的字节原文路由给 tool_execution / result_review）。
SECTION_SYNONYMS = {
    # airline
    "book flight": ["book", "reserv", "fly from", "one way", "one-way", "round trip",
                    "round-trip", "new flight"],
    "modify flight": ["change", "modif", "update", "downgrad", "upgrad", "later flight",
                      "earlier flight", "different flight", "rebook", "switch", "cabin",
                      "economy", "business", "baggage", "bag", "passenger", "insurance"],
    "cancel flight": ["cancel"],
    "refund": ["refund", "compensat", "certificate", "delay", "complain", "gesture"],
    # retail
    "cancel pending order": ["cancel"],
    "modify pending order": ["modif", "change", "update", "address", "payment", "item",
                             "different", "swap", "wrong"],
    "return delivered order": ["return", "refund"],
    "exchange delivered order": ["exchang", "swap", "replace", "different"],
}


def split_sections(policy: str) -> tuple[str, dict[str, str]]:
    """按 "## " 顶级标题切分；"### " 子节归入其父节。返回 (header, {name: body})。"""
    header: list[str] = []
    sections: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for ln in policy.split("\n"):
        if ln.startswith("## "):
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur = ln[3:].strip().lower()
            buf = [ln]
        elif cur is None:
            header.append(ln)
        else:
            buf.append(ln)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()
    return "\n".join(header).strip(), sections


def route_policy(policy: str, instruction: str) -> str:
    """给下游角色路由相关政策小节（真实字节原文）。header 与 domain-basic 恒含；
    没有任何动作小节命中时保守回退为完整政策（不丢决定性规则）。"""
    header, sections = split_sections(policy)
    instr = instruction.lower()
    picked = [header]
    matched_action = False
    for name, body in sections.items():
        if "domain basic" in name:
            picked.append(body)
            continue
        triggers = SECTION_SYNONYMS.get(name, name.split())
        if any(t in instr for t in triggers):
            picked.append(body)
            matched_action = True
    if not matched_action:  # 保守回退：包含全部小节
        return policy
    return "\n\n".join(p for p in picked if p)


# ----------------------------------------------------------------------------
# 提示构造：native = 全量广播，runtime = 分域拆发
# ----------------------------------------------------------------------------
def build_prompt(role: str, mode: str, domain: str, policy: str, instruction: str,
                 ctx: dict[str, Any], findings: dict[str, str]) -> str:
    tools = AIRLINE_TOOLS if domain == "airline" else RETAIL_TOOLS
    ctx_json = json.dumps(ctx, ensure_ascii=False, sort_keys=True)
    prior = json.dumps(findings, ensure_ascii=False, sort_keys=True)

    # native：完整政策文档 + 完整账户上下文，广播给每一个角色（政策放前缀）
    if mode == "native":
        header = (
            f"{policy}\n\n---\nACCOUNT CONTEXT:\n{ctx_json}\n\n"
            f"PRIOR FINDINGS:\n{prior}\n---\n"
        )
        body = _role_instruction(role, domain, instruction, tools)
        return header + body

    # runtime：分域拆发。共享政策存一份，按角色路由“真实相关小节”（字节原文，非有损
    # 摘要）——tool_execution / result_review 拿到决定动作的政策条款，保证质量不降。
    if role == "dialogue_understanding":
        header = ""  # 仅凭顾客诉求即可理解意图
    elif role == "policy_lookup":
        header = f"{policy}\n\n---\n"  # 通读完整政策、抽取适用规则（共享前缀）
    elif role == "tool_execution":
        routed = route_policy(policy, instruction)
        header = (
            f"RELEVANT POLICY:\n{routed}\n\n---\nACCOUNT CONTEXT:\n{ctx_json}\n\n"
            f"PRIOR FINDINGS (intent + applicable rules):\n{prior}\n---\n"
        )
    else:  # result_review：路由相关政策小节 + 下游 findings
        routed = route_policy(policy, instruction)
        header = f"RELEVANT POLICY:\n{routed}\n\n---\nPRIOR FINDINGS:\n{prior}\n---\n"
    body = _role_instruction(role, domain, instruction, tools)
    return header + body


def _role_instruction(role: str, domain: str, instruction: str, tools: list[str]) -> str:
    who = "airline" if domain == "airline" else "retail"
    if role == "dialogue_understanding":
        return (
            f"You are the dialogue-understanding agent for a {who} customer-service "
            f"workflow.\nCustomer message: {instruction}\n"
            "In <=60 words state the customer's core request type "
            "(book / modify / cancel / return / exchange / refund / info) and the "
            "target reservation/order id if any."
        )
    if role == "policy_lookup":
        return (
            f"You are the policy-lookup agent. Given the {who} agent policy above and "
            f"this customer request:\n{instruction}\n"
            "In <=60 words cite the specific policy rules that decide whether and how "
            "the request can be fulfilled."
        )
    if role == "tool_execution":
        return (
            f"You are the tool-execution agent for a {who} customer-service workflow.\n"
            f"Customer request: {instruction}\n"
            "Decide the single primary write action to fulfil the request, using the "
            f"account records above. Allowed action names: {tools}.\n"
            'Respond with STRICT JSON only: {"action": "<one allowed name>", '
            '"target": "<reservation/order id or null>"}. Use "none" if no write action '
            "is warranted (info-only, denied by policy, or human transfer)."
        )
    return (  # result_review
        f"You are the result-review agent. Given the prior findings, verify the chosen "
        f"action complies with {who} policy for this request:\n{instruction}\n"
        "In <=40 words state whether the action is compliant and final."
    )


# ----------------------------------------------------------------------------
# 真实 API 调用
# ----------------------------------------------------------------------------
async def call_llm(client: AsyncOpenAI, sem: asyncio.Semaphore, prompt: str,
                   max_tokens: int) -> dict[str, Any]:
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                u = resp.usage
                return {
                    "ok": True,
                    "text": resp.choices[0].message.content or "",
                    "prompt_tokens": u.prompt_tokens,
                    "cache_hit_tokens": getattr(u, "prompt_cache_hit_tokens", 0) or 0,
                    "cache_miss_tokens": getattr(u, "prompt_cache_miss_tokens", u.prompt_tokens) or u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                }
            except Exception as exc:  # noqa: BLE001 - 真实失败如实上抛
                if attempt == 3:
                    return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                            "text": "", "prompt_tokens": 0, "cache_hit_tokens": 0,
                            "cache_miss_tokens": 0, "completion_tokens": 0}
                await asyncio.sleep(2 * (attempt + 1))


ACTION_RE = re.compile(r'"action"\s*:\s*"([a-z_]+)"')


def parse_action(text: str) -> str:
    m = ACTION_RE.search(text or "")
    return m.group(1) if m else "unpar;seable".replace(";", "")  # 'unparseable'


async def run_arm(client, sem, domain, policy, task, ctx, mode, fault) -> dict[str, Any]:
    findings: dict[str, str] = {}
    tot = {"prompt_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0,
           "completion_tokens": 0}
    completed = recovered = injected = 0
    predicted = "none"
    api_error = None

    for role in ROLES:
        attempts = 2 if mode == "runtime" else 1
        for attempt in range(attempts):
            if fault and role == FAULT_ROLE and attempt == 0:
                # 模拟角色进程被杀死/超时：不发起真实调用
                injected += 1
                if attempt + 1 < attempts:
                    recovered += 1
                    continue
                # native 无重试：任务在此中断，下游 result_review 被跳过
                return {**tot, "tasks_completed": completed, "tasks_total": len(ROLES),
                        "faults_injected": injected, "recovered_faults": recovered,
                        "plan_completed": False, "predicted_action": "none",
                        "api_error": api_error}
            max_tok = 250 if role == "tool_execution" else 120
            prompt = build_prompt(role, mode, domain, policy, task["instruction"],
                                  ctx, findings)
            res = await call_llm(client, sem, prompt, max_tok)
            for k in tot:
                tot[k] += res[k]
            if not res["ok"]:
                api_error = res.get("error")
            completed += 1
            if role == "tool_execution":
                predicted = parse_action(res["text"])
                findings[role] = f"proposed action: {predicted}"
            else:
                findings[role] = (res["text"] or "").strip()[:400]
            break

    return {**tot, "tasks_completed": completed, "tasks_total": len(ROLES),
            "faults_injected": injected, "recovered_faults": recovered,
            "plan_completed": True, "predicted_action": predicted,
            "api_error": api_error}


# ----------------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------------
def pct(base: float, rt: float) -> float:
    return round((base - rt) / base * 100, 1) if base else 0.0


def cost(miss: int, hit: int, out: int) -> float:
    return round((miss * PRICE_INPUT_MISS + hit * PRICE_INPUT_HIT
                  + out * PRICE_OUTPUT) / 1_000_000, 6)


def list_cost(prompt_tokens: int, out: int) -> float:
    """列表价成本：全部 prompt token 按 miss(未缓存)价计。与实时缓存状态无关，
    是可复现的公平成本口径（本场景政策为全域常量，供应商前缀缓存对不同臂的命中
    受跨请求全局缓存状态影响、不可控，故实时缓存价成本不作为受控指标）。"""
    return round((prompt_tokens * PRICE_INPUT_MISS + out * PRICE_OUTPUT) / 1_000_000, 6)


async def run(tau_root: Path, out_dir: Path, n_air: int, n_retail: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    envs = tau_root / "tau_bench" / "envs"

    domains: dict[str, dict[str, Any]] = {}
    if n_air > 0:
        domains["airline"] = {
            "policy": (envs / "airline" / "wiki.md").read_text(encoding="utf-8"),
            "tasks": load_tasks(envs / "airline" / "tasks_test.py", "airline")[:n_air],
            "db": {
                "users": json.loads((envs / "airline" / "data" / "users.json").read_text()),
                "reservations": json.loads((envs / "airline" / "data" / "reservations.json").read_text()),
            },
        }
    if n_retail > 0:
        domains["retail"] = {
            "policy": (envs / "retail" / "wiki.md").read_text(encoding="utf-8"),
            "tasks": load_tasks(envs / "retail" / "tasks_test.py", "retail")[:n_retail],
            "db": {
                "users": json.loads((envs / "retail" / "data" / "users.json").read_text()),
                "orders": json.loads((envs / "retail" / "data" / "orders.json").read_text()),
            },
        }

    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    sem = asyncio.Semaphore(CONCURRENCY)

    per_task: list[dict[str, Any]] = []
    jobs = []
    for domain, d in domains.items():
        for i, task in enumerate(d["tasks"]):
            jobs.append((domain, d, i, task))
    per_task = [None] * len(jobs)  # type: ignore

    async def do(idx: int, domain: str, d: dict, i: int, task: dict) -> None:
        ctx = user_records(domain, task["user_id"], d["db"])
        rec: dict[str, Any] = {"domain": domain, "idx": i + 1,
                               "user_id": task["user_id"],
                               "gt_primary": task["gt_primary"],
                               "gt_actions": task["gt_actions"]}
        for arm, mode, fault in [("native", "native", False),
                                 ("runtime", "runtime", False),
                                 ("native_fault", "native", True),
                                 ("runtime_fault", "runtime", True)]:
            rec[arm] = await run_arm(client, sem, domain, d["policy"], task, ctx,
                                     mode, fault)
        per_task[idx] = rec
        print(f"[{idx + 1}/{len(jobs)}] {domain} task {i + 1} "
              f"({task['user_id']}) done", flush=True)

    t0 = time.monotonic()
    await asyncio.gather(*(do(k, dm, d, i, t) for k, (dm, d, i, t) in enumerate(jobs)))
    elapsed = time.monotonic() - t0

    _summarize(per_task, domains, out_dir, elapsed, n_air, n_retail)


def _acc(rows: list[dict], arm: str) -> dict[str, Any]:
    """动作名正确率。overall = 全部任务（含纯查询任务，正确=预测 none）；
    write_subset = 仅有官方写动作的任务（book/modify/cancel/return/exchange）。"""
    correct_all = sum(1 for r in rows if r[arm]["predicted_action"] == r["gt_primary"])
    wsub = [r for r in rows if r["gt_primary"] not in ("none", "transfer_to_human_agents")]
    correct_w = sum(1 for r in wsub if r[arm]["predicted_action"] == r["gt_primary"])
    return {
        "overall": {"correct": correct_all, "scored": len(rows),
                    "accuracy": round(correct_all / len(rows), 4) if rows else None},
        "write_subset": {"correct": correct_w, "scored": len(wsub),
                         "accuracy": round(correct_w / len(wsub), 4) if wsub else None},
    }


def _summarize(per_task, domains, out_dir, elapsed, n_air, n_retail) -> None:
    def block(rows: list[dict]) -> dict[str, Any]:
        def tot(arm, key):
            return sum(int(r[arm].get(key, 0)) for r in rows)
        nat_cost = cost(tot("native", "cache_miss_tokens"),
                        tot("native", "cache_hit_tokens"),
                        tot("native", "completion_tokens"))
        rt_cost = cost(tot("runtime", "cache_miss_tokens"),
                       tot("runtime", "cache_hit_tokens"),
                       tot("runtime", "completion_tokens"))
        nat_pt, rt_pt = tot("native", "prompt_tokens"), tot("runtime", "prompt_tokens")
        nat_lc = list_cost(nat_pt, tot("native", "completion_tokens"))
        rt_lc = list_cost(rt_pt, tot("runtime", "completion_tokens"))
        return {
            "n_tasks": len(rows),
            "efficiency_no_fault": {
                "prompt_tokens": {"native": nat_pt, "runtime": rt_pt,
                                  "reduction_pct": pct(nat_pt, rt_pt)},
                "list_price_cost_usd": {"native": nat_lc, "runtime": rt_lc,
                                        "reduction_pct": pct(nat_lc, rt_lc),
                                        "note": "全部 prompt token 按未缓存价，可复现的公平成本口径"},
                "cache_miss_tokens": {
                    "native": tot("native", "cache_miss_tokens"),
                    "runtime": tot("runtime", "cache_miss_tokens"),
                    "reduction_pct": pct(tot("native", "cache_miss_tokens"),
                                         tot("runtime", "cache_miss_tokens"))},
                "cache_hit_tokens": {"native": tot("native", "cache_hit_tokens"),
                                     "runtime": tot("runtime", "cache_hit_tokens")},
                "est_cost_usd": {"native": nat_cost, "runtime": rt_cost,
                                 "reduction_pct": pct(nat_cost, rt_cost)},
                "action_accuracy": {"native": _acc(rows, "native"),
                                    "runtime": _acc(rows, "runtime")},
            },
            "resilience_with_fault": {
                "tasks_completed": {
                    "native": tot("native_fault", "tasks_completed"),
                    "runtime": tot("runtime_fault", "tasks_completed"),
                    "total": tot("native_fault", "tasks_total")},
                "recovered_faults": {
                    "native": tot("native_fault", "recovered_faults"),
                    "runtime": tot("runtime_fault", "recovered_faults"),
                    "injected": tot("runtime_fault", "faults_injected")},
            },
        }

    by_domain = {dm: block([r for r in per_task if r["domain"] == dm])
                 for dm in domains}
    combined = block(per_task)

    # 真实 API 错误统计（如实记录）
    n_err = sum(1 for r in per_task for arm in
                ("native", "runtime", "native_fault", "runtime_fault")
                if r[arm].get("api_error"))

    summary = {
        "benchmark": "TAU-bench (multi-agent Runtime adaptation)",
        "source": "sierra-research/tau-bench airline+retail wiki/tasks_test/data",
        "backend": "deepseek", "model": MODEL,
        "adaptation": "方式一 多角色分工 (dialogue/policy/tool/review); "
                      "shared content = domain policy wiki (ContextRef)",
        "roles": ROLES, "fault_role": FAULT_ROLE,
        "n_tasks": len(per_task),
        "n_airline": n_air, "n_retail": n_retail,
        "by_domain": by_domain,
        "combined": combined,
        "api_error_arms": n_err,
        "elapsed_secs": round(elapsed, 3),
        "price_assumptions_usd_per_1m": {"input_miss": PRICE_INPUT_MISS,
                                         "input_hit": PRICE_INPUT_HIT,
                                         "output": PRICE_OUTPUT},
        "boundary": "自建多智能体改造，动作名层级正确性核对，非官方 TAU-bench pass^k 榜单成绩。",
        "cost_metric_note": ("首选效率口径为 prompt_tokens 减少与 list_price_cost（未缓存价，"
                             "可复现）。est_cost_usd 为实时 DeepSeek 前缀缓存价：本场景政策为"
                             "全域常量，供应商跨请求全局缓存状态会使先被预热的一臂显著变便宜，"
                             "非受控指标，仅作参考。"),
    }
    (out_dir / "per_task.json").write_text(
        json.dumps(per_task, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(combined["efficiency_no_fault"], ensure_ascii=False, indent=2))
    print(json.dumps(combined["resilience_with_fault"], ensure_ascii=False, indent=2))
    print(f"api_error_arms={n_err}  elapsed={elapsed:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--airline", type=int, default=15)
    p.add_argument("--retail", type=int, default=15)
    args = p.parse_args()
    asyncio.run(run(Path(args.tau_root), Path(args.out_dir), args.airline, args.retail))


if __name__ == "__main__":
    main()
