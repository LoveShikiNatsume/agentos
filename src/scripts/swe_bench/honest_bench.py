"""Honest framework comparison on real SWE-bench tasks.

For each framework (LangGraph, AutoGen) and each real SWE-bench task we run two
arms whose ONLY difference is "does the LLM call go through our Runtime policy?":

  native  : the framework orchestrates the 6-agent DAG; every agent is handed
            the full repository code; a real crash in the patcher propagates and
            kills the downstream agents (no recovery).
  runtime : same framework, same DAG, same model — but our Runtime (a) scopes
            context so synthesis agents don't re-read the whole repo, and (b)
            catches the patcher crash and really retries it.

All token/cache numbers come from real DeepSeek `usage` (prompt_cache_hit_tokens
/ prompt_cache_miss_tokens). Nothing is hardcoded; the patcher fault is a real
raised exception; the recovery is a real re-run.

Usage:
    .venv/bin/python scripts/swe_bench/honest_bench.py --backend mock --limit 1
    DEEPSEEK_API_KEY=... .venv/bin/python scripts/swe_bench/honest_bench.py \
        --backend deepseek --frameworks langgraph,autogen --limit 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import operator
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from typing_extensions import TypedDict

from scripts.swe_bench import swe_workload as W

# DeepSeek V4 Flash list-price assumptions (USD / 1M tokens) for the cost line only.
# Updated 2026-07-10 (registry N13); previous constants were V3-era prices.
PRICE_MISS, PRICE_HIT, PRICE_OUT = 0.14, 0.0028, 0.28
MAX_TOKENS = 220


@dataclass
class Acc:
    """Per-arm accumulator (real usage)."""
    prompt: int = 0
    completion: int = 0
    hit: int = 0
    miss: int = 0
    latency: float = 0.0
    calls: int = 0
    completed: int = 0
    recovered: int = 0
    faults: int = 0
    findings: list[str] = field(default_factory=list)

    def record(self, u: dict[str, Any]) -> None:
        self.prompt += u["prompt_tokens"]
        self.completion += u["completion_tokens"]
        self.hit += u["cache_hit_tokens"]
        self.miss += u["cache_miss_tokens"]
        self.latency += u["latency_secs"]
        self.calls += 1

    def to_metrics(self) -> dict[str, Any]:
        total_cache = self.hit + self.miss
        cost = (self.miss * PRICE_MISS + self.hit * PRICE_HIT
                + self.completion * PRICE_OUT) / 1e6
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "cache_hit_tokens": self.hit,
            "cache_miss_tokens": self.miss,
            "cache_hit_ratio": round(self.hit / total_cache, 4) if total_cache else 0.0,
            "est_cost_usd": round(cost, 6),
            "avg_latency_secs": round(self.latency / self.calls, 3) if self.calls else 0.0,
            "tasks_completed": self.completed,
            "tasks_total": len(W.ROLE_ORDER),
            "recovered_faults": self.recovered,
            "faults_injected": self.faults,
        }


# --------------------------------------------------------------------------- #
# Model call layer — real DeepSeek (real cache tokens) or a mock for dry-runs.
# --------------------------------------------------------------------------- #
def make_backend(kind: str, base_url: str, model: str):
    if kind == "deepseek":
        from agentos.llm.deepseek_backend import DeepSeekBackend
        return DeepSeekBackend(api_key="local-bench", base_url=base_url, model=model)
    return None  # mock


async def call_model(backend, messages: list[dict[str, str]]) -> dict[str, Any]:
    if backend is None:  # mock: tokens proportional to prompt length, no cache
        pt = sum(len(m["content"]) for m in messages) // 4
        return {"prompt_tokens": pt, "completion_tokens": 18,
                "cache_hit_tokens": 0, "cache_miss_tokens": pt,
                "latency_secs": 0.0, "content": "[mock output]"}
    from agentos.llm.base import LLMRequest
    req = LLMRequest(agent_id="swe_bench", messages=messages,
                     max_tokens=MAX_TOKENS, temperature=0.3, backend="deepseek")
    resp = await backend.generate(req)
    m = resp.metrics
    return {
        "prompt_tokens": m.prompt_tokens if m else 0,
        "completion_tokens": m.completion_tokens if m else 0,
        "cache_hit_tokens": m.cache_hit_tokens if m else 0,
        "cache_miss_tokens": m.cache_miss_tokens if m else 0,
        "latency_secs": resp.latency_secs,
        "content": resp.content,
    }


async def run_agent(backend, acc: Acc, role: str, mode: str, task: W.SweTask,
                    fault: bool) -> None:
    """Run one agent; inject a REAL crash on the patcher and (runtime) retry it."""
    findings = "\n".join(acc.findings)
    msgs = W.messages_for(role, mode, task, findings)
    attempts = 2 if mode == "runtime" else 1
    for attempt in range(attempts):
        try:
            if fault and role == W.FAULT_ROLE and attempt == 0:
                acc.faults += 1
                raise RuntimeError(f"injected fault: {role} crashed")
            u = await call_model(backend, msgs)
            acc.record(u)
            acc.completed += 1
            if role in W.CODE_ROLES:
                acc.findings.append(f"[{role}] {u['content'][:400]}")
            return
        except RuntimeError:
            if attempt + 1 < attempts:        # runtime: real recovery
                acc.recovered += 1
                continue
            raise                              # native: propagate -> kills downstream


# --------------------------------------------------------------------------- #
# Framework 1: LangGraph StateGraph (real orchestration entrypoint).
# --------------------------------------------------------------------------- #
class _LGState(TypedDict, total=False):
    done: Annotated[list[str], operator.add]


async def run_langgraph(backend, task: W.SweTask, mode: str, fault: bool) -> dict[str, Any]:
    from langgraph.graph import StateGraph

    acc = Acc()
    State = _LGState
    builder = StateGraph(State)
    for role in W.ROLE_ORDER:
        def make_node(r: str):
            async def node(state: State) -> dict:
                await run_agent(backend, acc, r, mode, task, fault)
                return {"done": [r]}
            return node
        builder.add_node(role, make_node(role))
    for role in W.ROLE_ORDER:
        if not W.DEPS[role]:
            builder.add_edge("__start__", role)
        for dep in W.DEPS[role]:
            builder.add_edge(dep, role)
    for role in W.ROLE_ORDER:
        if not any(role in W.DEPS[o] for o in W.ROLE_ORDER):
            builder.add_edge(role, "__end__")
    graph = builder.compile()

    try:
        await graph.ainvoke({"done": []})
    except RuntimeError:
        pass  # native patcher crash kills the graph; acc reflects what finished
    return acc.to_metrics()


# --------------------------------------------------------------------------- #
# Framework 2: AutoGen AssistantAgent (real orchestration entrypoint).
# --------------------------------------------------------------------------- #
def _make_autogen_client(backend, acc: Acc):
    from autogen_core.models import RequestUsage, CreateResult, ModelInfo

    class _Client:
        @property
        def model_info(self):
            return ModelInfo(vision=False, function_calling=False, json_output=False,
                             family="unknown", structured_output=False)

        @property
        def capabilities(self):
            return {"vision": False, "function_calling": False, "json_output": False}

        async def create(self, messages, *, tools=None, tool_choice="auto",
                          json_output=None, extra_create_args=None, cancellation_token=None):
            openai = []
            for msg in messages:
                content = msg.content if hasattr(msg, "content") else str(msg)
                role = "user"
                if hasattr(msg, "source") and getattr(msg, "source") not in ("user", None):
                    role = "system" if "system" in str(type(msg)).lower() else "user"
                openai.append({"role": role, "content": content if isinstance(content, str) else str(content)})
            u = await call_model(backend, openai)
            acc.record(u)
            return CreateResult(finish_reason="stop", content=u["content"],
                                usage=RequestUsage(prompt_tokens=u["prompt_tokens"],
                                                   completion_tokens=u["completion_tokens"]),
                                cached=False)

        async def create_stream(self, *a, **k):
            r = await self.create(*a, **k)
            yield r

        def actual_usage(self):
            from autogen_core.models import RequestUsage
            return RequestUsage(prompt_tokens=acc.prompt, completion_tokens=acc.completion)

        def total_usage(self):
            return self.actual_usage()

        def count_tokens(self, messages, *, tools=None):
            return sum(len(str(getattr(m, "content", m))) for m in messages) // 4

        def remaining_tokens(self, messages, *, tools=None):
            return 128000 - self.count_tokens(messages)

        async def close(self):
            pass

        def dump_component(self):
            from autogen_core import ComponentModel
            return ComponentModel(type="SweBackendClient", config={})

    return _Client()


async def run_autogen(backend, task: W.SweTask, mode: str, fault: bool) -> dict[str, Any]:
    from autogen_agentchat.agents import AssistantAgent

    acc = Acc()
    client = _make_autogen_client(backend, acc)

    # AutoGen agents orchestrated over the same DAG order (topological).
    for role in W.ROLE_ORDER:
        attempts = 2 if mode == "runtime" else 1
        findings = "\n".join(acc.findings)
        user_prompt = W.build_user_prompt(role, mode, task, findings)
        agent = AssistantAgent(name=role, model_client=client, system_message=W.SYSTEM)
        crashed = False
        for attempt in range(attempts):
            try:
                if fault and role == W.FAULT_ROLE and attempt == 0:
                    acc.faults += 1
                    raise RuntimeError("injected fault: patcher crashed")
                calls_before = acc.calls
                await agent.run(task=user_prompt)
                if acc.calls > calls_before:
                    acc.completed += 1
                    if role in W.CODE_ROLES:
                        acc.findings.append(f"[{role}] (autogen finding)")
                break
            except RuntimeError:
                if attempt + 1 < attempts:
                    acc.recovered += 1
                    continue
                crashed = True
        if crashed:
            break  # native: patcher crash stops the pipeline
    return acc.to_metrics()


# --------------------------------------------------------------------------- #
# Framework 3: OpenAI Agents SDK (real Agent + Runner orchestration).
# --------------------------------------------------------------------------- #
async def run_openai_agents(backend, task: W.SweTask, mode: str, fault: bool) -> dict[str, Any]:
    from agents import Agent, Runner, set_tracing_disabled, ModelResponse, Usage
    from agents.models.interface import Model

    set_tracing_disabled(True)
    acc = Acc()

    class _BenchModel(Model):
        """Model shim routing Runner calls through our call_model() layer."""

        async def get_response(self, system_instructions, input, model_settings,
                               tools, output_schema, handoffs, tracing,
                               previous_response_id=None, conversation_id=None, prompt=None):
            from openai.types.responses import ResponseOutputMessage, ResponseOutputText
            msgs: list[dict[str, str]] = []
            if system_instructions:
                msgs.append({"role": "system", "content": system_instructions})
            if isinstance(input, str):
                msgs.append({"role": "user", "content": input})
            elif isinstance(input, list):
                for item in input:
                    content = item.content if hasattr(item, "content") else str(item)
                    msgs.append({"role": "user", "content": str(content)})
            u = await call_model(backend, msgs)
            acc.record(u)
            output_msg = ResponseOutputMessage(
                id="msg", role="assistant", status="completed", type="message",
                content=[ResponseOutputText(annotations=[], text=u["content"], type="output_text")],
            )
            return ModelResponse(
                output=[output_msg],
                usage=Usage(input_tokens=u["prompt_tokens"], output_tokens=u["completion_tokens"]),
                response_id=None,
            )

        async def stream_response(self, *args, **kwargs):
            result = await self.get_response(*args, **kwargs)
            yield result

    model = _BenchModel()

    for role in W.ROLE_ORDER:
        attempts = 2 if mode == "runtime" else 1
        findings = "\n".join(acc.findings)
        user_prompt = W.build_user_prompt(role, mode, task, findings)
        crashed = False
        for attempt in range(attempts):
            try:
                if fault and role == W.FAULT_ROLE and attempt == 0:
                    acc.faults += 1
                    raise RuntimeError("injected fault: patcher crashed")
                agent = Agent(name=role, instructions=W.SYSTEM, model=model)
                calls_before = acc.calls
                await Runner.run(agent, user_prompt)
                if acc.calls > calls_before:
                    acc.completed += 1
                    if role in W.CODE_ROLES:
                        acc.findings.append(f"[{role}] (openai_agents finding)")
                break
            except RuntimeError:
                if attempt + 1 < attempts:
                    acc.recovered += 1
                    continue
                crashed = True
        if crashed:
            break
    return acc.to_metrics()


# --------------------------------------------------------------------------- #
# Framework 4: CrewAI (real Crew + Agent + Task orchestration).
# --------------------------------------------------------------------------- #
async def run_crewai(backend, task: W.SweTask, mode: str, fault: bool) -> dict[str, Any]:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew, Process, LLM

    acc = Acc()

    class _BenchLLM(LLM):
        """CrewAI LLM subclass routing calls through our call_model() layer."""

        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            import asyncio
            msgs: list[dict[str, str]] = []
            if isinstance(messages, str):
                msgs = [{"role": "user", "content": messages}]
            else:
                for m in messages:
                    if isinstance(m, dict):
                        msgs.append(m)
                    elif hasattr(m, "content"):
                        msgs.append({"role": getattr(m, "role", "user"), "content": str(m.content)})
                    else:
                        msgs.append({"role": "user", "content": str(m)})

            async def _do():
                u = await call_model(backend, msgs)
                acc.record(u)
                return u["content"]

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _do()).result()
            return asyncio.run(_do())

        def supports_function_calling(self):
            return False

        def supports_stop_words(self):
            return True

    llm_instance = _BenchLLM(
        model=getattr(backend, "_model", "openai/deepseek-v4-flash"),
        temperature=0.3,
        max_tokens=MAX_TOKENS,
        api_key="local-bench",
        base_url=getattr(backend, "_base_url", "https://api.deepseek.com"),
    )

    for role in W.ROLE_ORDER:
        attempts = 2 if mode == "runtime" else 1
        findings = "\n".join(acc.findings)
        user_prompt = W.build_user_prompt(role, mode, task, findings)
        crashed = False
        for attempt in range(attempts):
            try:
                if fault and role == W.FAULT_ROLE and attempt == 0:
                    acc.faults += 1
                    raise RuntimeError("injected fault: patcher crashed")
                calls_before = acc.calls
                agent = CrewAgent(
                    role=role, goal=W.ROLE_INSTRUCTION[role],
                    backstory=W.SYSTEM, llm=llm_instance,
                    verbose=False, allow_delegation=False,
                )
                crew_task = CrewTask(
                    description=user_prompt,
                    expected_output="A concise technical response.",
                    agent=agent,
                )
                crew = Crew(
                    agents=[agent], tasks=[crew_task],
                    process=Process.sequential, verbose=False,
                )
                await crew.kickoff_async()
                if acc.calls > calls_before:
                    acc.completed += 1
                    if role in W.CODE_ROLES:
                        acc.findings.append(f"[{role}] (crewai finding)")
                break
            except RuntimeError:
                if attempt + 1 < attempts:
                    acc.recovered += 1
                    continue
                crashed = True
        if crashed:
            break
    return acc.to_metrics()


FRAMEWORKS = {"langgraph": run_langgraph, "autogen": run_autogen,
              "openai_agents": run_openai_agents, "crewai": run_crewai}


# --------------------------------------------------------------------------- #
async def run_all(backend_kind: str, frameworks: list[str], tasks_path: Path,
                  out_dir: Path, limit: int | None, base_url: str,
                  model: str) -> None:
    backend = make_backend(backend_kind, base_url, model)
    data = json.loads(tasks_path.read_text())
    tasks = [W.parse_task(t) for t in data["tasks"][:limit]] if limit else \
            [W.parse_task(t) for t in data["tasks"]]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task: list[dict[str, Any]] = []
    for i, task in enumerate(tasks):
        print(f"[{i + 1}/{len(tasks)}] {task.instance_id} ({task.repo}, ~{task.est_context_tokens} tok)")
        rec: dict[str, Any] = {"instance_id": task.instance_id, "repo": task.repo,
                               "est_context_tokens": task.est_context_tokens}
        for fw in frameworks:
            runner = FRAMEWORKS[fw]
            rec[fw] = {}
            for arm, mode, fault in [("native", "native", False), ("runtime", "runtime", False),
                                     ("native_fault", "native", True), ("runtime_fault", "runtime", True)]:
                m = await runner(backend, task, mode, fault)
                rec[fw][arm] = m
                print(f"    {fw:10s} {arm:14s} calls={m['llm_calls']} "
                      f"ptok={m['prompt_tokens']:>7} miss={m['cache_miss_tokens']:>7} "
                      f"hit={m['cache_hit_ratio']:.0%} cost=${m['est_cost_usd']:.4f} "
                      f"done={m['tasks_completed']}/{m['tasks_total']} recov={m['recovered_faults']}")
        per_task.append(rec)

    summary = aggregate(per_task, frameworks)
    (out_dir / "per_task.json").write_text(json.dumps(per_task, indent=2, ensure_ascii=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out_dir}/summary.json")
    print_summary(summary, frameworks)


def aggregate(per_task: list[dict], frameworks: list[str]) -> dict[str, Any]:
    def pct(b: float, r: float) -> float:
        return round((b - r) / b * 100, 1) if b else 0.0

    out: dict[str, Any] = {"n_tasks": len(per_task), "frameworks": {}}
    for fw in frameworks:
        def col(arm: str, field: str) -> int:
            return sum(t[fw][arm][field] for t in per_task)
        b_ptok, r_ptok = col("native", "prompt_tokens"), col("runtime", "prompt_tokens")
        b_miss, r_miss = col("native", "cache_miss_tokens"), col("runtime", "cache_miss_tokens")
        b_cost = round(sum(t[fw]["native"]["est_cost_usd"] for t in per_task), 4)
        r_cost = round(sum(t[fw]["runtime"]["est_cost_usd"] for t in per_task), 4)
        out["frameworks"][fw] = {
            "efficiency_no_fault": {
                "prompt_tokens": {"native": b_ptok, "runtime": r_ptok, "reduction_pct": pct(b_ptok, r_ptok)},
                "cache_miss_tokens": {"native": b_miss, "runtime": r_miss, "reduction_pct": pct(b_miss, r_miss)},
                "est_cost_usd": {"native": b_cost, "runtime": r_cost, "reduction_pct": pct(b_cost, r_cost)},
            },
            "resilience_with_fault": {
                "tasks_completed": {"native": col("native_fault", "tasks_completed"),
                                    "runtime": col("runtime_fault", "tasks_completed"),
                                    "total": col("native_fault", "tasks_total")},
                "recovered_faults": {"native": col("native_fault", "recovered_faults"),
                                     "runtime": col("runtime_fault", "recovered_faults"),
                                     "injected": col("runtime_fault", "faults_injected")},
            },
        }
    out["price_assumptions_usd_per_1m"] = {"input_miss": PRICE_MISS, "input_hit": PRICE_HIT, "output": PRICE_OUT}
    return out


def print_summary(s: dict, frameworks: list[str]) -> None:
    print("=" * 70)
    print(f"  HONEST framework comparison on {s['n_tasks']} real SWE-bench tasks")
    print("=" * 70)
    for fw in frameworks:
        f = s["frameworks"][fw]
        e, r = f["efficiency_no_fault"], f["resilience_with_fault"]
        print(f"  [{fw}]  native -> native+Runtime")
        print(f"    prompt tokens (no fault)  : {e['prompt_tokens']['native']:>8} -> "
              f"{e['prompt_tokens']['runtime']:>8}   (-{e['prompt_tokens']['reduction_pct']}%)")
        print(f"    full-price miss tokens    : {e['cache_miss_tokens']['native']:>8} -> "
              f"{e['cache_miss_tokens']['runtime']:>8}   (-{e['cache_miss_tokens']['reduction_pct']}%)")
        print(f"    est cost USD              : {e['est_cost_usd']['native']:>8} -> "
              f"{e['est_cost_usd']['runtime']:>8}   (-{e['est_cost_usd']['reduction_pct']}%)")
        print(f"    tasks completed (fault)   : native {r['tasks_completed']['native']}/{r['tasks_completed']['total']}"
              f"  vs  runtime {r['tasks_completed']['runtime']}/{r['tasks_completed']['total']}")
        print(f"    faults recovered          : native {r['recovered_faults']['native']}"
              f"  vs  runtime {r['recovered_faults']['runtime']}  (injected {r['recovered_faults']['injected']})")


def main() -> None:
    ap = argparse.ArgumentParser(prog="honest-swe-bench")
    ap.add_argument("--backend", default="mock", choices=["mock", "deepseek"])
    ap.add_argument("--frameworks", default="langgraph,autogen")
    ap.add_argument("--tasks", default="runs/swe_bench/tasks.json")
    ap.add_argument("--out-dir", default="runs/swe_bench/framework_result")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--model", default="openai/deepseek-v4-flash")
    args = ap.parse_args()
    fws = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    asyncio.run(run_all(args.backend, fws, Path(args.tasks), Path(args.out_dir),
                        args.limit, args.base_url, args.model))


if __name__ == "__main__":
    main()
