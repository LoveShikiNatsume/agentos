"""Network-free AST fixture backend; real transport is shared with experiment A."""
from __future__ import annotations

import ast
import json
import math

from agentos.execution.backend import _usage_metrics
from agentos.llm.base import LLMResponse
from experiments.context_selection.backends import LimitedBackend, SeededOpenAIBackend
from experiments.context_selection.generate import _facts


class StubBackend(SeededOpenAIBackend):
    """Reads request evidence only; never receives gold answers or the manifest.

    This intentionally simplistic fake answerer keeps the earliest fact per
    symbol, making stale scratch observable. Its behavior is not model evidence.
    """
    async def generate(self, request):
        payload = next(json.loads(m["content"]) for m in reversed(request.messages)
                       if m["role"] == "user" and m.get("content", "").startswith('{"stage":'))
        if payload["stage"] == "extract":
            document = payload["document"]
            scope = document.get("definition_scope", "")
            tree = ast.parse(("class _Scope:\n" if scope else "") + document["text"])
            body = tree.body[0].body if scope else tree.body
            facts = []
            for node in body:
                for fact in _facts(node, document["module"], document.get("definition_scope", "")):
                    if fact["symbol"] in payload["symbols"]:
                        default = fact["first_default"]
                        facts.append({"symbol": fact["symbol"], "parameter_count": fact["parameter_count"],
                            "first_default_parameter": default[0] if default else None,
                            "first_default": default[1] if default else None})
            answer = {"facts": facts}
        elif payload["stage"] == "answer":
            earliest = {}
            for message in request.messages:
                if message["role"] != "system":
                    continue
                for line in message["content"].splitlines():
                    if line.startswith("PRIVATE_FACTS="):
                        for fact in json.loads(line.removeprefix("PRIVATE_FACTS=")).get("facts", []):
                            earliest.setdefault(fact["symbol"], fact)
            facts = [f for f in earliest.values() if f["symbol"] in payload["symbols"]]
            winner = max(facts, key=lambda f: f["parameter_count"]) if facts else None
            answer = ({"winner": winner["symbol"], "parameter": winner["first_default_parameter"],
                       "default": winner["first_default"]} if winner else
                      {"winner": "stub.missing", "parameter": "missing", "default": "None"})
        else:
            raise ValueError("unknown stub stage")
        content = json.dumps(answer, ensure_ascii=False)
        prompt = math.ceil(sum(len(m.get("content") or "") for m in request.messages) / 4)
        completion = math.ceil(len(content) / 4)
        usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                 "prompt_tokens_details": {"cached_tokens": 0}, "measurement": "stub_chars_div_4_not_model_tokens"}
        metrics, total = _usage_metrics(usage, model=self.config["model"], elapsed=0, finish_reason="stop")
        return LLMResponse(content, total, self.config["model"], 0, backend="stub", metrics=metrics,
                           metadata={"usage": usage, "finish_reason": "stop",
                                     "assistant_message": {"role": "assistant", "content": content}})


def make_backend(config):
    return StubBackend(config) if config["backend"] == "stub" else SeededOpenAIBackend(config)
