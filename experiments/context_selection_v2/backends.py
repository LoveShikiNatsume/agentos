"""Real payload wrapper and non-factual, network-free smoke backend."""
from __future__ import annotations
import json
import math
from agentos.execution.backend import OpenAIBackend, _usage_metrics
from agentos.llm.base import LLMResponse
from .common import RESPONSE_FORMAT

class SeededOpenAIBackend(OpenAIBackend):
    def request_payload(self, request):
        payload = {**super().request_payload(request), "seed": self.config["seed"]}
        if self.config.get("enable_thinking") is not None:
            # Qwen3 系列的聊天模板开关；思考内容会占用输出上限，需在协议中写明取值。
            payload["chat_template_kwargs"] = {"enable_thinking": self.config["enable_thinking"]}
        payload["response_format"] = dict(RESPONSE_FORMAT)
        return payload

class StubBackend(SeededOpenAIBackend):
    """Only inspects submitted messages; has no manifest or gold access."""
    async def generate(self, request):
        payload = json.loads(next(m["content"] for m in request.messages if m["role"] == "user"))
        stage = payload["stage"]
        if stage == "planner":
            answer = {"readers": [{"id": f"reader{i+1}",
                       "goal": "Extract signature facts for " + row["name"],
                       "context_refs": [row["id"]]}
                       for i, row in enumerate(payload["catalog"][:2])]}
        elif stage == "reader":
            responses = [m for m in request.messages if m["role"] == "assistant"]
            feedback = [m for m in request.messages if m["role"] == "user"][1:]
            denied = any(json.loads(m["content"]).get("result", {}).get("denied") for m in feedback)
            missing = [r["id"] for r in payload["catalog"] if r["id"] not in request.context_refs]
            if missing and not denied and len(responses) < 2:
                answer = {"need": missing[:2 if not responses else 1], "reason": "Stub exercises bounded missing-source requests."}
            else:
                answer = {"facts": []}
        elif stage == "answerer":
            answer = ({"callee": "stub.invalid", "parameter_count": 0, "parameter": None, "default": None}
                      if payload["family"] == "hop" else
                      {"winner": "stub.invalid", "parameter": "stub", "default": "None"})
        else:
            raise ValueError("unknown stub stage")
        content = "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"
        prompt = math.ceil(sum(len(m.get("content") or "") for m in request.messages) / 4)
        completion = math.ceil(len(content) / 4)
        usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                 "total_tokens": prompt + completion, "prompt_tokens_details": {"cached_tokens": 0},
                 "measurement": "stub_chars_div_4_not_model_tokens"}
        metrics, total = _usage_metrics(usage, model=self.config["model"], elapsed=0,
                                       finish_reason="stop")
        return LLMResponse(content, total, self.config["model"], 0, backend="stub",
                           metrics=metrics, metadata={"usage": usage, "finish_reason": "stop",
                           "assistant_message": {"role": "assistant", "content": content}})

def make_backend(config):
    return StubBackend(config) if config["backend"] == "stub" else SeededOpenAIBackend(config)


class LimitedBackend:
    def __init__(self, backend, semaphore):
        self.backend, self.semaphore = backend, semaphore

    def request_payload(self, request):
        return self.backend.request_payload(request)

    async def generate(self, request):
        async with self.semaphore:
            return await self.backend.generate(request)
