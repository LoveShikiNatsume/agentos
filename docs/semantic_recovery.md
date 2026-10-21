# Validator-driven semantic recovery

Runtime cannot judge arbitrary reasoning. Applications can declare a deterministic,
side-effect-free acceptance condition and get bounded retries, feedback, private
context rewind, repetition detection and events.

```python
from agentos.adapters.workflow import PlannedTask, adapt_plan
from agentos.fault.recovery import SemanticRecoveryPolicy, ValidationResult
from agentos.runtime.engine import Runtime

async def answer(ctx):
    # First attempt is 1. Use ctx.recovery_feedback for direct backend calls;
    # ctx.llm_call / ctx.llm_batch_call append the latest feedback automatically.
    response = await ctx.llm_call([{"role": "user", "content": "What is 6 * 7?"}])
    return response.content.strip()

def check_answer(result):
    return ValidationResult(result == "42", "The answer must be the exact integer 42.")

dag, handlers = adapt_plan([
    PlannedTask("q", "multiply", "solver", output_ref="answers/q",
                metadata={"validator": "integer_answer"}),
], {"solver": answer}, framework="application")

runtime = Runtime(
    llm_backend=backend, handlers=handlers,
    validators={"integer_answer": check_answer},
    recovery_policy=SemanticRecoveryPolicy(max_attempts=2, restore_context=True),
)
results = await runtime.run(dag)
```

The validator receives only the task result. Register a per-task closure when the
acceptance condition depends on its inputs. Sync and async functions are supported;
they must return `ValidationResult(passed: bool, reason: str = "")`. Unknown validator
names fail before execution. Exceptions or invalid return types are terminal
`ValidationExecutionError` (`validation_error`), never semantic rejection. Async
validation and accepted publication share the handler's attempt timeout. As with
asyncio generally, synchronous blocking code cannot be interrupted by that timeout.

`SemanticRecoveryPolicy(max_attempts=2, restore_context=True)` counts the initial
attempt in the limit. Semantic rejection raises `SemanticValidationError` (failure
type `semantic`); exceptions and timeouts retain failure type `execution`. Both use
the same bounded retry path and receive a user feedback message describing the
failure. `restore_context=True` selects `CONTEXT_RETRY`; `False` selects `RETRY`.
This is the single experimental difference between snapshot and dirty retries.
Legacy policies remain usable. Without `metadata["validator"]`, validation and
repetition checks are disabled by default and legacy retry policies retain behavior.

Feedback lives outside the private snapshot, is delivered after restore, appears
in `ctx.recovery_feedback`, and is also recorded as a `recovery_feedback` bus message.
The latest message is appended to `ctx.llm_call` / `ctx.llm_batch_call` prompts.
Subprocess retries receive copied messages in `WorkerSpec.payload["recovery_feedback"]`;
the original `worker_payload` metadata is unchanged. Workers must consume this field
when they build their own prompts.

A validated task stops on two consecutive rejected results with equal normalized
SHA-256 hashes, even when retry budget remains. JSON-like result objects use sorted
keys and NFC/whitespace-normalized string values. Raw JSON object/array strings are
parsed first; duplicate keys and nonfinite values fall back to text normalization.
Case, numbers and list order matter. An intervening exception breaks the result
chain. A passing validator decision always wins over a prior equal rejected value.
Unsupported result objects fall back to `repr`; return JSON-like results for reliable
repetition detection.

Inside an attempt, more than `metadata["repeated_tool_call_threshold"]` identical
`ctx.tool_call` calls (default 3) stop before executing the extra call. Tool argument
keys are canonicalized, but string contents remain exact. Counts reset on retry.
Repetition is a terminal `RepetitionError` regardless of the configured policy and
is latched even if the handler catches it and returns. `detect_repetition=True`
explicitly enables tool detection without a validator. Calls made outside
`ctx.tool_call`, including worker-local tools, cannot be observed by this mechanism.

Events include `fault.validation_started`, `fault.validation_completed`,
`fault.retry_scheduled`, `fault.repetition_detected`, and `fault.recovery_stopped`.
They include 1-based attempt, failure type and reason (successful validation has
`failure_type=None`). Existing recovery events also include failure classification.
The old `fault.recovery_completed.attempt` convention is retained for unvalidated
legacy tasks; validated tasks report the actual 1-based successful attempt.

`RunnerHandler` (including the static LangGraph adapter) publishes output refs and
notifications only after acceptance. Custom handlers can implement
`async accept_result(ctx, result)` to defer publication similarly. Runtime cannot
undo arbitrary side effects from `run`, tools or external services. Snapshot restore
rewinds private context only; shared state and external tool effects are not rolled
back. This is an acceptance/retry mechanism, not a general transaction protocol.
