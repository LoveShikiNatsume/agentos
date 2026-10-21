# AgentOS-Runtime Benchmark Comparison

## Scenario
- **DAG**: plan → code → test → fix → deploy + independent review
- **Fault**: kill injected on `code` task
- **Backend**: cache-simulating mock (prefix overlap → cache hits)

## Results

| Metric | Baseline | Runtime | Delta |
|--------|----------|---------|-------|
| Latency | 0.101s | 0.139s | +37.7% |
| LLM Calls | 2 | 6 | +4 |
| Prompt Tokens | 490 | 1606 | +1116 |
| Cache Hit Tokens | 0 | 1250 | +1250 |
| Cache Hit Ratio | 0.0% | 77.8% | +77.8% |
| Tasks Completed | 2/6 | 6/6 | +4 |
| Fault Recovery | CRASHED | RECOVERED | ✓ |
| Context Refs Restored | - | 4 | - |
| Tokens Saved (Recovery) | - | 247 | - |
| Events (Observability) | 49 | 109 | +60 |

## Key Findings

1. **Cache Hit Improvement**: Runtime achieves 77.8% cache hit ratio vs 0.0% baseline through stable ContextRef prefix.
2. **Fault Tolerance**: Baseline crashes on fault; Runtime recovers via ContextRetry preserving 4 context refs.
3. **Full Completion**: Runtime completes all 6 tasks; Baseline only completes 2.
4. **Observability**: Runtime emits 109 structured events for full-trace debugging.
