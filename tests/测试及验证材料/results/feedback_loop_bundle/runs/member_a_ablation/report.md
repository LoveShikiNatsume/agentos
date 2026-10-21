# Member A Open/Closed Loop Ablation

- Iterations: 1

## Modes

| Mode | Usage Samples | Feedback Applied | Affected Decisions | Kernel Publishes | Elapsed Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dag_only` | 0.00 | 0.00 | 0.00 | 2.00 | 0.0014s |
| `observation_only` | 8.00 | 0.00 | 0.00 | 1.00 | 0.0746s |
| `feedback_control` | 17.00 | 1.00 | 1.00 | 2.00 | 0.1471s |
| `feedback_bpf_publish` | 17.00 | 1.00 | 1.00 | 2.00 | 0.1440s |

## Comparisons

- `observation_adds_usage_samples`: True
- `feedback_adds_cgroup_decisions`: True
- `bpf_publish_records_kernel_events`: True
