# Context selection analysis

Backend: provider; token counts come from reported request usage.

Split: `probe`; manifest: `2b2c04222aa8875ecde007089d2311987244051c19af2969e571c58631d06d85`; plans: `c81e4f890bf23548dc71b145dc2defb78d16ce155ab80b51d09c7d13043cf849`.

Replicates are averaged within each question. ITT includes every non-truncated question and counts failures as incorrect. Valid accuracy requires every replicate to be valid. Any truncated replicate excludes its entire question from ITT, valid means, and paired inference.

| Arm | Questions / runs | ITT accuracy (n) | Valid accuracy (n) | Mean input / valid question replicate | Mean calls / valid question replicate | Execution truncation | Planner truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 20 / 20 | 65.00% (20) | 65.00% (20) | 45,414.25 | 3.50 | 0.00% (0) | 0.00% (0) |
| declared | 20 / 20 | 85.00% (20) | 85.00% (20) | 5,233.10 | 3.50 | 0.00% (0) | 0.00% (0) |
| bm25 | 20 / 20 | 65.00% (20) | 65.00% (20) | 5,323.95 | 3.50 | 0.00% (0) | 0.00% (0) |
| oracle | 20 / 20 | 80.00% (20) | 80.00% (20) | 15,965.65 | 3.50 | 0.00% (0) | 0.00% (0) |

Shared planner cost is excluded above and from paired comparisons. The table below shows recorded aggregate input, including failed and truncated runs. NA means usage was missing; it is never replaced with zero.

| Arm | Execution calls | Execution input | Known input / missing runs | Shared planner calls / input | Including planner calls / input |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 70.00 | 908,285.00 | 908,285.00 / 0 | 20.00 / 29,374.00 | 90.00 / 937,659.00 |
| declared | 70.00 | 104,662.00 | 104,662.00 / 0 | 20.00 / 29,374.00 | 90.00 / 134,036.00 |
| bm25 | 70.00 | 106,479.00 | 106,479.00 / 0 | 20.00 / 29,374.00 | 90.00 / 135,853.00 |
| oracle | 70.00 | 319,313.00 | 319,313.00 / 0 | 20.00 / 29,374.00 | 90.00 / 348,687.00 |

Paired differences are arm minus all. A negative input difference denotes savings. Confidence intervals use question-level paired percentile bootstrap.

| Arm vs all | Valid coverage | Accuracy difference [95% CI] | Input difference [95% CI] | Relative input change | Token coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| declared | 20 / 20 (100.00%) | 20.00 pp [5.00, 40.00] | -40181.15 [-43705.97, -36818.46] | -88.48% | 100.00% |
| bm25 | 20 / 20 (100.00%) | 0.00 pp [-25.00, 25.00] | -40090.30 [-43701.33, -36654.37] | -88.28% | 100.00% |
| oracle | 20 / 20 (100.00%) | 15.00 pp [0.00, 30.00] | -29448.60 [-31470.33, -27496.26] | -64.84% | 100.00% |

| Arm | Required-document selection | Questions | ITT accuracy | Valid accuracy |
| --- | --- | ---: | ---: | ---: |
| all | complete | 20 | 65.00% | 65.00% |
| all | incomplete | 0 | NA | NA |
| all | unknown | 0 | NA | NA |
| declared | complete | 20 | 85.00% | 85.00% |
| declared | incomplete | 0 | NA | NA |
| declared | unknown | 0 | NA | NA |
| bm25 | complete | 9 | 77.78% | 77.78% |
| bm25 | incomplete | 11 | 54.55% | 54.55% |
| bm25 | unknown | 0 | NA | NA |
| oracle | complete | 20 | 80.00% | 80.00% |
| oracle | incomplete | 0 | NA | NA |
| oracle | unknown | 0 | NA | NA |

Invalid, truncated, and failed runs (question IDs and full failure details):

```json
{
  "all": {
    "invalid": [],
    "truncated": [],
    "failures": [
      {
        "question_id": "q011",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "all answer values must be strings"
          }
        ]
      }
    ]
  },
  "declared": {
    "invalid": [],
    "truncated": [],
    "failures": []
  },
  "bm25": {
    "invalid": [],
    "truncated": [],
    "failures": [
      {
        "question_id": "q002",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "all answer values must be strings"
          }
        ]
      },
      {
        "question_id": "q003",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "all answer values must be strings"
          }
        ]
      }
    ]
  },
  "oracle": {
    "invalid": [],
    "truncated": [],
    "failures": []
  }
}
```
