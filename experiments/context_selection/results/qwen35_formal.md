# Context selection analysis

Backend: provider; token counts come from reported request usage.

Split: `formal`; manifest: `2b2c04222aa8875ecde007089d2311987244051c19af2969e571c58631d06d85`; plans: `ea027c70cd5a7a1c06bbb3700a35f9408c6f1b5ce97a7fea80d39a3f627892c6`.

Replicates are averaged within each question. ITT includes every non-truncated question and counts failures as incorrect. Valid accuracy requires every replicate to be valid. Any truncated replicate excludes its entire question from ITT, valid means, and paired inference.

| Arm | Questions / runs | ITT accuracy (n) | Valid accuracy (n) | Mean input / valid question replicate | Mean calls / valid question replicate | Execution truncation | Planner truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 100 / 100 | 66.00% (100) | 66.67% (99) | 46,210.66 | 3.51 | 0.00% (0) | 0.00% (0) |
| declared | 100 / 100 | 72.00% (100) | 72.73% (99) | 5,518.08 | 3.51 | 0.00% (0) | 0.00% (0) |
| bm25 | 100 / 100 | 48.00% (100) | 48.48% (99) | 5,532.71 | 3.51 | 0.00% (0) | 0.00% (0) |
| oracle | 100 / 100 | 65.00% (100) | 65.66% (99) | 16,808.10 | 3.51 | 0.00% (0) | 0.00% (0) |

Shared planner cost is excluded above and from paired comparisons. The table below shows recorded aggregate input, including failed and truncated runs. NA means usage was missing; it is never replaced with zero.

| Arm | Execution calls | Execution input | Known input / missing runs | Shared planner calls / input | Including planner calls / input |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 347.00 | 4,574,855.00 | 4,574,855.00 / 0 | 101.00 / 157,838.00 | 448.00 / 4,732,693.00 |
| declared | 347.00 | 546,290.00 | 546,290.00 / 0 | 101.00 / 157,838.00 | 448.00 / 704,128.00 |
| bm25 | 347.00 | 547,738.00 | 547,738.00 / 0 | 101.00 / 157,838.00 | 448.00 / 705,576.00 |
| oracle | 347.00 | 1,664,002.00 | 1,664,002.00 / 0 | 101.00 / 157,838.00 | 448.00 / 1,821,840.00 |

Paired differences are arm minus all. A negative input difference denotes savings. Confidence intervals use question-level paired percentile bootstrap.

| Arm vs all | Valid coverage | Accuracy difference [95% CI] | Input difference [95% CI] | Relative input change | Token coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| declared | 99 / 100 (99.00%) | 6.06 pp [-3.03, 15.15] | -40692.58 [-42236.07, -39131.99] | -88.06% | 99.00% |
| bm25 | 99 / 100 (99.00%) | -18.18 pp [-30.30, -5.05] | -40677.95 [-42239.97, -39105.34] | -88.03% | 99.00% |
| oracle | 99 / 100 (99.00%) | -1.01 pp [-9.09, 7.07] | -29402.56 [-30403.62, -28413.08] | -63.63% | 99.00% |

| Arm | Required-document selection | Questions | ITT accuracy | Valid accuracy |
| --- | --- | ---: | ---: | ---: |
| all | complete | 99 | 66.67% | 66.67% |
| all | incomplete | 0 | NA | NA |
| all | unknown | 1 | 0.00% | NA |
| declared | complete | 99 | 72.73% | 72.73% |
| declared | incomplete | 0 | NA | NA |
| declared | unknown | 1 | 0.00% | NA |
| bm25 | complete | 28 | 75.00% | 75.00% |
| bm25 | incomplete | 71 | 38.03% | 38.03% |
| bm25 | unknown | 1 | 0.00% | NA |
| oracle | complete | 99 | 65.66% | 65.66% |
| oracle | incomplete | 0 | NA | NA |
| oracle | unknown | 1 | 0.00% | NA |

Invalid, truncated, and failed runs (question IDs and full failure details):

```json
{
  "all": {
    "invalid": [
      {
        "question_id": "q079",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      }
    ],
    "truncated": [],
    "failures": [
      {
        "question_id": "q049",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "answer must be one JSON object"
          }
        ]
      },
      {
        "question_id": "q052",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q057",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q064",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "parameter must be a Python identifier"
          }
        ]
      },
      {
        "question_id": "q072",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q079",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      },
      {
        "question_id": "q106",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q116",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      }
    ]
  },
  "declared": {
    "invalid": [
      {
        "question_id": "q079",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      }
    ],
    "truncated": [],
    "failures": [
      {
        "question_id": "q049",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "answer must be one JSON object"
          }
        ]
      },
      {
        "question_id": "q052",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q064",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q079",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      },
      {
        "question_id": "q115",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "answer must be one JSON object"
          }
        ]
      }
    ]
  },
  "bm25": {
    "invalid": [
      {
        "question_id": "q079",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      }
    ],
    "truncated": [],
    "failures": [
      {
        "question_id": "q027",
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
        "question_id": "q031",
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
        "question_id": "q035",
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
        "question_id": "q044",
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
        "question_id": "q049",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "answer must be one JSON object"
          }
        ]
      },
      {
        "question_id": "q052",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q070",
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
        "question_id": "q074",
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
        "question_id": "q079",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      },
      {
        "question_id": "q081",
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
        "question_id": "q091",
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
        "question_id": "q093",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "parameter must be a Python identifier"
          }
        ]
      },
      {
        "question_id": "q110",
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
        "question_id": "q115",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "answer must be one JSON object"
          }
        ]
      }
    ]
  },
  "oracle": {
    "invalid": [
      {
        "question_id": "q079",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      }
    ],
    "truncated": [],
    "failures": [
      {
        "question_id": "q034",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q036",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q049",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "answer must be one JSON object"
          }
        ]
      },
      {
        "question_id": "q052",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q060",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q064",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "parameter must be a Python identifier"
          }
        ]
      },
      {
        "question_id": "q079",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      },
      {
        "question_id": "q106",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "default must be a Python expression"
          }
        ]
      },
      {
        "question_id": "q115",
        "replicate": 0,
        "valid": true,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          {
            "kind": "answer_format",
            "reason": "answer must be one JSON object"
          }
        ]
      }
    ]
  }
}
```
