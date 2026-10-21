# Context selection analysis

Backend: provider; token counts come from reported request usage.

Split: `formal`; manifest: `2b2c04222aa8875ecde007089d2311987244051c19af2969e571c58631d06d85`; plans: `288e43333ca3b39fda029a0af85768b85ed839561b4b77e02b063bc0a9500715`.

Replicates are averaged within each question. ITT includes every non-truncated question and counts failures as incorrect. Valid accuracy requires every replicate to be valid. Any truncated replicate excludes its entire question from ITT, valid means, and paired inference.

| Arm | Questions / runs | ITT accuracy (n) | Valid accuracy (n) | Mean input / valid question replicate | Mean calls / valid question replicate | Execution truncation | Planner truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 100 / 100 | 85.00% (100) | 98.84% (86) | 45,220.33 | 3.43 | 0.00% (0) | 0.00% (0) |
| declared | 100 / 100 | 83.00% (100) | 96.51% (86) | 5,354.59 | 3.43 | 0.00% (0) | 0.00% (0) |
| bm25 | 100 / 100 | 51.00% (100) | 59.30% (86) | 5,333.38 | 3.43 | 0.00% (0) | 0.00% (0) |
| oracle | 100 / 100 | 84.00% (100) | 97.67% (86) | 16,029.97 | 3.43 | 0.00% (0) | 0.00% (0) |

Shared planner cost is excluded above and from paired comparisons. The table below shows recorded aggregate input, including failed and truncated runs. NA means usage was missing; it is never replaced with zero.

| Arm | Execution calls | Execution input | Known input / missing runs | Shared planner calls / input | Including planner calls / input |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 295.00 | 3,888,948.00 | 3,888,948.00 / 0 | 200.00 / 337,854.00 | 495.00 / 4,226,802.00 |
| declared | 295.00 | 460,495.00 | 460,495.00 / 0 | 200.00 / 337,854.00 | 495.00 / 798,349.00 |
| bm25 | 295.00 | 458,671.00 | 458,671.00 / 0 | 200.00 / 337,854.00 | 495.00 / 796,525.00 |
| oracle | 295.00 | 1,378,577.00 | 1,378,577.00 / 0 | 200.00 / 337,854.00 | 495.00 / 1,716,431.00 |

Paired differences are arm minus all. A negative input difference denotes savings. Confidence intervals use question-level paired percentile bootstrap.

| Arm vs all | Valid coverage | Accuracy difference [95% CI] | Input difference [95% CI] | Relative input change | Token coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| declared | 86 / 100 (86.00%) | -2.33 pp [-6.98, 2.33] | -39865.73 [-41558.16, -38145.43] | -88.16% | 86.00% |
| bm25 | 86 / 100 (86.00%) | -39.53 pp [-50.00, -29.07] | -39886.94 [-41566.25, -38177.03] | -88.21% | 86.00% |
| oracle | 86 / 100 (86.00%) | -1.16 pp [-4.65, 2.33] | -29190.36 [-30297.34, -28075.81] | -64.55% | 86.00% |

| Arm | Required-document selection | Questions | ITT accuracy | Valid accuracy |
| --- | --- | ---: | ---: | ---: |
| all | complete | 86 | 98.84% | 98.84% |
| all | incomplete | 0 | NA | NA |
| all | unknown | 14 | 0.00% | NA |
| declared | complete | 86 | 96.51% | 96.51% |
| declared | incomplete | 0 | NA | NA |
| declared | unknown | 14 | 0.00% | NA |
| bm25 | complete | 22 | 100.00% | 100.00% |
| bm25 | incomplete | 64 | 45.31% | 45.31% |
| bm25 | unknown | 14 | 0.00% | NA |
| oracle | complete | 86 | 97.67% | 97.67% |
| oracle | incomplete | 0 | NA | NA |
| oracle | unknown | 14 | 0.00% | NA |

Invalid, truncated, and failed runs (question IDs and full failure details):

```json
{
  "all": {
    "invalid": [
      {
        "question_id": "q024",
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
        "question_id": "q028",
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
        "question_id": "q030",
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
        "question_id": "q036",
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
        "question_id": "q038",
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
        "question_id": "q067",
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
        "question_id": "q070",
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
        "question_id": "q072",
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
        "question_id": "q076",
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
        "question_id": "q080",
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
        "question_id": "q096",
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
        "question_id": "q110",
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
        "question_id": "q112",
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
        "question_id": "q116",
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
        "question_id": "q024",
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
        "question_id": "q028",
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
        "question_id": "q030",
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
        "question_id": "q036",
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
        "question_id": "q038",
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
        "question_id": "q067",
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
        "question_id": "q070",
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
        "question_id": "q072",
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
        "question_id": "q076",
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
        "question_id": "q080",
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
        "question_id": "q096",
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
        "question_id": "q110",
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
        "question_id": "q112",
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
        "question_id": "q116",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      }
    ]
  },
  "declared": {
    "invalid": [
      {
        "question_id": "q024",
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
        "question_id": "q028",
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
        "question_id": "q030",
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
        "question_id": "q036",
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
        "question_id": "q038",
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
        "question_id": "q067",
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
        "question_id": "q070",
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
        "question_id": "q072",
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
        "question_id": "q076",
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
        "question_id": "q080",
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
        "question_id": "q096",
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
        "question_id": "q110",
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
        "question_id": "q112",
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
        "question_id": "q116",
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
        "question_id": "q024",
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
        "question_id": "q028",
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
        "question_id": "q030",
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
        "question_id": "q036",
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
        "question_id": "q038",
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
        "question_id": "q053",
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
        "question_id": "q067",
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
        "question_id": "q070",
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
        "question_id": "q072",
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
        "question_id": "q076",
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
        "question_id": "q080",
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
        "question_id": "q096",
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
        "question_id": "q110",
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
        "question_id": "q112",
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
        "question_id": "q116",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      }
    ]
  },
  "bm25": {
    "invalid": [
      {
        "question_id": "q024",
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
        "question_id": "q028",
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
        "question_id": "q030",
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
        "question_id": "q036",
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
        "question_id": "q038",
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
        "question_id": "q067",
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
        "question_id": "q070",
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
        "question_id": "q072",
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
        "question_id": "q076",
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
        "question_id": "q080",
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
        "question_id": "q096",
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
        "question_id": "q110",
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
        "question_id": "q112",
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
        "question_id": "q116",
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
        "question_id": "q024",
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
        "question_id": "q028",
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
        "question_id": "q029",
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
        "question_id": "q030",
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
        "question_id": "q036",
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
        "question_id": "q037",
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
        "question_id": "q038",
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
        "question_id": "q041",
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
        "question_id": "q043",
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
        "question_id": "q046",
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
        "question_id": "q056",
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
        "question_id": "q060",
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
        "question_id": "q061",
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
        "question_id": "q062",
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
        "question_id": "q067",
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
        "question_id": "q070",
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
        "question_id": "q072",
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
        "question_id": "q076",
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
        "question_id": "q077",
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
        "question_id": "q080",
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
            "reason": "all answer values must be strings"
          }
        ]
      },
      {
        "question_id": "q096",
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
        "question_id": "q100",
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
        "question_id": "q101",
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
        "question_id": "q108",
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
        "question_id": "q110",
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
        "question_id": "q112",
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
            "reason": "all answer values must be strings"
          }
        ]
      },
      {
        "question_id": "q116",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      }
    ]
  },
  "oracle": {
    "invalid": [
      {
        "question_id": "q024",
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
        "question_id": "q028",
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
        "question_id": "q030",
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
        "question_id": "q036",
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
        "question_id": "q038",
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
        "question_id": "q067",
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
        "question_id": "q070",
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
        "question_id": "q072",
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
        "question_id": "q076",
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
        "question_id": "q080",
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
        "question_id": "q096",
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
        "question_id": "q110",
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
        "question_id": "q112",
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
        "question_id": "q116",
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
        "question_id": "q024",
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
        "question_id": "q028",
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
        "question_id": "q030",
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
        "question_id": "q036",
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
        "question_id": "q038",
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
        "question_id": "q049",
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
        "question_id": "q067",
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
        "question_id": "q070",
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
        "question_id": "q072",
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
        "question_id": "q076",
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
        "question_id": "q080",
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
        "question_id": "q096",
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
        "question_id": "q110",
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
        "question_id": "q112",
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
        "question_id": "q116",
        "replicate": 0,
        "valid": false,
        "truncated_calls": 0,
        "execution_truncated_calls": 0,
        "planner_truncated_calls": 0,
        "failures": [
          "WorkflowExecutionError: workflow failed in 1 task(s): planner; 0 completed, 0 skipped"
        ]
      }
    ]
  }
}
```
