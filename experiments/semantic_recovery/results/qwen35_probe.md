# Experiment C

逐请求 Token 来自服务端 usage。

全部分配题（包括失败和截断）保留在 ITT，按最终验收状态评分；截断另行计数。完成须通过精确答案校验，因此完成率与准确率在本协议中一致。

| Arm | 题数 | 准确率 | 完成率 | 调用数 | 输入 Token | 截断调用 | 语义检出/计划 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dirty | 20 | 0.9 | 0.9 | 117 | 191938 | 0 | 10/10 |
| no_retry | 20 | 0.0 | 0.0 | 47 | 76304 | 0 | 10/10 |
| snapshot | 20 | 0.85 | 0.85 | 117 | 182883 | 0 | 10/10 |

配对差为 snapshot − dirty；95% CI 按题目成对重采样，执行与语义注入分别分层。输入 Token 为负表示节省；未知 usage 不填 0。

| 分层 | 指标 | 配对题数 | 差值 | 95% CI |
| --- | --- | ---: | ---: | --- |
| all | correct | 20 | -0.05 | [-0.2, 0.1] |
| all | completed | 20 | -0.05 | [-0.2, 0.1] |
| all | model_calls | 20 | 0.0 | [0.0, 0.0] |
| all | prompt_tokens | 20 | -452.75 | [-591.805, -330.19500000000005] |
| all | completion_tokens | 20 | -9.2 | [-39.8, 15.1] |
| all | attempts | 20 | 0.0 | [0.0, 0.0] |
| execution | correct | 10 | -0.1 | [-0.3, 0.0] |
| execution | completed | 10 | -0.1 | [-0.3, 0.0] |
| execution | model_calls | 10 | 0.0 | [0.0, 0.0] |
| execution | prompt_tokens | 10 | -342.1 | [-543.2, -186.0] |
| execution | completion_tokens | 10 | 4.8 | [-19.902499999999996, 31.0] |
| execution | attempts | 10 | 0.0 | [0.0, 0.0] |
| semantic | correct | 10 | 0.0 | [-0.3, 0.3] |
| semantic | completed | 10 | 0.0 | [-0.3, 0.3] |
| semantic | model_calls | 10 | 0.0 | [0.0, 0.0] |
| semantic | prompt_tokens | 10 | -563.4 | [-724.0, -415.39750000000004] |
| semantic | completion_tokens | 10 | -23.2 | [-76.2025, 15.502499999999964] |
| semantic | attempts | 10 | 0.0 | [0.0, 0.0] |

逐题失败、尝试分布、缺失 usage 覆盖与注入检出率详见 analysis.json。
