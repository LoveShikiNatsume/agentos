# Experiment C

逐请求 Token 来自服务端 usage。

全部分配题（包括失败和截断）保留在 ITT，按最终验收状态评分；截断另行计数。完成须通过精确答案校验，因此完成率与准确率在本协议中一致。

| Arm | 题数 | 准确率 | 完成率 | 调用数 | 输入 Token | 截断调用 | 语义检出/计划 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dirty | 100 | 0.96 | 0.96 | 573 | 954654 | 0 | 50/50 |
| no_retry | 100 | 0.0 | 0.0 | 223 | 364687 | 0 | 50/50 |
| snapshot | 100 | 0.88 | 0.88 | 573 | 923405 | 0 | 50/50 |

配对差为 snapshot − dirty；95% CI 按题目成对重采样，执行与语义注入分别分层。输入 Token 为负表示节省；未知 usage 不填 0。

| 分层 | 指标 | 配对题数 | 差值 | 95% CI |
| --- | --- | ---: | ---: | --- |
| all | correct | 100 | -0.08 | [-0.15, -0.02] |
| all | completed | 100 | -0.08 | [-0.15, -0.02] |
| all | model_calls | 100 | 0.0 | [0.0, 0.0] |
| all | prompt_tokens | 100 | -312.49 | [-343.16175000000004, -282.15925000000004] |
| all | completion_tokens | 100 | -0.59 | [-4.27, 3.08] |
| all | attempts | 100 | 0.0 | [0.0, 0.0] |
| execution | correct | 50 | -0.16 | [-0.3, -0.04] |
| execution | completed | 50 | -0.16 | [-0.3, -0.04] |
| execution | model_calls | 50 | 0.0 | [0.0, 0.0] |
| execution | prompt_tokens | 50 | -216.36 | [-250.9605, -183.51950000000002] |
| execution | completion_tokens | 50 | -1.12 | [-8.48, 6.34] |
| execution | attempts | 50 | 0.0 | [0.0, 0.0] |
| semantic | correct | 50 | 0.0 | [0.0, 0.0] |
| semantic | completed | 50 | 0.0 | [0.0, 0.0] |
| semantic | model_calls | 50 | 0.0 | [0.0, 0.0] |
| semantic | prompt_tokens | 50 | -408.62 | [-443.5015, -374.66] |
| semantic | completion_tokens | 50 | -0.06 | [-0.16, 0.0] |
| semantic | attempts | 50 | 0.0 | [0.0, 0.0] |

逐题失败、尝试分布、缺失 usage 覆盖与注入检出率详见 analysis.json。
