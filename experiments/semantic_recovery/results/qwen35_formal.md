# Experiment C

逐请求 Token 来自服务端 usage。

全部分配题（包括失败和截断）保留在 ITT，按最终验收状态评分；截断另行计数。完成须通过精确答案校验，因此完成率与准确率在本协议中一致。

| Arm | 题数 | 准确率 | 完成率 | 调用数 | 输入 Token | 截断调用 | 语义检出/计划 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dirty | 100 | 0.89 | 0.89 | 573 | 973225 | 0 | 50/50 |
| no_retry | 100 | 0.0 | 0.0 | 223 | 367801 | 0 | 50/50 |
| snapshot | 100 | 0.87 | 0.87 | 573 | 931647 | 0 | 50/50 |

配对差为 snapshot − dirty；95% CI 按题目成对重采样，执行与语义注入分别分层。输入 Token 为负表示节省；未知 usage 不填 0。

| 分层 | 指标 | 配对题数 | 差值 | 95% CI |
| --- | --- | ---: | ---: | --- |
| all | correct | 100 | -0.02 | [-0.09, 0.05] |
| all | completed | 100 | -0.02 | [-0.09, 0.05] |
| all | model_calls | 100 | 0.0 | [0.0, 0.0] |
| all | prompt_tokens | 100 | -415.78 | [-479.94175, -355.51] |
| all | completion_tokens | 100 | 16.33 | [4.949750000000001, 27.61074999999999] |
| all | attempts | 100 | 0.0 | [0.0, 0.0] |
| execution | correct | 50 | 0.04 | [-0.04, 0.12] |
| execution | completed | 50 | 0.04 | [-0.04, 0.12] |
| execution | model_calls | 50 | 0.0 | [0.0, 0.0] |
| execution | prompt_tokens | 50 | -268.86 | [-334.161, -210.97850000000003] |
| execution | completion_tokens | 50 | 29.42 | [16.0, 44.3405] |
| execution | attempts | 50 | 0.0 | [0.0, 0.0] |
| semantic | correct | 50 | -0.08 | [-0.2, 0.02] |
| semantic | completed | 50 | -0.08 | [-0.2, 0.02] |
| semantic | model_calls | 50 | 0.0 | [0.0, 0.0] |
| semantic | prompt_tokens | 50 | -562.7 | [-657.1235, -472.619] |
| semantic | completion_tokens | 50 | 3.24 | [-13.381, 20.62] |
| semantic | attempts | 50 | 0.0 | [0.0, 0.0] |

逐题失败、尝试分布、缺失 usage 覆盖与注入检出率详见 analysis.json。
