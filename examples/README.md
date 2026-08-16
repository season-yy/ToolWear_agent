# 样例输入与输出

本目录提供不含 PHM2010 原始信号、密钥和本机路径的最小样例，用于说明“刃知”的稳定接口和 AgentTeams 任务格式。

## 文件说明

| 文件 | 含义 |
| --- | --- |
| `create_experiment_request.json` | 创建 C1 四阶段磨损分类实验的 FastAPI 请求体 |
| `agentteams_task_request.json` | 人类向 ExperimentManagerAgent 提交的任务上下文 |
| `golden_run_summary.json` | 已完成真实演示 Run 的脱敏结果摘要 |
| `agentteams_task_result.json` | 六 Agent 协作与最终停止决策的脱敏输出 |

## 调用创建实验接口

先按仓库 README 启动 FastAPI，再在 PowerShell 中执行：

```powershell
$body = Get-Content .\examples\create_experiment_request.json -Raw -Encoding UTF8
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:18100/api/v1/experiments `
  -ContentType application/json `
  -Headers @{ "Idempotency-Key" = "example-create-c1-v1" } `
  -Body $body
```

如果配置了 `TOOL_API_TOKEN`，还需增加 `Authorization: Bearer <token>`。样例不会自动开始训练；后续候选、审批和训练操作由页面或受控 API 触发。

`golden_run_summary.json` 和 `agentteams_task_result.json` 是真实运行的脱敏摘要，不是伪造训练数据。完整证据关系见 `docs/run-evidence.md`。
