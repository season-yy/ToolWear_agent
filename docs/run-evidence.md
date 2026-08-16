# 真实运行证据

本文提供初赛代码包的脱敏运行证据入口。所有结果均来自同一条真实实验闭环，公开内容不包含 PHM2010 原始数据、模型权重、密钥或本机私有日志。

## 运行标识

```text
实验：刀研智协初赛演示实验
experiment_id：experiment-def463e61abd4da1b5f4c84a58878265
trace_id：trace-fb0fbc1f707642faad890d6796bd963d
run_id：run-5060d635be9bc80a9cccb7f6
```

历史实验名称只用于定位已经完成的运行记录；产品正式名称为“刃知：基于 AgentTeams 的刀具磨损监测算法辅助平台”。

## 输入与数据治理

| 项目 | 结果 |
| --- | --- |
| 数据集 | PHM2010 C1 |
| 样本来源 | 315 个 cut、7 个信号通道 |
| 窗口 | 4096 点、50% 重叠、每个 cut 最多 32 个窗口 |
| 切分 | `group_by_cut`，60% / 20% / 20% |
| 泄漏审计 | 通过，同一 cut 不跨数据集合 |
| 小样本 | 仅从训练集内部按磨损阶段比例抽取 20% |
| final test | 未用于候选、训练调整或诊断 |

## 候选、训练与评估

| 项目 | 结果 |
| --- | --- |
| LLM Provider | qwen |
| 候选数量 | 3，均通过 Registry 校验，未使用固定模板回退 |
| 人工审批方案 | `statistical_features_random_forest` |
| Validation 样本 | 2016 |
| Validation Macro-F1 | 0.937540 |
| Validation Balanced Accuracy | 0.933633 |
| 决策 | `stop`，完成小样本闭环 |
| Evidence | 28 项 |

## AgentTeams 证据

```text
AgentTeams：v1.2.2
Team：ToolWear_agent
Leader：ExperimentManagerAgent
Worker：5 个专业角色
结构化 LLM 调用：7
输出策略拒绝并恢复：1
Skill：10 个白名单工具
```

算法架构 Agent 首次调用因 Registry 上下文字段不完整被输出策略拒绝，补齐 `available_pipeline_ids` 后重试成功。失败和恢复均保留在 Trace 中，用于证明 Schema 与权限策略真实生效。

## 可公开证据

- 输入样例：`examples/create_experiment_request.json`、`examples/agentteams_task_request.json`
- 输出样例：`examples/golden_run_summary.json`、`examples/agentteams_task_result.json`
- 全流程截图：`docs/assets/demo/01-new-experiment.png` 至 `12-element-agentteams-team.png`
- AgentTeams 清单：`docs/evidence/agentteams_e2e_final_manifest.json`
- PPT 逐页映射：`docs/submission/PPT证据索引.md`

## 复验

```powershell
python -m pytest -q
python .\scripts\verify_golden_flow.py
python .\scripts\verify_agentteams_deployment.py
python .\scripts\verify_submission_readiness.py
```

真实实验复验依赖使用者通过 `AI_INFRA_ROOT` 配置的运行记录和 AgentTeams 环境；公开代码包在没有 PHM2010 原始数据时仍可执行单元测试和提交完整性检查。
