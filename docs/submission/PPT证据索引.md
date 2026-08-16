# “刃知”初赛 PPT 证据索引

本索引将 19 页初赛方案中的关键陈述映射到代码、干净截图和真实运行记录。公开仓库只保存可公开材料；使用者在 `$AI_INFRA_ROOT` 配置的原始数据、模型、日志和私有运行产物仅用于本地复验。

## 统一实验标识

```text
实验名称：刀研智协初赛演示实验
experiment_id：experiment-def463e61abd4da1b5f4c84a58878265
trace_id：trace-fb0fbc1f707642faad890d6796bd963d
run_id：run-5060d635be9bc80a9cccb7f6
审批方案：statistical_features_random_forest
```

历史实验名称用于识别已经完成的真实 Run，不随产品中文名“刃知”一同改写。

## 页面与证据

| 页码 | 页面主题 | 关键结论 | 主要证据 |
| ---: | --- | --- | --- |
| 1 | 刃知封面 | 项目正式名称和 P0 演示范围 | `README.md`、`docs/submission/初赛作品简介.md` |
| 2 | 一页总览 | 六 Agent、真实训练、人工审批和 Evidence 构成闭环 | `docs/architecture.md`、`docs/competition-mapping.md` |
| 3 | 目录 | 方案按官方评分方向组织 | 官方初赛模板、`docs/competition-mapping.md` |
| 4 | 场景与价值 | 把算法试错变成可恢复、可审批、可审计流程 | `toolwear_agent/state/`、`toolwear_agent/backend/` |
| 5 | 实验定义与数据治理 | 页面可配置数据、标签、通道、滑窗和小样本策略；cut 级切分无泄漏 | `docs/assets/demo/01-new-experiment.png`、`02-experiment-defined.png`、`03-data-prepared.png`、`toolwear_agent/training/` |
| 6 | 方案总览 | AgentTeams 负责协作，FastAPI/Registry 执行确定性业务 | `docs/architecture.md`、`toolwear_agent/agentteams/`、`toolwear_agent/backend/` |
| 7 | 七阶段状态轨道 | 实验从定义、准备、候选、审批、训练、评估到决策归档 | `toolwear_agent/state/`、`toolwear_agent/frontend/experiment_workspace.py` |
| 8 | 多 Agent 协同 | 六个 Agent 具有独立身份、输入输出、权限和失败边界 | `docs/agent-identities.md`、`toolwear_agent/agents/catalog.py` |
| 9 | AgentTeams 实证 | 六 Agent 共 7 条真实 LLM 调用；一次输出策略拒绝后补齐 Registry 输入并重试成功 | `docs/assets/demo/11-agent-collaboration.png`、`12-element-agentteams-team.png`、`docs/evidence/agentteams_e2e_final_manifest.json` |
| 10 | Skill 工程体系 | 自然语言建议必须通过 Registry 与 Tool API 契约才能执行 | `docs/skill-manifest.md`、`toolwear_agent/agentteams/official_adapter.py` |
| 11 | Skill 契约实证 | 10 个 Skill 均有 Owner、文档、客户端、输入输出 Schema、权限和失败边界，并关联真实 AgentTeams 调用上下文 | `docs/assets/demo/13-skill-registry-evidence.png`、`docs/skill-manifest.md`、`docs/evidence/agentteams_e2e_final_manifest.json` |
| 12 | 工程验证与安全 | 状态机、幂等、审批、final test 隔离、Evidence 哈希和失败恢复由代码约束 | `docs/security.md`、`toolwear_agent/delivery/`、`tests/` |
| 13 | 真实 Demo | 真实候选、训练、指标和诊断来自同一 experiment/trace/run | `docs/assets/demo/04-llm-candidates.png`、`06-training-result.png`、`07-evaluation-metrics.png`、`08-evaluation-diagnosis.png` |
| 14 | 开放与开源 | 公开代码、配置样例、测试、文档和复验入口 | `README.md`、`.env.example`、`pyproject.toml`、`tests/` |
| 15 | 开源边界 | 不发布密钥、原始数据、模型权重和私有日志；说明依赖与可替换边界 | `.gitignore`、`docs/security.md`、`scripts/build_submission_package.py` |
| 16 | 当前进展 | P0 已完成前端交互、真实训练、AgentTeams、诊断、报告和证据链 | `项目进展说明.md`、`scripts/verify_golden_flow.py` |
| 17 | 停止决策 | EvaluationGovernorAgent 识别过拟合差距和类别不平衡，用户停止额外训练并归档 28 项 Evidence | `docs/assets/demo/08-evaluation-diagnosis.png`、`09-decision-archive.png`、`10-evidence-timeline.png` |
| 18 | 团队与未来 | 机械领域需求驱动，工程结论以代码、测试和 Evidence 为准 | `README.md`、`docs/项目使用说明.md` |
| 19 | 路线图 | P1 扩展 C4/C6 与迁移学习，P1.5/P2 扩展回归、融合和企业治理 | `README.md`“当前边界与路线”、`docs/competition-mapping.md` |

## 真实运行结果

```text
数据：PHM2010 C1，315 个 cut，7 通道
窗口：4096 点，50% 重叠，每个 cut 最多 32 个窗口
切分：cut 级 60% / 20% / 20%，泄漏审计通过
小样本：仅在训练集内按四阶段比例抽取 20%
候选：3 个，Provider=qwen，Fallback=False
Validation 样本：2016
Validation Macro-F1：0.937540
Validation Balanced Accuracy：0.933633
训练 F1：0.997823
决策：stop
Evidence：28 项
final test：未读取
```

## AgentTeams 证据

```text
框架：AgentTeams v1.2.2
Team 资源：toolwear-phm2010-c1-team
Team 运行名：ToolWear_agent
Leader：toolwear-experiment-manager
普通 Worker：5 个专业角色
Skill：10 个白名单工具
Higress Provider：qwen-toolwear
当前模型：qwen3.7-flash-2026-07-15
```

Element 截图只展示 `Team: ToolWear_agent`、`toolwear-*` 角色和 Skill 状态，不包含桌面、浏览器外框、账号密码或无关 baseline Team。

## 复验命令

```powershell
python -m pytest -q
python .\scripts\verify_golden_flow.py
python .\scripts\verify_agentteams_deployment.py
```

## 边界说明

- 所有定量指标只来自 validation；final test 未参与候选、调参、诊断或本轮停止决策。
- t-SNE 只使用 validation 特征，仅用于观察分布，不单独作为继续训练依据。
- CUDA 1D-CNN 已验证训练链路，但本轮批准与归档方案是 RandomForest。
- LLM 负责候选、解释和诊断；训练、指标、状态转换和 SHA-256 由确定性代码完成。
- 公开仓库不包含 `.env`、API Key、Token、Matrix 密码、PHM2010 原始数据、模型权重、Docker volume、缓存和私有日志。
