# 核心 Skill 清单

每个 Skill 均生成 `SKILL.md`、`scripts/client.py`、`schema/input.schema.json` 和 `schema/output.schema.json`。Worker 通过固定 HTTP 白名单调用 ToolWear FastAPI；POST 必须携带 Bearer Token、`Idempotency-Key` 和 `confirm_write=true`，API 还会校验 Agent 与 Skill 的归属关系。

| Skill | Owner | 用途与主要输出 | 调用条件 | 失败处理 | 复用价值 |
| --- | --- | --- | --- | --- | --- |
| `toolwear-data-profile` | DataSteward | 登记数据、体检通道/采样率/文件数量，返回 DataProfile 与 EvidenceRef | 实验已创建且数据集可用 | 数据缺失、Schema 不符或鉴权失败时不推进状态 | 可复用于任意实现 Dataset Adapter 的状态监测数据 |
| `toolwear-stage-label` | DataSteward | 按 VB 聚合策略和阈值生成磨损阶段标签 | Profile 已通过，LabelPolicy 合法 | 标签缺失或阈值非法时返回稳定 error_code | 标签策略可替换，不与 PHM2010 文件名绑定 |
| `toolwear-window-split` | DataSteward | cut 级切分、滑窗、抽样和泄漏审计 | 标签已生成，SplitSpec/窗口参数合法 | 检出交叉 cut、锁冲突或样本不足时阻断训练 | 可复用于按设备/工件/批次分组防泄漏的时序任务 |
| `toolwear-pipeline-recommend` | AlgorithmArchitect | 结合目标、数据、GPU、Registry 与 Memory 生成 2-3 个 PipelineSpec | 数据准备完成且 Module Registry 可用 | LLM 输出先做 Schema 和 Registry 校验；失败后重试/切换模型 | 新增模块只需注册，无需改 Agent 协议 |
| `toolwear-human-selection` | ExperimentManager | 保存用户候选选择与审批记录 | 处于等待方案选择状态 | 无候选、重复审批或非法状态时拒绝 | 适用于所有高风险或高成本的人机审批节点 |
| `toolwear-mini-train` | CodeTrainingEngineer | 验证 Pipeline、生成 RunBundle、真实 smoke/mini train、记录日志与模型 | Pipeline 已批准且预算允许 | OOM/shape/import/timeout 记录失败证据，不覆盖旧 Run | sklearn/PyTorch Trainer 可通过 Registry 扩展 |
| `toolwear-visualization` | EvaluationGovernor | 生成混淆矩阵、loss、t-SNE 等评估产物 | Run 成功且存在可验证指标 | 单个可选图失败不伪造，保留主评估并记录降级 | 图表生成与模型解耦，可复用于其他分类/回归任务 |
| `toolwear-diagnosis` | EvaluationGovernor | 确定性规则先行，再由 LLM 解释 validation 失败模式 | EvaluationReport 已形成 | 指标缺失、冲突或 test 污染时降低置信度/阻断 | 可复用于不同模型的统一诊断契约 |
| `toolwear-decision` | EvaluationGovernor | 生成继续、调参、换结构、完整训练或停止建议 | 诊断完成且预算可计算 | 所有训练性动作强制 `requires_human_approval=true` | 决策策略可替换但状态机和审计保持稳定 |
| `toolwear-report-trace` | ReportMemoryCurator | 生成 Markdown 报告、Evidence 索引和 MemoryCase | 证据可索引且哈希有效 | 缺失证据时只列缺口，不改写原始指标 | 可作为科研实验报告和后续案例检索的统一输出 |

## 等价工具集成契约

P0 没有单独部署 MCP Server，采用可直接迁移到 MCP 的等价契约：

- 协议：HTTP/JSON，稳定 `/api/v1` 路由。
- 鉴权：Tool API Bearer Token 从容器 Secret/环境读取，不进入 Skill 文档、日志或提交包。
- Schema：Pydantic 业务模型 + 每个 Skill 的 JSON Schema。
- 权限：Agent/Skill 白名单、允许主机白名单和路径白名单。
- 可靠性：超时、有限重试、稳定 error_code、POST 幂等键。
- 审计：记录 Agent、Skill、correlation id、trace id、耗时、结果与 EvidenceRef。
- MCP 迁移：保留业务 Schema 与审计信封，只新增 MCP transport adapter，无需重写确定性工具链。

实现来源：`toolwear_agent/agentteams/worker_skill_client.py` 与 `toolwear_agent/agentteams/official_adapter.py`。
