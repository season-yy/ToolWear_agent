# GOAI Agent Infra 初赛要求映射

依据：`D:\desk\AI infra 比赛要求.pdf` 与官方赛道页面。初赛必交作品简介和方案 PPT/PDF；代码包可选。本项目额外提交可运行脱敏代码包和真实证据，以降低“仅概念方案”的风险。

## 核心要求映射

| 比赛要求 | ToolWear 实现 | 验证证据 | 状态 |
| --- | --- | --- | --- |
| 真实行业场景与可复制价值 | 面向机械加工科研/状态监测的多源刀具磨损算法研发；Dataset Adapter、PipelineSpec、Registry 使流程不绑定 C1 | `README.md`、`docs/architecture.md`、前端创建实验页 | 已完成 P0；跨刀具是 P1 |
| 不少于 3 个不同职能 Agent | 固定六 Agent，各有 Identity、Prompt、Schema、Skill 权限和失败行为 | `docs/agent-identities.md`、`toolwear_agent/agents/catalog.py` | 已完成 |
| 以 AgentTeams 为协同基点 | 官方 AgentTeams v1.2.2 Team；Leader + 5 Workers；Matrix 真实分派；Higress Qwen 路由 | `docs/evidence/agentteams_e2e_final_manifest.json`、`deployment_status.py` | 已验证 |
| 任务拆解、上下文、协同与状态追踪 | Team Leader 生成 DAG；Worker 共享 experiment/trace/correlation/Evidence；SQLite 状态机记录每次 transition | AgentTeams E2E 报告、SQLite events、前端 Timeline | 已完成 |
| 完整闭环 | 输入 -> 数据 -> 候选 -> 人工审批 -> 训练 -> validation 评估 -> 决策 -> 报告/Memory | Golden Flow 报告和前端状态页 | 已通过只读复验 |
| Skill 必选且工程化 | 10 个 Skill，每个含文档、可执行 client、输入/输出 JSON Schema | `docs/skill-manifest.md`、`official_adapter.py`、Skill 审计日志 | 已完成 |
| 工具稳定接入 | FastAPI `/api/v1` + Bearer Token + 角色/Skill 权限 + timeout/retry/idempotency/error mapping | `worker_skill_client.py`、Tool API 安全测试 | 已完成 |
| 未使用 MCP 时给出等价契约 | HTTP/JSON 等价契约，业务 Schema 与 transport 分离，后续仅新增 MCP adapter | `docs/skill-manifest.md` | 已说明，MCP Server 为 P1 |
| RAG/上下文至少 2 项 | SQLite Memory/FTS5、共享 ExperimentState、轨迹与证据持久化，共 3 项 | health 的 FTS5、state repository、Event/Evidence | 已完成轻量实现 |
| 可观测 | Agent/Skill/LLM/训练事件含 Trace、结构化 Log、指标、耗时、Token 和 EvidenceRef | Golden Flow、Agent call JSON、Skill audit、run logs | 已完成 |
| 结果验证 | Macro-F1、Balanced Accuracy、per-class、混淆矩阵、loss；Golden Flow 复算 31 个 SHA-256 | Run result、Golden Flow report | 已完成 |
| 审批、回滚与审计 | Pipeline、结构、完整训练和报告发布审批；revision 不覆盖历史；幂等与 cancel | approvals/state events/tests | 已完成 P0 |
| 开放/开源计划 | 开放 Agent/Skill/API/Registry/状态机/测试/部署与脱敏样例；不开放密钥、原始数据和私有产物 | `docs/security.md` | 已说明 |
| 安全发布 | 白名单 ZIP、secret scan、危险后缀/路径/大文件拦截、拒绝覆盖已有包 | `scripts/build_submission_package.ps1` 与测试 | 已完成 |

## 多 Agent 闭环八项

1. **任务输入**：前端接收数据集、任务、标签、通道、窗口、预算和自然语言目标。
2. **任务拆解**：ExperimentManagerAgent 通过 AgentTeams DAG 分派五类专业任务。
3. **上下文传递**：ExperimentState、revision、PipelineSpec、AgentResult、EvidenceRef 和 Matrix 任务文件。
4. **工具调用**：Worker Skill 经 FastAPI 调 Dataset/Registry/Training/Evaluation/Report 服务。
5. **结果验证**：validation 指标、混淆矩阵、loss、确定性规则和 LLM 诊断。
6. **执行证据沉淀**：SQLite、JSONL、代码快照、模型、图表、报告、AgentTeams 事件和 SHA-256。
7. **审批与回滚**：候选/结构/完整训练/发布审批，revision 和旧 Run 保留。
8. **经验沉淀**：ReportMemoryCurator 生成 Markdown 报告和 SQLite MemoryCase。

## 评分维度对应

- **场景价值与行业复制性 25%**：将领域专家的算法试错过程产品化；Adapter/Registry 支持相似状态监测任务迁移。
- **多 Agent 协同与闭环 25%**：六角色真实 LLM + AgentTeams/Matrix + Human approval + 状态机。
- **Skill 工程与生态复用 25%**：10 个可执行 Skill、Schema、权限、失败处理和等价 MCP 契约。
- **工程落地、安全审计 20%**：真实训练、CUDA、FastAPI、SQLite、Trace、Evidence、测试和安全发布。
- **开放/开源贡献 5%**：开放模板、接口、测试与脱敏样例，披露商业 API 和数据边界。

## 推荐组件取舍

- 已使用：AgentTeams、Higress、SQLite/FTS5、ToolWear 可执行 Skills。
- P0 不引入：Nacos、PolarDB、RocketMQ、Milvus。比赛不按堆叠数量评分，当前以本地可运行、接口清晰和证据完整为优先。
- 后续迁移：配置/Prompt Registry 可接 Nacos；Repository 可接 PostgreSQL/PolarDB；事件接口可接 RocketMQ；工具 transport 可接 MCP。

## 真实演示指标

- PHM2010 C1：315 cut、7 通道、50 kHz、10,080 个窗口。
- cut 级切分：train/validation/test 为 188/63/64 cut，对应 6,016/2,016/2,048 窗口，泄漏问题 0。
- 20% train 小样本：1,205 窗口，固定 seed 42，覆盖 188 个 train cut。
- RF Golden Flow：validation Macro-F1 `0.937540`，Balanced Accuracy `0.933633`，final test 未运行。
- CUDA 1D-CNN：真实 2 epoch，设备 NVIDIA GeForce RTX 5070 Ti Laptop GPU，loss 与 checkpoint 已保存；该 smoke 用于证明深度训练链路，不冒充最佳模型。
