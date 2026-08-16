# 初赛 Demo 演示说明

## 演示目标

用一条可恢复的 PHM2010 C1 实验展示：用户配置参数，六个 Agent 在 AgentTeams 中协作，Skill 调用真实 Tool API，模型真实训练，系统输出 validation 指标、诊断、人工决策、报告和证据。

## 启动

在 PowerShell 7 中进入仓库根目录后执行：

```powershell
.\scripts\start_local.ps1
```

入口：

- Streamlit：`http://127.0.0.1:18101`
- FastAPI：`http://127.0.0.1:18100`
- SQLite：`$AI_INFRA_ROOT/state/toolwear.db`（未配置时位于仓库 `.runtime/state`）

## 页面演示顺序

1. **系统状态**：展示 FastAPI、SQLite、LLM、CUDA、AgentTeams 和 Higress；确认固定六 Agent、Leader Ready、5/5 Worker Ready。
2. **创建/恢复实验**：选择 PHM2010 C1、四阶段分类、VB max、阈值 90/130/160 um、7 通道、窗口 4096、重叠率 0.5、小样本比例 0.2。
3. **数据治理**：运行/展示体检、标签、cut 级切分与泄漏审计。说明 315 cut、10,080 窗口、泄漏为 0。
4. **候选与审批**：输入用户目标，调用 LLM 生成 RF、ExtraTrees 和 1D-CNN 三个 PipelineSpec；展示理由、风险、成本并人工选择。
5. **训练**：批准后运行 smoke/mini train；页面展示后端、设备、日志、代码快照和 Run 状态。主 Golden Flow 可展示 RF；CUDA 证据页展示真实 1D-CNN 两 epoch loss。
6. **评估诊断**：展示 validation Macro-F1、Balanced Accuracy、分类召回、混淆矩阵、loss 和 validation-only t-SNE；明确 final test 未参与调参或 t-SNE 绘图。
7. **人工决策**：EvaluationGovernor 给出继续、调整、换方案或停止建议；用户审批一次下一动作。
8. **报告与证据**：展示 Markdown 报告、ToolWear Event Timeline、六 Agent LLM Trace、AgentTeams/Matrix 事件、Skill 审计和 Evidence 哈希。

## Golden Flow 只读复验

```powershell
python .\scripts\verify_golden_flow.py
```

预期关键结果：

```text
status: passed
experiment_id: p0-diagnosis-smoke-20260815
run_id: run-2ce31c26d36c1adabca69829
candidate_count: 3
llm_agent_count: 6
evidence hashes: 31/31
validation Macro-F1: 0.937540
validation Balanced Accuracy: 0.933633
AgentTeams/Higress: verified
```

## 异常分支演示

- 未批准 Pipeline 时训练按钮不可执行。
- 非法模块组合在 PipelineValidate 阶段返回稳定 error_code。
- 同一 cut 跨集合会被 LeakageAudit 阻断。
- 评估建议继续训练时必须进入人工审批，LLM 不能自行开始下一轮。
- AgentTeams 状态文件缺失或不完整时页面显示“待验证”，不会猜测成功。

## 演示边界

P0 只证明 C1 单刀具四阶段分类闭环。C4/C6、跨刀具迁移、DANN/MMD/CORAL、独立 VB 回归完整验证和任意代码生成属于 P1，不在本次结果中冒充完成。
