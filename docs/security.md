# 安全、合规与发布边界

## 当前安全措施

- `.env`、Token、Matrix 凭据、Docker volume、原始 PHM2010、模型权重、缓存、日志和运行时均不进入 Git/提交包。
- AgentTeams Worker 只能访问固定 ToolWear API 主机和固定路由；Skill 与 Agent 归属由 API 再次校验。
- 所有写操作要求 Bearer Token、显式确认和幂等键。
- 原始数据只读；生成文件只能写入配置的 `AI_INFRA_ROOT`。
- 训练使用 Registry 后端和受控参数，不允许模型直接执行任意命令。
- 日志对敏感字段脱敏，Agent Trace 不保存 API Key。
- 发布脚本采用白名单，不采用“整个目录压缩”；构建前扫描密钥模式和危险后缀。

## 数据与模型披露

- Demo 数据：PHM 2010 公共刀具磨损数据集；原始数据不随代码包分发，使用者需自行取得并遵守数据集许可。
- LLM：Qwen 商业 API，通过 OpenAI-compatible 接口/Higress 调用；密钥由使用者配置。
- 可替代性：LLM Provider 已抽象，兼容同类 OpenAI-compatible 模型；替换模型需要重新做结构化输出 smoke。
- 训练模型：RandomForest、ExtraTrees 和 PyTorch 1D-CNN；模型文件属于运行产物，不默认发布。

## 人工审批与回滚

- 候选选择、结构变更、继续调参、完整训练、超预算和报告发布均需人工确认。
- Experiment revision 不覆盖历史；失败 Run 和旧模型保留。
- 幂等记录防止页面双击或网络重试重复触发写操作。
- Cancel 通过状态机进入可审计状态，不直接删除运行目录。

## 开放计划

计划开放：六 Agent Identity/Prompt、Pipeline/Skill/API Schema、Dataset/Module/Trainer Registry、状态机、测试、一键启动、部署说明和脱敏 Evidence 样例。暂不开放：第三方密钥、原始数据、私人实验数据、Docker volume 和本机模型权重。开源协议将在正式公开仓库前由团队确认，第三方依赖按各自许可证使用。
