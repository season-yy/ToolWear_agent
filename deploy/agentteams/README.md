# AgentTeams CoPaw Worker 复现说明

本目录保存 ToolWear 当前 AgentTeams `v1.2.2` Worker 镜像的最小兼容修复，不保存 Matrix 密码、LLM API Key、Tool API Token、容器数据或 Docker volume。

## 生成可执行 Worker Skills

在项目 Python 环境中执行：

```powershell
python -m toolwear_agent official-agentteams-c1
```

命令会从 `toolwear_agent/agentteams/official_adapter.py` 和 `worker_skill_client.py` 生成 10 个 Worker Skill。每个 Skill 都包含：

```text
SKILL.md
scripts/client.py
schema/input.schema.json
schema/output.schema.json
```

默认生成目录为 `$AI_INFRA_ROOT/agentteams/phm2010_c1_minimal/worker-skills`，并同步安装到 Manager workspace。生成包不写入 API Key 或 Tool Token；客户端运行时只从环境变量或外部 secret 文件读取凭据。

## 为什么需要修复

当前 Team 的 Kubernetes 资源名是 `toolwear-phm2010-c1-team`，运行时名称是 `ToolWear_agent`。AgentTeams 为 MinIO 下发的共享前缀以运行时名称为准，而 CoPaw `v1.2.2` 默认从资源名推导共享目录，导致 Worker 启动后访问错误前缀并收到 403。

修复后的 Worker 优先读取 AgentTeams 下发的 `runtime/runtime.yaml` 中 `storage.sharedPrefix`，读取失败才回退到官方原逻辑。同时统一入口 shell 脚本的 LF 换行，避免 Windows 工作区出现 `env: bash\r`。

## 构建

```powershell
.\deploy\agentteams\build_worker_image.ps1
```

构建脚本只生成镜像，不停止或删除现有容器。默认要求本机已存在官方 `v1.2.2` 源码目录；当前验证源码标签为 `v1.2.2`，提交为 `849182a`。

## 运行验证

Team 和 Skill 已运行后，在仓库根目录执行：

```powershell
python .\scripts\verify_agentteams_deployment.py
```

该命令只读查询 Docker 与 `agt`，并把脱敏结果写入 `$AI_INFRA_ROOT/agentteams/status.json` 和对应 evidence 目录。
