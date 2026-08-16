param(
    [string]$OfficialSourceRoot = "F:\Toolwear_agent\baseline\.cache\AgentTeams-v1.2.2",
    [string]$BaseImage = "toolwear_agent/agentteams-copaw-worker:v1.2.2",
    [string]$OutputImage = "toolwear_agent/agentteams-copaw-worker:v1.2.2-teamfix"
)

$ErrorActionPreference = "Stop"

# 只构建镜像，不停止容器、不删除镜像，也不修改 AgentTeams 数据卷。
$sourceRoot = (Resolve-Path -LiteralPath $OfficialSourceRoot).Path
$officialDockerfile = Join-Path $sourceRoot "copaw\Dockerfile"
if (-not (Test-Path -LiteralPath $officialDockerfile -PathType Leaf)) {
    throw "未找到官方 CoPaw Dockerfile：$officialDockerfile"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$patchContext = Join-Path $repoRoot "deploy\agentteams\copaw"

Write-Host "[1/2] 基于本机官方 v1.2.2 源码构建基础 Worker 镜像"
docker build --file $officialDockerfile --tag $BaseImage $sourceRoot
if ($LASTEXITCODE -ne 0) {
    throw "官方 CoPaw Worker 基础镜像构建失败。"
}

Write-Host "[2/2] 应用 ToolWear Team 共享目录兼容修复"
docker build `
    --build-arg "SOURCE_IMAGE=$BaseImage" `
    --tag $OutputImage `
    $patchContext
if ($LASTEXITCODE -ne 0) {
    throw "ToolWear CoPaw Worker 镜像构建失败。"
}

Write-Host "构建完成：$OutputImage"
