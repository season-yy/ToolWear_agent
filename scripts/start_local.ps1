param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = if ($env:AI_INFRA_ROOT) {
    [System.IO.Path]::GetFullPath($env:AI_INFRA_ROOT, $repositoryRoot)
}
else {
    Join-Path $repositoryRoot ".runtime"
}
$logRoot = Join-Path $runtimeRoot "logs\services"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

if (-not $PythonExe) {
    $PythonExe = $env:TOOLWEAR_PYTHON
}
if (-not $PythonExe) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $PythonExe = $pythonCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "未找到 Python。请激活项目环境，设置 TOOLWEAR_PYTHON，或通过 -PythonExe 指定解释器。"
}
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Start-ToolWearService {
    param(
        [int]$Port,
        [string[]]$Arguments,
        [string]$Name
    )

    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
        if ($process.CommandLine -notmatch "toolwear_agent|streamlit_app") {
            throw "端口 $Port 已被其他程序占用：$($process.Name)"
        }
        Write-Host "$Name 已运行：PID $($listener.OwningProcess)，端口 $Port"
        return
    }

    $process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $repositoryRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot "$Name-$stamp.out.log") `
        -RedirectStandardError (Join-Path $logRoot "$Name-$stamp.err.log") `
        -PassThru
    Write-Host "$Name 已启动：PID $($process.Id)，端口 $Port"
}

Start-ToolWearService `
    -Port 18100 `
    -Name "api" `
    -Arguments @("-m", "uvicorn", "toolwear_agent.backend.main:app", "--host", "127.0.0.1", "--port", "18100")
Start-ToolWearService `
    -Port 18101 `
    -Name "ui" `
    -Arguments @("-m", "streamlit", "run", "toolwear_agent/frontend/streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", "18101", "--server.headless", "true")

$deadline = (Get-Date).AddSeconds(30)
do {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:18100/api/v1/health" -TimeoutSec 3
        $page = Invoke-WebRequest -Uri "http://127.0.0.1:18101/" -UseBasicParsing -TimeoutSec 3
        break
    }
    catch {
        Start-Sleep -Seconds 1
    }
} while ((Get-Date) -lt $deadline)

if (-not $health -or $page.StatusCode -ne 200) {
    throw "ToolWear 服务未在 30 秒内通过健康检查，请查看 $logRoot。"
}
Write-Host "API：$($health.status)"
Write-Host "AgentTeams：$($health.components.agentteams.status)"
Write-Host "Higress：$($health.components.higress.status)"
Write-Host "页面：http://127.0.0.1:18101/"
