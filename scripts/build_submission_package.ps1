param(
    [string]$OutputFile = "",
    [string]$PythonExe = "F:\uploadtool\anaconda\envs\ToolWear_agent\python.exe"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$builder = Join-Path $PSScriptRoot "build_submission_package.py"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "未找到 ToolWear Python：$PythonExe"
}

# Python 构建器直接从白名单写 ZIP，不建立 staging 目录，也不删除任何现有文件。
$arguments = @($builder)
if ($OutputFile) {
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputFile, $repositoryRoot)
    $arguments += @("--output", $resolvedOutput)
}

& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "初赛提交包构建失败。"
}
