param(
    [string]$OutputFile = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$builder = Join-Path $PSScriptRoot "build_submission_package.py"

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
