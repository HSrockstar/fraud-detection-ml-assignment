param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -c 'import sys; assert sys.version_info[:2] == (3, 12), "请使用 Python 3.12 创建环境。"'
    if ($LASTEXITCODE -ne 0) { throw 'Python 版本不匹配，请使用 -Python 指定 Python 3.12。' }
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw '创建 .venv 失败，请指定 Python 3.12。' }
}
& .\.venv\Scripts\python.exe -c 'import sys; assert sys.version_info[:2] == (3, 12), "已有 .venv 不是 Python 3.12。"'
if ($LASTEXITCODE -ne 0) { throw '已有 .venv 版本不匹配。' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip 更新失败。' }
$taskRequirements = if (Test-Path -LiteralPath 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
& .\.venv\Scripts\python.exe -m pip install -r $taskRequirements
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败。' }
Write-Host '环境已就绪，运行 .\.venv\Scripts\python.exe scripts\run.py --help 查看命令。'
