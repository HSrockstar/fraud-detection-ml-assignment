$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw '请先运行 scripts/setup.ps1 创建项目环境。' }
$env:PIP_CACHE_DIR = Join-Path $taskRoot 'artifacts\pip_cache'
& .\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch CUDA 安装失败。' }
$taskRequirements = if (Test-Path -LiteralPath 'requirements-neural-lock.txt') { 'requirements-neural-lock.txt' } else { 'requirements-neural.txt' }
& .\.venv\Scripts\python.exe -m pip install -r $taskRequirements
if ($LASTEXITCODE -ne 0) { throw '表格神经网络依赖安装失败。' }
& .\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw '依赖兼容性检查失败。' }
