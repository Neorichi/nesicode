# Builds dist\nesicode\ (PyInstaller) and the installer dist\nesicode-<version>-windows-x64-setup.exe
# Needs: Python/uv, Node 20+, Inno Setup 6 (iscc on PATH or in Program Files).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$version = (Select-String -Path pyproject.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value

Write-Host "==> web"
Push-Location web; npm ci --silent; npm run build --silent; Pop-Location

Write-Host "==> python deps"
uv sync --extra desktop --group build

Write-Host "==> bundle"
Remove-Item -Recurse -Force build, dist\nesicode -ErrorAction SilentlyContinue
uv run pyinstaller packaging/nesicode.spec --noconfirm --log-level WARN

Write-Host "==> installer"
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
& $iscc "/DAppVersion=$version" packaging\windows\nesicode.iss
Write-Host "OK: dist\Nesicode-$version-windows-x64-setup.exe"
