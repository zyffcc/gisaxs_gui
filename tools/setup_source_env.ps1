# Sets up GIMaP in a source folder (a git clone, or a ZIP of the repository) and starts it - no Python
# knowledge needed. Start_GIMaP.bat calls this script when the folder has no portable runtime\.
#
#   Double-click Start_GIMaP.bat
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_source_env.ps1             set up, start
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_source_env.ps1 -NoStart    set up only
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_source_env.ps1 -Reinstall  build again
#
# The first run downloads uv (a small Python installer from GitHub, the pinned version below, checked
# against its published SHA-256), Python 3.10 and the packages of requirements.txt into the folder
# .gimap\ of this checkout: about 2 GB, 3-20 minutes depending on the connection. Later runs start GIMaP
# at once. The packages of requirements-sputter.txt (PyTorch, CPU) are added too, for the Sputter toolkit
# (research code bundled in utils\sputter_toolkit). When a requirements file changed (after an update)
# the missing packages are added before the start.
# Nothing outside this folder is changed (settings live in %APPDATA%\GIMaP as always). Delete .gimap\
# to remove the environment completely.
#
# Exit code: 0 started (or set up with -NoStart), 1 failed (the reason is printed).
# Keep this file ASCII: Windows PowerShell 5.1 reads a script without a byte-order mark in the system
# code page.
[CmdletBinding()]
param(
    # Set up or update the environment, but do not start GIMaP.
    [switch]$NoStart,
    # Delete .gimap\env and build it again.
    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"
# GIMAP_SETUP_NOSTART=1 sets up without starting (checks, automation), also when called by Start_GIMaP.bat.
if ($env:GIMAP_SETUP_NOSTART -eq "1") { $NoStart = $true }
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with its progress bar in PS 5.1
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Base = Join-Path $Root ".gimap"
$ToolDir = Join-Path $Base "tools"
$EnvDir = Join-Path $Base "env"
$PythonExe = Join-Path $EnvDir "Scripts\python.exe"
$PythonWExe = Join-Path $EnvDir "Scripts\pythonw.exe"
$StampFile = Join-Path $EnvDir "gimap_requirements.sha256"
$UvVersion = "0.5.11"
$UvAsset = "uv-x86_64-pc-windows-msvc.zip"
$UvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$UvAsset"
$PythonVersion = "3.10"
$RequirementNames = @("requirements.txt")
# The Sputter workspace's bundled toolkit runs in this environment (skipped if its folder was removed).
$SputterEntry = Join-Path $Root "utils\sputter_toolkit\scripts\sputter_struct_v1.py"
if (Test-Path -LiteralPath $SputterEntry -PathType Leaf) { $RequirementNames += "requirements-sputter.txt" }

function Say([string]$Text) { Write-Host "[GIMaP] $Text" }

function Fail([string]$Text) {
    Write-Host ""
    Write-Host "[GIMaP] SETUP FAILED: $Text"
    Write-Host "[GIMaP] Nothing outside $Base was changed. Run Start_GIMaP.bat again to retry;"
    Write-Host "[GIMaP] delete the folder .gimap to start from scratch."
    exit 1
}

function Get-Uv {
    $Uv = Join-Path $ToolDir "uv.exe"
    if (Test-Path -LiteralPath $Uv -PathType Leaf) { return $Uv }
    New-Item -ItemType Directory -Force -Path $ToolDir | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $Zip = Join-Path $ToolDir $UvAsset
    Say "Downloading uv $UvVersion (about 15 MB) from GitHub ..."
    Invoke-WebRequest -UseBasicParsing -Uri $UvUrl -OutFile $Zip
    $Published = (Invoke-WebRequest -UseBasicParsing -Uri "$UvUrl.sha256").Content
    if ($Published -is [byte[]]) { $Published = [Text.Encoding]::ASCII.GetString($Published) }
    $Expected = ($Published -split '\s+')[0].Trim().ToLowerInvariant()
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Zip).Hash.ToLowerInvariant()
    if ($Expected -notmatch '^[0-9a-f]{64}$' -or $Actual -ne $Expected) {
        Remove-Item -LiteralPath $Zip -Force
        Fail "the downloaded uv does not match its published SHA-256 (expected $Expected, got $Actual)."
    }
    Expand-Archive -LiteralPath $Zip -DestinationPath $ToolDir -Force
    Remove-Item -LiteralPath $Zip -Force
    if (-not (Test-Path -LiteralPath $Uv -PathType Leaf)) { Fail "uv.exe was not found in the downloaded archive." }
    return $Uv
}

function Get-RequirementsStamp {
    $Lines = foreach ($Name in $RequirementNames) {
        $Path = Join-Path $Root $Name
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            "$Name $((Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash)"
        }
    }
    return ($Lines -join "`n")
}

function Invoke-Uv([string]$Uv, [string[]]$Arguments, [string]$What) {
    # uv writes its progress to stderr; with "Stop" Windows PowerShell would end at the first line.
    $ErrorActionPreference = "Continue"
    & $Uv @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "$What (uv exit code $LASTEXITCODE). See the messages above." }
}

try {
    Set-Location -LiteralPath $Root
    foreach ($Required in @("main.py", "requirements.txt")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $Required) -PathType Leaf)) {
            Fail "$Required was not found next to the tools folder: this is not a GIMaP source folder."
        }
    }
    # Everything uv downloads stays in this checkout.
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Base "python"
    $env:UV_CACHE_DIR = Join-Path $Base "cache"
    $env:UV_LINK_MODE = "copy"
    $env:UV_PYTHON_PREFERENCE = "only-managed"

    if ($Reinstall -and (Test-Path -LiteralPath $EnvDir)) {
        Say "Removing the old environment ..."
        Remove-Item -LiteralPath $EnvDir -Recurse -Force
    }
    $Wanted = Get-RequirementsStamp
    $Have = if (Test-Path -LiteralPath $StampFile) { (Get-Content -LiteralPath $StampFile -Raw).Trim() } else { "" }
    if (-not (Test-Path -LiteralPath $PythonWExe -PathType Leaf) -or $Have -ne $Wanted.Trim()) {
        $Uv = Get-Uv
        if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
            Say "First start: setting up Python $PythonVersion and the packages GIMaP needs."
            Say "This installs about 2 GB and takes 3-20 minutes; later starts are immediate."
            Invoke-Uv $Uv @("python", "install", $PythonVersion) "Python $PythonVersion could not be installed"
            Invoke-Uv $Uv @("venv", "--quiet", $EnvDir, "--python", $PythonVersion) "the environment could not be created"
        }
        else {
            Say "The requirements changed (an update, or the Sputter toolkit was added): adding the missing packages ..."
        }
        $InstallArgs = @("pip", "install", "--python", $PythonExe)
        foreach ($Name in $RequirementNames) {
            $Path = Join-Path $Root $Name
            if (Test-Path -LiteralPath $Path -PathType Leaf) { $InstallArgs += @("-r", $Path) }
        }
        Invoke-Uv $Uv $InstallArgs "the packages could not be installed"
        Set-Content -LiteralPath $StampFile -Value $Wanted -Encoding ASCII
        # The downloaded package files are no longer needed: the environment holds copies.
        if (Test-Path -LiteralPath $env:UV_CACHE_DIR) {
            Remove-Item -LiteralPath $env:UV_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue
        }
        Say "The environment is ready: $EnvDir"
    }

    if ($NoStart) { exit 0 }
    $env:PYTHONNOUSERSITE = "1"
    $MplDir = Join-Path $env:LOCALAPPDATA "GIMaP\matplotlib"
    New-Item -ItemType Directory -Force -Path $MplDir | Out-Null
    $env:MPLCONFIGDIR = $MplDir
    Say "Starting GIMaP ..."
    Start-Process -FilePath $PythonWExe -ArgumentList @("main.py") -WorkingDirectory $Root
    exit 0
}
catch {
    Fail $_.Exception.Message
}
