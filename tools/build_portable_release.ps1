# Builds the portable Windows release: the packed Conda environment plus the application files.
#
# The application files are an allow list, never "everything in the folder":
#   1. the files git tracks, and
#   2. the model files 1D Predict needs at run time that git ignores (*.h5, *.npz, *.keras under
#      modules/, outside the development records).
# A developer's folder also holds personal settings (config/user_*.json), the instrument geometry of
# their own beamtime (config/instrument_profiles.json), reports, research folders, notebooks and
# caches. None of them may reach a release: GIMaP imports <program>/config/*.json into a new user's
# data folder on the first start, so a shipped profile would be applied to that person's frames.
# The script stops when such a path is in the list.
#
# Check the list without building the environment (about 5 GB):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_portable_release.ps1 -ListOnly
# With -ListOnly the standard output is the list alone (one path per line); the summary and any
# problem go to the error stream, and the exit code is 1 when there is a problem.
#
# Keep this file ASCII: Windows PowerShell 5.1 reads a script without a byte-order mark in the
# system code page.
[CmdletBinding()]
param(
    [string]$EnvironmentName = "GUI",
    # Default: the version the program reports (src/gimap/shared/version.py).
    [string]$Version = "",
    [string]$OutputDirectory = "release",
    [int]$ReleasePartSizeMiB = 1900,
    [string]$GitHubRepository = "zyffcc/gisaxs_gui",
    [string]$ReleaseTag = "",
    # Print the application file list and stop: nothing is packed, nothing is written.
    [switch]$ListOnly,
    # Build although the program folders hold files git does not track (they are left out).
    [switch]$AllowUntracked
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Require-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. $InstallHint"
    }
}

function Write-Note([string]$Text) {
    if ($ListOnly) { [Console]::Error.WriteLine($Text) } else { Write-Host $Text }
}

function Get-GitPaths([string[]]$Arguments) {
    # git writes paths in UTF-8; Windows PowerShell would read them in the console code page.
    $PreviousEncoding = [Console]::OutputEncoding
    try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }
    try {
        $Lines = & git -C $ProjectRoot -c core.quotepath=false @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw ("git $($Arguments -join ' ') failed (exit code $LASTEXITCODE). The release is built " +
                "from a git checkout: the application files are the ones git tracks.")
        }
    }
    finally {
        try { [Console]::OutputEncoding = $PreviousEncoding } catch { }
    }
    return @($Lines | Where-Object { $_ })
}

function Test-AnyPattern([string]$Path, [string[]]$Patterns) {
    foreach ($Pattern in $Patterns) {
        if ($Path -match $Pattern) { return $true }
    }
    return $false
}

# ---------------------------------------------------------------------------------------------
# The version: one place (src/gimap/shared/version.py) unless -Version says otherwise.
# ---------------------------------------------------------------------------------------------
$ProgramVersion = ""
$VersionFile = Join-Path $ProjectRoot "src\gimap\shared\version.py"
if (Test-Path -LiteralPath $VersionFile) {
    $VersionLine = Select-String -LiteralPath $VersionFile -Pattern '^__version__\s*=\s*"([^"]+)"' |
        Select-Object -First 1
    if ($VersionLine) {
        $ProgramVersion = $VersionLine.Matches[0].Groups[1].Value
    }
}
if (-not $Version) {
    if ($ProgramVersion) { $Version = $ProgramVersion } else { $Version = "dev" }
}
$OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputDirectory))
$PackageName = "GIMaP-$Version-windows-x64-portable"
$ArchivePath = Join-Path $OutputRoot "$PackageName.zip"
if (-not $ReleaseTag) {
    $ReleaseTag = "v$Version"
}

# ---------------------------------------------------------------------------------------------
# The application file list.
# ---------------------------------------------------------------------------------------------
Require-Command "git" "Install Git for Windows: the application file list is what git tracks."

# Tracked, but not part of the application: working papers and development records.
$NotShipped = @(
    '^docs/research/',
    '^work/',
    '^GISAXS_HANDOFF_[^/]*\.md$'
)
# Never in a release, whatever git says. Paths are relative to the project root, with "/".
$Forbidden = @(
    @{ Pattern = '^config/user_[^/]*\.json$'; Why = "a person's own settings" },
    @{ Pattern = '^config/instrument_profiles\.json$'; Why = "a person's own instrument geometry" },
    @{ Pattern = '(^|/)(_handoff|research|validation)/'; Why = "personal or research material" },
    @{ Pattern = '^docs/agents/eval/local/'; Why = "an evaluation case on a person's own data" },
    @{ Pattern = '^modules/Fitting_1D_Model/Workflow_v5/development/'; Why = "development records" },
    @{ Pattern = '\.ipynb$'; Why = "a notebook" },
    @{ Pattern = '(^|/)(__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.gimap_cache|\.ipynb_checkpoints)/'; Why = "a cache" },
    @{ Pattern = '\.py[co]$'; Why = "a cache" },
    @{ Pattern = '(^|/)(gimap_analysis|AI_Fitting_Output|trainset_jobs|test_output|test_results)/'; Why = "analysis output" },
    @{ Pattern = '^(runtime|release)/'; Why = "the folder of the packed environment or of the releases" }
)
# Folders the program is loaded from: a file here that git does not track would be missing.
$ProgramPaths = @("src", "modules", "utils", "assets", "config", "main.py")
# The model manifests the program checks, file by file (SHA-256), before 1D Predict runs.
$ModelManifests = @(
    "modules/Fitting_1D_Model/Workflow_v5/conditional_v2/conditional_fast_manifest_v2.json",
    "modules/Fitting_1D_Model/Workflow_v5/stable_blue_rc_v1/MANIFEST.json"
)

$Problems = [System.Collections.Generic.List[string]]::new()
$Warnings = [System.Collections.Generic.List[string]]::new()

$Tracked = @(Get-GitPaths @("ls-files"))
$IgnoredModelFiles = @(
    Get-GitPaths @("ls-files", "--others", "--ignored", "--exclude-standard", "--", "modules") |
        Where-Object { $_ -match '\.(h5|npz|keras)$' -and $_ -notmatch '(^|/)development/' }
)
$LeftOut = @($Tracked | Where-Object { Test-AnyPattern $_ $NotShipped })
$ApplicationFiles = @(
    @($Tracked) + @($IgnoredModelFiles) |
        Where-Object { -not (Test-AnyPattern $_ $NotShipped) } |
        Sort-Object -Unique
)
if (-not $ApplicationFiles) {
    throw "git lists no file in $ProjectRoot."
}
$Listed = [System.Collections.Generic.HashSet[string]]::new(
    [string[]]$ApplicationFiles, [System.StringComparer]::OrdinalIgnoreCase)

$TotalBytes = [int64]0
$BytesByTop = @{}
$CountByTop = @{}
foreach ($RelativePath in $ApplicationFiles) {
    foreach ($Rule in $Forbidden) {
        if ($RelativePath -match $Rule.Pattern) {
            $Problems.Add("Not allowed in a release ($($Rule.Why)): $RelativePath")
            break
        }
    }
    $FullPath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        if ($FullPath.Length -ge 260) {
            # Windows PowerShell cannot open such a path, so the file cannot be packed either.
            $Problems.Add("The full path has $($FullPath.Length) characters, more than the 259 Windows " +
                "PowerShell can open: build from a folder with a shorter path: $RelativePath")
        }
        else {
            $Problems.Add("Tracked by git but not on disk: $RelativePath")
        }
        continue
    }
    $Top = "(top folder)"
    if ($RelativePath.Contains("/")) { $Top = $RelativePath.Substring(0, $RelativePath.IndexOf("/")) + "/" }
    $Length = (Get-Item -LiteralPath $FullPath -Force).Length
    $TotalBytes += $Length
    $BytesByTop[$Top] = [int64]$BytesByTop[$Top] + $Length
    $CountByTop[$Top] = [int]$CountByTop[$Top] + 1
}

foreach ($Manifest in $ModelManifests) {
    $ManifestPath = Join-Path $ProjectRoot $Manifest
    if (-not $Listed.Contains($Manifest) -or -not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        $Problems.Add("The model manifest is not in the file list: $Manifest")
        continue
    }
    $Folder = $Manifest.Substring(0, $Manifest.LastIndexOf("/"))
    $Checksums = (Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json).files
    foreach ($Entry in $Checksums.PSObject.Properties) {
        $RelativePath = "$Folder/$($Entry.Name)"
        $FullPath = Join-Path $ProjectRoot $RelativePath
        if (-not $Listed.Contains($RelativePath)) {
            $Problems.Add("1D Predict needs this model file, and it is missing: $RelativePath")
        }
        elseif (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
            continue  # reported above: tracked by git but not on disk
        }
        elseif ((Get-FileHash -LiteralPath $FullPath -Algorithm SHA256).Hash -ne [string]$Entry.Value) {
            $Problems.Add("Differs from the checksum in $($Manifest.Substring($Manifest.LastIndexOf('/') + 1)) " +
                "(changed, or its line endings were converted): $RelativePath")
        }
    }
}

$Untracked = @(Get-GitPaths (@("ls-files", "--others", "--exclude-standard", "--") + $ProgramPaths))
if ($Untracked) {
    $Shown = (@($Untracked | Select-Object -First 10) -join ", ")
    if ($Untracked.Count -gt 10) { $Shown += ", ..." }
    $Message = "$($Untracked.Count) file(s) in the program folders are not tracked by git and are NOT packed: $Shown"
    if ($AllowUntracked) {
        $Warnings.Add($Message)
    }
    else {
        $Problems.Add("$Message. Commit them (or remove them) first, or pass -AllowUntracked to build without them.")
    }
}
$Changed = @(Get-GitPaths @("--no-optional-locks", "status", "--porcelain", "--untracked-files=no"))
if ($Changed) {
    $Warnings.Add("$($Changed.Count) tracked file(s) differ from the last commit: the package takes them as they are on disk.")
}
if ($ProgramVersion -and $Version -ne $ProgramVersion) {
    $Warnings.Add("The package is named $Version, but the program reports $ProgramVersion (src/gimap/shared/version.py).")
}

if ($ListOnly) {
    $ApplicationFiles | ForEach-Object { Write-Output $_ }
}
Write-Note ("Application files for $PackageName" + ": $($ApplicationFiles.Count) files, " +
    ("{0:N1} MB" -f ($TotalBytes / 1MB)) + " (from $ProjectRoot)")
Write-Note ("  tracked by git: $($Tracked.Count - $LeftOut.Count); model files git ignores: " +
    "$($IgnoredModelFiles.Count); tracked but left out (working papers): $($LeftOut.Count)")
foreach ($Top in ($CountByTop.Keys | Sort-Object)) {
    Write-Note ("  {0,-16} {1,5} files {2,8:N1} MB" -f $Top, $CountByTop[$Top], ($BytesByTop[$Top] / 1MB))
}
foreach ($ModelFile in $IgnoredModelFiles) {
    Write-Note "  model file: $ModelFile"
}
foreach ($Warning in $Warnings) {
    Write-Note "WARNING: $Warning"
}
foreach ($Problem in $Problems) {
    Write-Note "PROBLEM: $Problem"
}
if ($ListOnly) {
    if ($Problems.Count) {
        Write-Note "$($Problems.Count) problem(s): this list must not be released."
        exit 1
    }
    exit 0
}
if ($Problems.Count) {
    throw "The application file list has $($Problems.Count) problem(s) (listed above). Nothing was packed."
}

Require-Command "conda" "Open an Anaconda/Miniconda PowerShell prompt."
Require-Command "python" "Install conda-pack for the current user with: python -m pip install --user conda-pack"

$CondaPackAvailable = & python -c "import conda_pack" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "conda-pack is not available. Run: python -m pip install --user conda-pack"
}

$EnvironmentJson = & conda env list --json | ConvertFrom-Json
$EnvironmentPath = $EnvironmentJson.envs | Where-Object {
    (Split-Path $_ -Leaf) -eq $EnvironmentName
} | Select-Object -First 1
if (-not $EnvironmentPath) {
    throw "Conda environment '$EnvironmentName' was not found."
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if (Test-Path $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

Write-Host "Packing Conda environment '$EnvironmentName'..."
& python -c "from conda_pack.cli import main; main()" -p $EnvironmentPath -o $ArchivePath --format zip --arcroot "$PackageName/runtime" --n-threads -1 --force --quiet
if ($LASTEXITCODE -ne 0) {
    throw "conda-pack failed with exit code $LASTEXITCODE."
}

Write-Host "Copying $($ApplicationFiles.Count) application files..."
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [System.IO.Compression.ZipFile]::Open($ArchivePath, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    # Only the checked list: the bytes on disk, which are the ones the model manifests were made from.
    foreach ($RelativePath in $ApplicationFiles) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Archive,
            (Join-Path $ProjectRoot $RelativePath),
            "$PackageName/$RelativePath",
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $Archive.Dispose()
}

Write-Host "Portable release created: $ArchivePath"
Write-Host "Test it on a clean Windows computer by extracting the whole ZIP and double-clicking Start_GIMaP.bat."

$ArchiveFile = Get-Item -LiteralPath $ArchivePath
$PartSize = [int64]$ReleasePartSizeMiB * 1MB
if ($ArchiveFile.Length -gt $PartSize) {
    Write-Host "Splitting the ZIP into GitHub Release-compatible parts..."
    Get-ChildItem -LiteralPath $OutputRoot -Filter "$PackageName.zip.part*" -File |
        Remove-Item -Force
    $InputStream = [System.IO.File]::OpenRead($ArchivePath)
    try {
        $Buffer = New-Object byte[] (8MB)
        $PartNumber = 1
        while ($InputStream.Position -lt $InputStream.Length) {
            $PartPath = Join-Path $OutputRoot ("$PackageName.zip.part{0:D2}" -f $PartNumber)
            $OutputStream = [System.IO.File]::Create($PartPath)
            try {
                $Written = [int64]0
                while ($Written -lt $PartSize -and $InputStream.Position -lt $InputStream.Length) {
                    $Requested = [int][Math]::Min($Buffer.Length, $PartSize - $Written)
                    $Read = $InputStream.Read($Buffer, 0, $Requested)
                    if ($Read -le 0) { break }
                    $OutputStream.Write($Buffer, 0, $Read)
                    $Written += $Read
                }
            }
            finally {
                $OutputStream.Dispose()
            }
            $PartNumber++
        }
    }
    finally {
        $InputStream.Dispose()
    }

    $InstallerPath = Join-Path $OutputRoot "Install_GIMaP_parts.bat"
    $InstallerLines = @(
        "@echo off",
        "setlocal",
        "cd /d `"%~dp0`"",
        "set `"PACKAGE=$PackageName`"",
        "set `"ARCHIVE=$PackageName.zip`"",
        "if exist `"%ARCHIVE%`" del /q `"%ARCHIVE%`"",
        "powershell -NoProfile -ExecutionPolicy Bypass -Command `"`$parts=Get-ChildItem -LiteralPath . -Filter '$PackageName.zip.part*' | Sort-Object Name; if(-not `$parts){exit 2}; `$out=[IO.File]::Create('%ARCHIVE%'); try{foreach(`$p in `$parts){`$in=[IO.File]::OpenRead(`$p.FullName); try{`$in.CopyTo(`$out)}finally{`$in.Dispose()}}}finally{`$out.Dispose()}`"",
        "if errorlevel 1 (echo Could not join the release parts. & pause & exit /b 1)",
        "tar -xf `"%ARCHIVE%`"",
        "if errorlevel 1 (echo Could not extract the portable package. & pause & exit /b 1)",
        "del /q `"%ARCHIVE%`"",
        "start `"`" `"%PACKAGE%\Start_GIMaP.bat`"",
        "endlocal"
    )
    [System.IO.File]::WriteAllLines($InstallerPath, $InstallerLines, [System.Text.Encoding]::ASCII)

    $PartFiles = Get-ChildItem -LiteralPath $OutputRoot -Filter "$PackageName.zip.part*" -File | Sort-Object Name
    $OnlineInstallerPath = Join-Path $OutputRoot "00_Download_and_Install_GIMaP-$Version.bat"
    $BaseUrl = "https://github.com/$GitHubRepository/releases/download/$ReleaseTag"
    $OnlineInstallerLines = [System.Collections.Generic.List[string]]::new()
    @(
        "@echo off",
        "setlocal EnableExtensions",
        "cd /d `"%~dp0`"",
        "title GIMaP $Version Installer",
        "set `"BASE_URL=$BaseUrl`"",
        "set `"PACKAGE=$PackageName`"",
        "set `"ARCHIVE=$PackageName.zip`"",
        "echo Downloading GIMaP $Version portable package..."
    ) | ForEach-Object { $OnlineInstallerLines.Add($_) }
    foreach ($PartFile in $PartFiles) {
        $PartHash = (Get-FileHash -LiteralPath $PartFile.FullName -Algorithm SHA256).Hash
        $OnlineInstallerLines.Add("call :download `"$($PartFile.Name)`" `"$PartHash`"")
        $OnlineInstallerLines.Add("if errorlevel 1 goto download_failed")
    }
    @(
        "echo Joining package parts...",
        "if exist `"%ARCHIVE%`" del /q `"%ARCHIVE%`"",
        "powershell -NoProfile -ExecutionPolicy Bypass -Command `"`$parts=Get-ChildItem -LiteralPath . -Filter '%PACKAGE%.zip.part*' | Sort-Object Name; `$out=[IO.File]::Create('%ARCHIVE%'); try{foreach(`$p in `$parts){`$in=[IO.File]::OpenRead(`$p.FullName); try{`$in.CopyTo(`$out)}finally{`$in.Dispose()}}}finally{`$out.Dispose()}`"",
        "if errorlevel 1 goto install_failed",
        "echo Extracting GIMaP...",
        "tar -xf `"%ARCHIVE%`"",
        "if errorlevel 1 goto install_failed",
        "del /q `"%ARCHIVE%`"",
        "del /q `"%PACKAGE%.zip.part*`"",
        "start `"`" `"%PACKAGE%\Start_GIMaP.bat`"",
        "exit /b 0",
        "",
        ":download",
        "set `"FILE=%~1`"",
        "set `"EXPECTED_HASH=%~2`"",
        "if exist `"%FILE%`" call :verify",
        "if exist `"%FILE%`" if not errorlevel 1 exit /b 0",
        "if exist `"%FILE%`" del /q `"%FILE%`"",
        "echo Downloading %FILE%...",
        "where curl.exe >nul 2>&1",
        "if errorlevel 1 goto download_bits",
        "curl.exe -fL --retry 5 --retry-delay 3 --output `"%FILE%`" `"%BASE_URL%/%FILE%`"",
        "if errorlevel 1 exit /b 1",
        "goto verify",
        "",
        ":download_bits",
        "powershell -NoProfile -ExecutionPolicy Bypass -Command `"Import-Module BitsTransfer; Start-BitsTransfer -Source '%BASE_URL%/%FILE%' -Destination (Join-Path (Get-Location) '%FILE%')`"",
        "if errorlevel 1 exit /b 1",
        "",
        ":verify",
        "echo Verifying %FILE%...",
        "powershell -NoProfile -ExecutionPolicy Bypass -Command `"if((Get-FileHash -LiteralPath '%FILE%' -Algorithm SHA256).Hash -ne '%EXPECTED_HASH%'){exit 3}`"",
        "exit /b %errorlevel%",
        "",
        ":download_failed",
        "echo Download failed or the file did not pass verification.",
        "echo Check your network connection, then run this installer again.",
        "pause",
        "exit /b 1",
        "",
        ":install_failed",
        "echo The package could not be joined or extracted.",
        "pause",
        "exit /b 1"
    ) | ForEach-Object { $OnlineInstallerLines.Add($_) }
    [System.IO.File]::WriteAllLines($OnlineInstallerPath, $OnlineInstallerLines, [System.Text.Encoding]::ASCII)
    Write-Host "GitHub assets created: $PackageName.zip.part01 ..., Install_GIMaP_parts.bat, and $($OnlineInstallerPath | Split-Path -Leaf)"
}

# Check the list:  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_portable_release.ps1 -ListOnly
# Build:           powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_portable_release.ps1 -EnvironmentName GUI
