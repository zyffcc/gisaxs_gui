[CmdletBinding()]
param(
    [string]$EnvironmentName = "GUI",
    [string]$Version = "dev",
    [string]$OutputDirectory = "release",
    [int]$ReleasePartSizeMiB = 1900,
    [string]$GitHubRepository = "zyffcc/gisaxs_gui",
    [string]$ReleaseTag = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputDirectory))
$PackageName = "GIMaP-$Version-windows-x64-portable"
$ArchivePath = Join-Path $OutputRoot "$PackageName.zip"
if (-not $ReleaseTag) {
    $ReleaseTag = "v$Version"
}

function Require-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. $InstallHint"
    }
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

Write-Host "Copying application files..."
$ExcludedDirectoryNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
@(
    ".git", ".agents", ".codex", ".idea", ".vscode", ".gimap_cache",
    "release", "runtime", "__pycache__", ".pytest_cache", "AI_Fitting_Output",
    "Experiment_data", "Experiment_SAXS_data", "test_output", "test_results"
) | ForEach-Object { [void]$ExcludedDirectoryNames.Add($_) }
$ExcludedExtensions = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
@(".pyc", ".pyo", ".log", ".zip") | ForEach-Object { [void]$ExcludedExtensions.Add($_) }

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [System.IO.Compression.ZipFile]::Open($ArchivePath, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    $Files = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Force | Where-Object {
        $RelativePath = $_.FullName.Substring($ProjectRoot.Length).TrimStart("\")
        $Segments = $RelativePath -split "[\\/]"
        -not ($Segments | Where-Object { $ExcludedDirectoryNames.Contains($_) }) -and
        -not $ExcludedExtensions.Contains($_.Extension)
    }
    foreach ($File in $Files) {
        $RelativePath = $File.FullName.Substring($ProjectRoot.Length).TrimStart("\").Replace("\", "/")
        $EntryName = "$PackageName/$RelativePath"
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Archive,
            $File.FullName,
            $EntryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $Archive.Dispose()
}

Write-Host "Portable release created: $ArchivePath"
Write-Host "Test it on a clean Windows computer by extracting the whole ZIP and double-clicking 启动_GIMaP.bat."

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

# powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_portable_release.ps1 -EnvironmentName GUI -Version 0.1.1
