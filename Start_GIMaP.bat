@echo off
setlocal

rem GIMaP portable launcher. Keep this file beside main.py and runtime\.
cd /d "%~dp0"

rem Qt (the window library) cannot load its plugins from a folder whose path has non-English letters
rem (for example a Chinese "Downloads" folder): it stops with "no Qt platform plugin could be initialized".
rem The path goes to PowerShell through a variable, so this file stays ASCII. GIMAP_ALLOW_NON_ASCII=1 skips it.
set "GIMAP_ROOT=%~dp0"
if not "%GIMAP_ALLOW_NON_ASCII%"=="1" (
    powershell -NoProfile -Command "if ($env:GIMAP_ROOT -match '[^\u0000-\u007F]') { exit 3 } else { exit 0 }"
    if errorlevel 3 (
        echo [GIMaP] This folder path contains non-English characters:
        echo %GIMAP_ROOT%
        echo.
        echo Qt cannot start from such a folder. Move the whole GIMaP folder to a short English path,
        echo for example D:\GIMaP or C:\GIMaP. For a source folder, also delete its .gimap folder there,
        echo then start Start_GIMaP.bat again: the environment is set up again for the new place.
        echo.
        pause
        exit /b 1
    )
)

set "PYTHON_EXE=%~dp0runtime\pythonw.exe"
set "RUNTIME_PYTHON=%~dp0runtime\python.exe"
set "CONDA_UNPACK_SCRIPT=%~dp0runtime\Scripts\conda-unpack-script.py"
set "FIRST_RUN_MARKER=%~dp0runtime\.gimap_unpacked"

if not exist "%PYTHON_EXE%" if exist "%~dp0main.py" if exist "%~dp0tools\setup_source_env.ps1" (
    rem A source folder (git clone or ZIP of the repository): set up its own environment once, then start.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup_source_env.ps1"
    if errorlevel 1 (
        echo.
        pause
        exit /b 1
    )
    exit /b 0
)

if not exist "%PYTHON_EXE%" (
    echo [GIMaP] Portable Python was not found:
    echo %PYTHON_EXE%
    echo.
    echo Please download the complete portable release and extract all files.
    pause
    exit /b 1
)

rem First start: this folder is written into the packed environment (Qt finds its plugins through it).
rem tools\portable_unpack.py does it in little memory: conda-unpack read every listed file whole, one of
rem them 698 MB, and on computers with little free memory the first start stopped with "Not enough memory
rem resources are available to process this command". Both run with the packed python.exe, not through
rem conda-unpack.exe, whose failure to start set no error code (the folder was then marked as prepared
rem although it was not). The marker is written only after the step succeeded.
set "PORTABLE_UNPACK=%~dp0tools\portable_unpack.py"
if not exist "%FIRST_RUN_MARKER%" (
    if exist "%CONDA_UNPACK_SCRIPT%" (
        echo [GIMaP] Preparing the portable environment for this folder, this takes a few minutes...
        if exist "%PORTABLE_UNPACK%" (
            "%RUNTIME_PYTHON%" "%PORTABLE_UNPACK%" && type nul > "%FIRST_RUN_MARKER%"
        ) else (
            "%RUNTIME_PYTHON%" "%CONDA_UNPACK_SCRIPT%" && type nul > "%FIRST_RUN_MARKER%"
        )
    ) else (
        type nul > "%FIRST_RUN_MARKER%"
    )
)
if not exist "%FIRST_RUN_MARKER%" (
    echo.
    echo [GIMaP] The portable environment could not be prepared, so GIMaP was not started.
    echo   1. Restart the computer and close other programs, then run Start_GIMaP.bat again.
    echo   2. Check that virtual memory is on: System - About - Advanced system settings -
    echo      Performance Settings - Advanced - Virtual memory: "Automatically manage paging file size".
    echo   3. Keep the folder on a short English path such as D:\GIMaP, with a few GB free on that drive.
    echo   4. If it still fails, extract the release ZIP again into an empty folder.
    pause
    exit /b 1
)

set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%~dp0"
set "PATH=%~dp0runtime;%~dp0runtime\Library\bin;%~dp0runtime\Scripts;%PATH%"
set "MPLCONFIGDIR=%LOCALAPPDATA%\GIMaP\matplotlib"
if not exist "%LOCALAPPDATA%\GIMaP\matplotlib" mkdir "%LOCALAPPDATA%\GIMaP\matplotlib" >nul 2>&1

start "GIMaP" /D "%~dp0" "%PYTHON_EXE%" "%~dp0main.py"
if errorlevel 1 (
    echo [GIMaP] Startup failed.
    pause
    exit /b 1
)

endlocal
