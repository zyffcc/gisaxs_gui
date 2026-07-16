@echo off
setlocal

rem GIMaP portable launcher. Keep this file beside main.py and runtime\.
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0runtime\pythonw.exe"
set "CONDA_UNPACK=%~dp0runtime\Scripts\conda-unpack.exe"
set "FIRST_RUN_MARKER=%~dp0runtime\.gimap_unpacked"

if not exist "%PYTHON_EXE%" (
    echo [GIMaP] Portable Python was not found:
    echo %PYTHON_EXE%
    echo.
    echo Please download the complete portable release and extract all files.
    pause
    exit /b 1
)

if not exist "%FIRST_RUN_MARKER%" (
    if exist "%CONDA_UNPACK%" (
        echo [GIMaP] Preparing the portable environment for this folder...
        "%CONDA_UNPACK%"
        if errorlevel 1 (
            echo [GIMaP] The portable environment could not be prepared.
            echo Try extracting the package to a short writable path such as C:\GIMaP.
            pause
            exit /b 1
        )
    )
    type nul > "%FIRST_RUN_MARKER%"
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
