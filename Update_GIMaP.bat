@echo off
setlocal

rem Updates this source folder of GIMaP from GitHub (needs Git) and says what it did.
rem New packages, if the update brings any, are added at the next start of Start_GIMaP.bat.
cd /d "%~dp0"

if not exist "%~dp0tools\update_source.ps1" (
    echo [GIMaP] tools\update_source.ps1 was not found: this is not a source folder of GIMaP.
    echo [GIMaP] The portable version is updated by running the installer of the new release.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\update_source.ps1"
set "UPDATE_CODE=%ERRORLEVEL%"
echo.
if "%UPDATE_CODE%"=="0" (
    echo [GIMaP] Done. Start GIMaP with Start_GIMaP.bat; it adds new packages first if the update needs any.
) else (
    echo [GIMaP] See the messages above.
)
pause
exit /b %UPDATE_CODE%
