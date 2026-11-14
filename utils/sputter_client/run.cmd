@echo off
rem iampred client launcher: run.cmd <command> [arguments]   (commands: analyze, static, intake, predict, report, check)
set "PKG=%~dp0"
set "IAM_GIMAP=%PKG%code\gimap"
set "IAM_DIST=%PKG%code\toolkit"
if "%IAM_DATA%"=="" set "IAM_DATA=%PKG%data"
set "IAM_WORK=%PKG%work"
set "IAM_RAW=%PKG%no_raw"
set "IAMPRED_MODELS=%PKG%models"
set "PYTHONPATH=%PKG%code\iam;%PYTHONPATH%"
set "TZ=Europe/Berlin"
set "OMP_NUM_THREADS=1"
set "IAMPRED_PY=python"
if exist "%PKG%python.txt" set /p IAMPRED_PY=<"%PKG%python.txt"
"%IAMPRED_PY%" -m iampred.%*
