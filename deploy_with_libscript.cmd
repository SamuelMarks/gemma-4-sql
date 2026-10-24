@echo off
setlocal EnableDelayedExpansion

REM deploy_with_libscript.cmd
REM Installs dependencies and orchestrates gemma-4-sql local, GPU, or TPU training/inference deployment via libscript.

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if exist "%SCRIPT_DIR%\..\libscript\libscript.cmd" (
    for %%I in ("%SCRIPT_DIR%\..\libscript") do set "LIBSCRIPT_ROOT=%%~fI"
    set "LIBSCRIPT_CMD=!LIBSCRIPT_ROOT!\libscript.cmd"
) else (
    if defined LIBSCRIPT_ROOT_DIR (
        set "LIBSCRIPT_ROOT=%LIBSCRIPT_ROOT_DIR%"
    ) else (
        set "LIBSCRIPT_ROOT=%USERPROFILE%\.libscript"
    )
    set "LIBSCRIPT_CMD=libscript"
)

set "MODE=local"
set "ACTION=deploy"
set "DRY_RUN_FLAG="

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help
if /i "%~1"=="--local" set "MODE=local"
if /i "%~1"=="--tpu-vm" set "MODE=gcp-tpu-vm"
if /i "%~1"=="--gcp-tpu-vm" set "MODE=gcp-tpu-vm"
if /i "%~1"=="--gpu-vm" set "MODE=gcp-gpu-vm"
if /i "%~1"=="--gcp-gpu-vm" set "MODE=gcp-gpu-vm"
if /i "%~1"=="--gke-xpk" set "MODE=gke-xpk"
if /i "%~1"=="--train" set "ACTION=train"
if /i "%~1"=="--infer" set "ACTION=infer"
if /i "%~1"=="--serve" set "ACTION=infer"
if /i "%~1"=="--serve-ml" set "ACTION=infer"
if /i "%~1"=="--dry-run" set "DRY_RUN_FLAG=--dry-run"
shift
goto :parse_args

:args_done
set "APP_DIR=%SCRIPT_DIR%"

if not "!DRY_RUN_FLAG!"=="" (
    set "ML_CMD=!ACTION!"
    if /i "!ML_CMD!"=="deploy" set "ML_CMD=train"
    call "!LIBSCRIPT_CMD!" !ML_CMD! "%SCRIPT_DIR%\libscript.json" --target "!MODE!" --dry-run
    exit /b !errorlevel!
)

echo Installing required dependencies via libscript...
call "!LIBSCRIPT_CMD!" install python 3.12
call "!LIBSCRIPT_CMD!" install duckdb latest
call "!LIBSCRIPT_CMD!" install sqlite latest
call "!LIBSCRIPT_CMD!" install huggingface-cli latest
call "!LIBSCRIPT_CMD!" install xpk latest
if exist "%SCRIPT_DIR%\libscript.json" (
    call "!LIBSCRIPT_CMD!" install-deps "%SCRIPT_DIR%\libscript.json"
)

if /i "!ACTION!"=="train" (
    call "!LIBSCRIPT_CMD!" train "%SCRIPT_DIR%\libscript.json" --target "!MODE!"
    exit /b !errorlevel!
)
if /i "!ACTION!"=="infer" (
    call "!LIBSCRIPT_CMD!" infer "%SCRIPT_DIR%\libscript.json" --target "!MODE!"
    exit /b !errorlevel!
)

if /i "!MODE!"=="gcp-tpu-vm" (
    call "!LIBSCRIPT_ROOT!\payloads\ai\gcp-tpu-vm\setup.cmd"
    call "!LIBSCRIPT_ROOT!\payloads\ai\gcp-tpu-vm\deploy.cmd"
) else if /i "!MODE!"=="gcp-gpu-vm" (
    call "!LIBSCRIPT_ROOT!\payloads\ai\gcp-gpu-vm\setup.cmd"
    call "!LIBSCRIPT_ROOT!\payloads\ai\gcp-gpu-vm\deploy.cmd"
) else if /i "!MODE!"=="gke-xpk" (
    call "!LIBSCRIPT_ROOT!\payloads\ai\gke-xpk-training\setup.cmd"
    call "!LIBSCRIPT_ROOT!\payloads\ai\gke-xpk-training\deploy.cmd"
) else (
    echo Installing gemma-4-sql in editable mode...
    call pip install -e "%SCRIPT_DIR%"
)
exit /b %errorlevel%

:help
echo Usage: deploy_with_libscript.cmd [--local ^| --tpu-vm ^| --gpu-vm ^| --gke-xpk] [--train ^| --infer ^| --serve] [--dry-run] [--help]
exit /b 0
