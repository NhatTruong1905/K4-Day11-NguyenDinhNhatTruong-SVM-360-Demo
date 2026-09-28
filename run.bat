@echo off
rem Windows Batch script for Day-11 SVM/360 fisheye demo
rem
rem Examples:
rem   run.bat
rem   run.bat --recreate
rem   run.bat --only object

setlocal enabledelayedexpansion

cd /d "%~dp0"

if "%PY%"=="" set PY=python

echo Checking Python dependencies...
"%PY%" -c "import cvat_sdk, numpy, PIL, requests, dotenv; print('ok')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Missing python dependencies. Please run: pip install -r requirements.txt
    exit /b 1
)

set ONLY_ARG=
set RECREATE_ARG=

:parse_args
if "%~1"=="" goto after_args
if /i "%~1"=="--recreate" (
    set RECREATE_ARG=--recreate
    shift
    goto parse_args
)
if /i "%~1"=="--only" (
    set ONLY_ARG=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:after_args

rem Task A: FishEye8K
if "%ONLY_ARG%"=="" goto do_fe
if /i "%ONLY_ARG%"=="object" goto do_fe
goto skip_fe

:do_fe
echo.
echo ^>^> Task A: FishEye8K object subset + golden GT (HuggingFace, no token) ...
"%PY%" scripts/prepare_fisheye8k.py
if errorlevel 1 exit /b %errorlevel%
:skip_fe

rem Tasks B/C/D: WoodScape
if "%ONLY_ARG%"=="" goto do_ws
if /i "%ONLY_ARG%"=="freespace" goto do_ws
if /i "%ONLY_ARG%"=="lines" goto do_ws
if /i "%ONLY_ARG%"=="ignore" goto do_ws
goto skip_ws

:do_ws
echo.
echo ^>^> Tasks B/C/D: WoodScape subset + golden GT (Kaggle, needs KAGGLE_API_TOKEN) ...
"%PY%" scripts/prepare_woodscape.py
if errorlevel 1 exit /b %errorlevel%
:skip_ws

echo.
echo ^>^> Creating CVAT tasks (split by camera_id) + a golden GT job for each ...
if "%ONLY_ARG%"=="" (
    "%PY%" scripts/setup_cvat.py %RECREATE_ARG%
) else (
    "%PY%" scripts/setup_cvat.py %RECREATE_ARG% --only %ONLY_ARG%
)
if errorlevel 1 exit /b %errorlevel%

echo.
echo ^>^> ALL DONE. Tasks created successfully!
