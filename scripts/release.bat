@echo off
setlocal enabledelayedexpansion

REM Build source and wheel distributions for AgencityLab on Windows.
REM Expected tools:
REM - python
REM - build

set ROOT_DIR=%~dp0..\..
cd /d "%ROOT_DIR%"

python -m pip install --upgrade build
if errorlevel 1 exit /b 1

python -m build
exit /b %errorlevel%
