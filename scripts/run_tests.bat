@echo off
setlocal enabledelayedexpansion

REM Run the AgencityLab test suite on Windows.
REM Expected tools:
REM - python
REM - pytest

set ROOT_DIR=%~dp0..\..
cd /d "%ROOT_DIR%"

python -m pytest tests
exit /b %errorlevel%
