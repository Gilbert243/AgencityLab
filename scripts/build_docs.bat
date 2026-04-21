@echo off
setlocal enabledelayedexpansion

REM Build AgencityLab documentation on Windows.
REM Expected tools:
REM - python
REM - sphinx-build (installed through the docs extra)

set ROOT_DIR=%~dp0..\..
cd /d "%ROOT_DIR%"

where sphinx-build >nul 2>nul
if errorlevel 1 (
    echo sphinx-build is not available. Install the docs extras first.
    exit /b 1
)

if exist docs\_build rmdir /s /q docs\_build
sphinx-build -b html docs docs\_build\html
echo Documentation generated in docs\_build\html
