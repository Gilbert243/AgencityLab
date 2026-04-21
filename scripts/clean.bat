@echo off
setlocal enabledelayedexpansion

REM Remove common build and cache directories on Windows.

set ROOT_DIR=%~dp0..\..
cd /d "%ROOT_DIR%"

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist docs\_build rmdir /s /q docs\_build
if exist .ruff_cache rmdir /s /q .ruff_cache

for /d /r %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d"
)

echo Cleanup complete.
