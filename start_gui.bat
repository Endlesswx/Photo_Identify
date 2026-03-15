@echo off
chcp 65001 >nul 2>&1
setlocal

:: 优先使用系统 uv，否则使用内置 uv
where uv >nul 2>&1
if errorlevel 1 (
    if exist "%~dp0uv\uv.exe" (
        set "UV=%~dp0uv\uv.exe"
    ) else (
        echo [错误] 未检测到 uv，请先运行 install.bat
        pause
        exit /b 1
    )
) else (
    set "UV=uv"
)

"%UV%" run python -m photo_identify
