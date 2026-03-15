@echo off
chcp 65001 >nul 2>&1
setlocal

:: 优先使用系统 uv，否则使用内置 uv
where uv >nul 2>&1
if errorlevel 1 (
    if exist "%~dp0uv\uv.exe" (
        set "UV=%~dp0uv\uv.exe"
        echo 使用内置 uv
    ) else (
        echo [错误] 未检测到 uv，且内置 uv 缺失。
        echo 请安装 uv：https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
) else (
    set "UV=uv"
)

echo 正在安装依赖...
"%UV%" sync
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo 安装完成！双击 start_gui.bat 启动程序。
pause
