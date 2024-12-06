@echo off
REM 设置错误时退出
setlocal enabledelayedexpansion
set ERRORLEVEL=0

REM 检查Podman是否可用
podman --version >nul 2>&1
if ERRORLEVEL 1 (
    echo Podman未安装或未配置正确，请先安装Podman。
    exit /b 1
)

REM 运行Podman容器
podman run -p 7860:7860 --device nvidia.com/gpu=all --net=host localhost/zhy_defect_detection
if ERRORLEVEL 1 (
    echo 容器运行失败。
    exit /b 1
)

echo 容器已成功启动。
pause

