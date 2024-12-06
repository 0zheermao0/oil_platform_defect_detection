#!/bin/bash

# 检查 Podman 是否已安装
if ! command -v podman &> /dev/null; then
    echo "Podman 未安装，请先安装 Podman。"
    exit 1
fi

# 执行 Podman 容器
echo "启动容器中..."
podman run -p 7860:7860 --device nvidia.com/gpu=all --net=host localhost/zhy_defect_detection

# 检查是否成功
if [ $? -ne 0 ]; then
    echo "容器启动失败，请检查命令和环境。"
    exit 1
else
    echo "容器已成功启动。"
fi

