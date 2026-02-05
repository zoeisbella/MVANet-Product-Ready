#!/bin/bash
# 生产环境启动脚本

set -e  # 遇到错误立即退出

echo "Starting MVANet API in production mode..."

# 检查模型文件是否存在
if [ ! -f "./weights/mvanet.pth" ]; then
    echo "Error: Model weights file not found at ./weights/mvanet.pth"
    exit 1
fi

# 设置生产环境变量
export PYTHONPATH="${PYTHONPATH}:."
export LOGURU_LEVEL="INFO"

# 启动服务
exec uvicorn app_optimized:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8000} \
    --workers ${WORKERS:-4} \
    --timeout-keep-alive 30 \
    --log-level ${LOG_LEVEL:-info} \
    --access-log