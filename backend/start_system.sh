#!/bin/bash
# 启动完整的MVANet系统

echo "Starting MVANet Image Segmentation System..."

# 启动后端服务
echo "Starting backend API server..."
cd backend
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 等待后端启动
sleep 5

# 启动前端服务
echo "Starting frontend server..."
cd ../frontend
npm start &
FRONTEND_PID=$!

echo "Services started!"
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:3000"
echo "Press Ctrl+C to stop"

# 等待进程结束
wait $BACKEND_PID $FRONTEND_PID