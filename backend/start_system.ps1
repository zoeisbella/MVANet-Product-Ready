# 启动完整的MVANet系统

Write-Host "Starting MVANet Image Segmentation System..." -ForegroundColor Green

# 启动后端服务
Write-Host "Starting backend API server..." -ForegroundColor Yellow
Set-Location "d:\_Cursor\_MVANet\backend"
Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"

# 等待后端启动
Start-Sleep -Seconds 5

# 启动前端服务
Write-Host "Starting frontend server..." -ForegroundColor Yellow
Set-Location "d:\_Cursor\_MVANet\frontend"
Start-Process -FilePath "npm" -ArgumentList "start"

Write-Host "Services started!" -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend UI: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Please make sure you have Node.js and Python environments ready" -ForegroundColor Magenta