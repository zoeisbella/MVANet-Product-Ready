import subprocess
import sys
import os

def install_requirements():
    """安装依赖包"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def start_server():
    """启动FastAPI服务器"""
    print("Starting MVANet API server...")
    subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    # 切换到当前脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 安装依赖（可选）
    if "--install" in sys.argv:
        install_requirements()
    
    # 启动服务器
    start_server()