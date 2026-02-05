# MVANet - Medical Visual Analysis Network

This repository contains a production-ready implementation of MVANet (Medical Visual Analysis Network) for image segmentation, featuring both backend API and frontend interface.

## 🏗️ Project Structure

```
MVANet/
├── backend/                # Backend API implementation
│   ├── src/                # Source code
│   │   ├── api/            # API endpoints
│   │   ├── models/         # Deep learning models
│   │   ├── services/       # Business logic
│   │   ├── utils/          # Utilities
│   │   ├── common/         # Shared components
│   │   └── core/           # Core functionality
│   ├── tests/              # Test suite
│   ├── docs/               # Documentation
│   ├── config/             # Configuration files
│   ├── scripts/            # Utility scripts
│   ├── assets/             # Assets and model weights
│   ├── requirements.txt    # Python dependencies
│   ├── pyproject.toml      # Project metadata
│   ├── Dockerfile          # Container configuration
│   ├── Makefile            # Build commands
│   ├── main.py             # Application entry point
│   ├── start_system.sh     # Linux/Mac startup script
│   ├── start_system.ps1    # Windows startup script
│   └── README.md           # Backend documentation
└── frontend/               # Frontend interface
    ├── index.html          # Main interface
    ├── server.js           # Frontend server
    ├── package.json        # Node.js dependencies
    └── README.md           # Frontend documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- CUDA-compatible GPU (optional)

### Installation

1. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
```

2. **Frontend Setup**
```bash
cd frontend
npm install
```

### Running the System

#### Option 1: Manual Start
1. Start backend: `cd backend && uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
2. Start frontend: `cd frontend && npm start`

#### Option 2: Using Startup Script (Windows)
```bash
powershell -ExecutionPolicy Bypass -File "d:\_Cursor\_MVANet\backend\start_system.ps1"
```

Access the system:
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:3000
- API Documentation: http://localhost:8000/docs

## 📋 Features

- **Production-ready API**: FastAPI-based backend with async support
- **Robust Error Handling**: Comprehensive error handling and logging
- **Memory Management**: Efficient GPU memory usage
- **Scalable Architecture**: Designed for high-throughput applications
- **User-friendly Interface**: Web-based UI for easy interaction
- **Comprehensive Testing**: Unit tests and stress tests included
- **Docker Support**: Containerized deployment ready
- **Configuration Management**: Environment-based configuration

## 🧪 Testing

Run backend tests:
```bash
cd backend
make test
```

Run stress tests:
```bash
cd backend
make benchmark
```

## 🚢 Deployment

The system is ready for production deployment with:
- Docker and Docker Compose configurations
- Kubernetes manifest templates
- Process managers (PM2, systemd)
- Load balancing configurations

## 🤝 Contributing

See individual README files in `backend/` and `frontend/` directories for specific contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.