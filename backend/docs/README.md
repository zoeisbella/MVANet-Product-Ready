# MVANet Image Segmentation API

High-performance image segmentation API using MVANet, designed for production environments with robust error handling and scalability.

## 📋 Table of Contents
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Performance](#performance)
- [Contributing](#contributing)

## 🏗️ Architecture

```
backend/
├── src/                    # Source code
│   ├── api/                # API endpoints and routing
│   ├── models/             # Deep learning models
│   ├── services/           # Business logic
│   ├── utils/              # Utilities and helpers
│   ├── common/             # Shared components
│   └── core/               # Core functionality
├── tests/                  # Test suite
├── docs/                   # Documentation
├── config/                 # Configuration files
├── scripts/                # Scripts for deployment and maintenance
├── assets/                 # Static assets and model weights
├── requirements.txt        # Dependencies
├── pyproject.toml          # Project metadata
└── Dockerfile              # Container configuration
```

## ✅ Prerequisites

- Python >= 3.8
- CUDA-compatible GPU (optional, for GPU acceleration)
- At least 4GB RAM
- 2GB free disk space for model weights

## 🚀 Installation

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

### Manual Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
pre-commit install
```

## ⚙️ Configuration

Create `.env` file in the config directory:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
DEBUG=false

# Model Configuration
MODEL_PATH=./assets/weights/mvanet.pth
DEVICE=auto  # auto, cpu, or cuda

# Performance Configuration
BATCH_SIZE=1
MAX_IMAGE_SIZE=1024
MEMORY_LIMIT_GB=4
```

## 📡 Usage

### Running the Server
```bash
# Development mode
make start

# Production mode
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker
```bash
# Build and run
make docker-build
make docker-run

# Or with docker-compose
docker-compose up --build
```

### API Example
```python
import requests
import base64

# Load image
with open('path/to/image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    'http://localhost:8000/predict',
    json={'image_base64': image_data}
)

result = response.json()
print(result['mask_base64'])
```

## 📚 API Documentation

The API provides the following endpoints:

- `GET /` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)
- `POST /predict` - Perform segmentation on an image
- `POST /predict_and_download` - Perform segmentation and return downloadable image
- `GET /health` - Health check with detailed status

### Request Format
```json
{
  "image_base64": "base64_encoded_image_string",
  "threshold": 0.5,
  "return_format": "base64"  // "base64" or "download"
}
```

### Response Format
```json
{
  "success": true,
  "mask_base64": "segmentation_mask_in_base64",
  "message": "Segmentation completed successfully",
  "inference_time": 0.234,
  "original_size": [width, height]
}
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest --cov=src --cov-report=html
```

### Linting and Formatting
```bash
# Check code style
make lint

# Auto-format code
make format

# Type checking
make check
```

### Performance Testing
```bash
# Run stress tests
make benchmark
```

## 🚢 Deployment

### Production Deployment
```bash
# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Using systemd (Linux)
sudo cp deployment/mvanet.service /etc/systemd/system/
sudo systemctl enable mvanet-api
sudo systemctl start mvanet-api
```

### Kubernetes Deployment
Refer to `deployment/k8s/` for Kubernetes manifests.

## ⚡ Performance

- Average inference time: ~200ms per 512×512 image
- Memory usage: ~2GB RAM, ~1GB GPU memory
- Throughput: ~5 requests/second (depending on image size)
- Supports batch processing for improved throughput

## 🔐 Security

- Input validation for all endpoints
- Rate limiting to prevent abuse
- Secure file upload handling
- Environment-based configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Write tests for new features
- Update documentation as needed
- Use type hints for all public APIs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support, please contact the development team or open an issue on GitHub.

---

**Built with ❤️ by the Development Team**