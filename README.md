# MVANet-Product-Ready

A production-ready image segmentation project based on the CVPR 2024 paper **"Multi-view Aggregation Network for Dichotomous Image Segmentation"**, refactoring academic research code into an engineering solution ready for deployment.

## Paper Information

- **Title**: Multi-view Aggregation Network for Dichotomous Image Segmentation
- **Conference**: CVPR 2024 (IEEE/CVF Conference on Computer Vision and Pattern Recognition)
- **Authors**: Qian Yu, Xiaoqi Zhao, Youwei Pang, Lihe Zhang*, Huchuan Lu
- **Affiliation**: Dalian University of Technology
- **GithubLink**: https://github.com/qianyu-dlut/MVANet

## Core Contributions

### Architecture Refactoring
- Refactored original script-based academic code into modular engineering architecture
- Introduced FastAPI asynchronous framework for high-performance backend API service
- Developed frontend web interface based on Node.js + Express
- Implemented modern frontend-backend separation architecture, completing the full pipeline from model loading to inference delivery

### Engineering Enhancements
- Integrated Loguru logging system for full-link monitoring, addressing the original project's lack of deployment support
- Implemented CORS cross-origin support to resolve cross-domain issues in frontend-backend integration
- Designed comprehensive error handling mechanisms to improve system stability
- Written structured project documentation to lower the barrier to entry

### Frontend & Backend Development
- **Frontend Interface**: Developed user-friendly web interface supporting image upload preview, real-time segmentation display, and result download
- **Backend API**: Implemented multiple interfaces including `/predict` (file upload) and `/predict_json` (Base64 data) to meet different scenario requirements
- **API Documentation**: Integrated Swagger UI, providing interactive API documentation and online debugging capabilities
<img width="1912" height="892" alt="978f0a762411c13c8d909f3bc06c10f" src="https://github.com/user-attachments/assets/ce48b086-13c8-4d61-a815-74277f0f6212" />
<img width="1920" height="892" alt="ce2f477698e4a4ae9ed2376c94da156" src="https://github.com/user-attachments/assets/2037705b-51b4-4e19-b0e9-3d34b4b2f8e6" />

### Bug Fixes & Optimization
- Resolved tensor size mismatch issues in model inference by fixing input size to 512×512 to ensure model stability
- Optimized image preprocessing and postprocessing pipelines to improve segmentation quality and user experience
- Implemented responsive interface design to adapt to different screen sizes

### Project Deployment
- Completed version control and GitHub open-source deployment of project code
- Provided complete installation and deployment guides and usage instructions
- Supported one-click local startup to reduce deployment complexity

## Tech Stack

- **Backend**: Python + FastAPI + PyTorch + Loguru
- **Frontend**: Node.js + Express + HTML5 + JavaScript
- **Model**: MVANet (Multi-view Aggregation Network based on Swin Transformer for Dichotomous Image Segmentation)

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the project
git clone https://github.com/zoeisbella/MVANet-Product-Ready.git
cd MVANet-Product-Ready

# Setup backend
cd backend
pip install -r requirements.txt

# Setup frontend
cd ../frontend
npm install
```

### Running the Application

```bash
# Terminal 1: Start backend
cd backend
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001

# Terminal 2: Start frontend
cd frontend
node server.js
```

### Access Services

- **Frontend Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8001/docs
- **Backend API**: http://localhost:8001

## API Endpoints

### POST /predict
Upload an image file for segmentation.

**Request**: `multipart/form-data` with image file

**Response**: JSON with base64-encoded segmentation mask

### POST /predict_json
Send base64-encoded image for segmentation.

**Request**: JSON with `image_base64` field

**Response**: JSON with base64-encoded segmentation mask

### POST /predict_and_download
Upload an image and download the segmentation result directly.

**Request**: `multipart/form-data` with image file

**Response**: PNG image file

## Project Structure

```
MVANet-Product-Ready/
├── backend/                 # Backend API service
│   ├── src/
│   │   ├── api/            # FastAPI application
│   │   │   ├── app.py      # Main API routes
│   │   │   └── models.py   # Pydantic models
│   │   └── model/          # Model loading and inference
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Backend documentation
├── frontend/               # Frontend web interface
│   ├── index.html          # Main web page
│   ├── server.js           # Express server
│   ├── package.json        # Node.js dependencies
│   └── README.md           # Frontend documentation
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Features

- 🚀 **High Performance**: Asynchronous API with FastAPI
- 🎨 **User Friendly**: Modern web interface with drag-and-drop upload
- 📊 **Real-time Processing**: Live segmentation with progress indication
- 🔧 **Easy Deployment**: Docker support and one-click startup
- 📚 **Well Documented**: Comprehensive API docs with Swagger UI
- 🌐 **Cross Origin**: CORS enabled for flexible integration

## Model Details

MVANet (Multi-view Aggregation Network) is a state-of-the-art model for dichotomous image segmentation that:
- Uses Swin Transformer as the backbone
- Employs multi-view feature aggregation
- Achieves superior performance on segmentation benchmarks

## Citation

If you use this project in your research, please cite the original paper:

```bibtex
@inproceedings{yu2024mvanet,
  title={Multi-view Aggregation Network for Dichotomous Image Segmentation},
  author={Yu, Qian and Zhao, Xiaoqi and Pang, Youwei and Zhang, Lihe and Lu, Huchuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original MVANet implementation by the authors
- FastAPI framework for high-performance APIs
- Swin Transformer for the backbone architecture

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a community refactoring project for educational and production use. The original model and paper belong to the respective authors.
