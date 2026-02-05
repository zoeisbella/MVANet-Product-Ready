# MVANet Frontend Interface

Web-based user interface for the MVANet Image Segmentation API, allowing users to upload images and view segmentation results.

## 📋 Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Contributing](#contributing)

## ✨ Features

- **User-friendly Interface**: Simple drag-and-drop or click-to-upload functionality
- **Real-time Preview**: Instant preview of uploaded images
- **Result Visualization**: Clear display of segmentation results
- **Download Capability**: Easy download of processed images
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Clear feedback for invalid inputs or processing errors

## 🏗️ Architecture

```
frontend/
├── index.html              # Main HTML interface
├── server.js               # Express server for serving frontend
├── package.json            # Node.js dependencies and scripts
└── README.md               # This documentation
```

## ✅ Prerequisites

- Node.js >= 14.x
- npm (usually comes with Node.js)

## 🚀 Installation

### Quick Start
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the frontend server
npm start
```

## 📡 Usage

### Running the Frontend Server
```bash
# Development mode
npm start

# Or with nodemon for auto-restart on changes
npm run dev
```

### Accessing the Interface
1. Make sure the backend API server is running on `http://localhost:8000`
2. Start the frontend server with `npm start`
3. Open your browser and navigate to `http://localhost:3000`
4. Upload an image and click "开始分割" to process it

### Using with Docker
```bash
# If using the main docker-compose setup
docker-compose up --build
```

## 🔌 API Integration

The frontend communicates with the backend API through the following endpoints:

- `POST /predict`: Sends image for segmentation and receives base64-encoded result
- `POST /predict_and_download`: Alternative endpoint for downloadable results
- `GET /health`: Checks backend API health status

The frontend automatically handles:
- Base64 encoding/decoding of images
- Loading states during processing
- Error handling and user feedback
- Result display and download functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow standard JavaScript/HTML/CSS best practices
- Maintain responsive design principles
- Ensure accessibility standards are met
- Keep dependencies up to date

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📞 Support

For support, please contact the development team or open an issue on GitHub.

---

**Built with ❤️ by the Development Team**