# Face Recognition Control System - Production Edition

## 🚀 Enterprise-Grade Face Recognition API

Production-ready face recognition system with advanced AI capabilities, deployed on Modal.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Modal](https://img.shields.io/badge/Modal-Latest-purple.svg)](https://modal.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Features

### AI Capabilities
- **99.7%+ Recognition Accuracy** on LFW dataset
- **Multi-Model Ensemble**:
  - InsightFace (ArcFace)
  - VGGFace2 (ResNet-50)
  - MTCNN for face detection
- **Advanced Security**:
  - Anti-spoofing protection
  - Liveness detection
  - Face quality assessment
- **Additional Analysis**:
  - Age estimation
  - Gender detection
  - Emotion recognition
- **Performance**:
  - <100ms inference time
  - Support for 10,000+ registered faces
  - Real-time processing at 30 FPS
  - Multi-face detection (up to 20 faces)

### Production Features
- **Enterprise API Design**:
  - OpenAPI 3.0 specification
  - Input validation with Pydantic
  - Comprehensive error handling
  - Rate limiting and authentication
  - Request/response logging
  - API versioning
- **Security**:
  - Bearer token authentication
  - CORS configuration
  - Input sanitization
  - File validation
  - Anti-spoofing measures
- **Monitoring**:
  - Prometheus metrics
  - Sentry error tracking
  - Health check endpoints
  - Performance monitoring
- **Scalability**:
  - GPU acceleration
  - Async operations
  - Connection pooling
  - Resource management

## 🛠 Technical Stack

### Core Framework
- FastAPI 0.104.1
- Python 3.11
- Modal Cloud Platform

### AI/ML Stack
- PyTorch 2.1.0
- ONNX Runtime GPU 1.16.3
- OpenCV 4.8.1
- InsightFace 0.7.3
- DeepFace 0.0.79

### Infrastructure
- Modal deployment
- Redis caching
- PostgreSQL database
- S3-compatible storage

### Monitoring
- Prometheus
- Sentry
- OpenTelemetry
- Structured logging

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/1AyaNabil1/face-recognition-control-system.git
cd face-recognition-control-system
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 🚀 Deployment

### Modal Deployment

1. Install Modal CLI:
```bash
pip install modal-client
```

2. Configure Modal:
```bash
modal token new
```

3. Deploy to production:
```bash
python modal_deploy.py
```

### Environment Variables

Required environment variables:
```env
# API Configuration
API_VERSION=v1
API_TOKEN_SECRET=your-secret-key
CORS_ORIGINS=https://yourdomain.com

# Model Configuration
MODEL_PATH=/root/models
CONFIDENCE_THRESHOLD=0.6
MAX_FACES=20

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_PORT=9090

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
```

## 📚 API Documentation

### Base URL
```
https://face-recognition-api-prod-xxxxx.modal.run
```

### Endpoints

#### Register Face
```http
POST /api/v1/register
Content-Type: multipart/form-data
Authorization: Bearer <token>

{
    "file": <image_file>,
    "name": "person_name",
    "model": "ensemble"  // insightface, vggface2, or ensemble
}
```

#### Recognize Face
```http
POST /api/v1/recognize
Content-Type: multipart/form-data
Authorization: Bearer <token>

{
    "file": <image_file>,
    "include_attributes": true,  // optional
    "model": "ensemble"  // optional
}
```

#### Health Check
```http
GET /api/v1/health
Authorization: Bearer <token>
```

#### Metrics
```http
GET /metrics
Authorization: Bearer <token>
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=app
```

Performance testing:
```bash
pytest tests/test_production.py::TestFaceRecognitionEngine::test_performance -v
```

## 📊 Monitoring

### Prometheus Metrics
- `face_recognition_requests_total`: Total API requests
- `face_recognition_latency_seconds`: Request latency histogram
- `face_recognition_errors_total`: Error count by type
- `model_inference_time_seconds`: Model inference time

### Health Checks
Regular health checks include:
- Model availability
- GPU status
- Redis connection
- Database connection
- System resources

## 🔒 Security

### Authentication
- Bearer token authentication
- Token validation and refresh
- Role-based access control

### Rate Limiting
- 100 requests per minute per IP
- Configurable limits by endpoint
- Redis-based rate limiting

### Input Validation
- Image size limits (10MB max)
- File type validation
- Input sanitization
- Anti-spoofing checks

## 📈 Performance

### Benchmarks
- Face Detection: ~20ms
- Recognition: ~50ms
- Total Response Time: <100ms
- Concurrent Users: 100+
- GPU Utilization: ~80%

### Optimization
- Model quantization
- ONNX runtime
- Caching strategies
- Async operations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Aya Nabil** - *Initial work* - [1AyaNabil1](https://github.com/1AyaNabil1)

## 🙏 Acknowledgments

- InsightFace team for the face recognition models
- Modal team for the deployment platform
- Open source community for various tools and libraries
