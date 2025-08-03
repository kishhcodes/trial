# Face Emotion & Attention Detection API

A FastAPI-based web API for face emotion and attention detection with real-time video streaming capabilities.

## Features

- Face recognition with name identification
- Emotion detection (happy, sad, angry, etc.)
- Attention tracking
- Break suggestions when user is distracted or sad
- Real-time video streaming
- Swagger UI documentation

## Requirements

- Python 3.11+
- OpenCV
- MediaPipe
- FER (Facial Emotion Recognition)
- face_recognition
- FastAPI
- Uvicorn

## Installation

### Option 1: Using Python Virtual Environment

1. Clone this repository
2. Create and activate a virtual environment
   ```bash
   python -m venv projenv
   source projenv/bin/activate  # On Windows: projenv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Docker

1. Clone this repository
2. Build and run using Docker Compose
   ```bash
   docker-compose up -d
   ```

## Usage

### Start the API Server

```bash
uvicorn main:app --reload
```

Or using Docker:

```bash
docker-compose up -d
```

### Access the API

- API Documentation (Swagger UI): http://localhost:8000/docs
- Video Stream: http://localhost:8000/video_feed
- API Status: http://localhost:8000/status
- Face Recognition: http://localhost:8000/face_recognition
- Emotion Detection Only: http://localhost:8000/emotion_only

### Add Faces for Recognition

Add person's face images to the `known_faces/` directory with the filename as the person's name (e.g., `john.jpg`).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to Swagger UI documentation |
| `/docs` | GET | Swagger UI API documentation |
| `/video_feed` | GET | Live video stream with face recognition, emotion, and attention detection |
| `/status` | GET | Current emotion, attention status, face recognition, and break suggestion |
| `/face_recognition` | GET | Face recognition details from current frame |
| `/emotion_only` | GET | Emotion detection results only |

## Notes for Docker Users

- When running in Docker, camera access requires additional configuration
- Uncomment the devices section in docker-compose.yml to use your camera
- For production use, consider setting up a frontend application to consume these API endpoints

## License

MIT
# krizzip
