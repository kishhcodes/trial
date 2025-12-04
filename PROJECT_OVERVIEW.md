# KRIZZIP Project Overview

## 🎯 Project Purpose

**Krizzip** is an intelligent attention and focus monitoring application designed to help users, particularly neurodivergent individuals, maintain productivity by detecting when they're distracted and suggesting breaks at the right moments.

The system uses real-time facial analysis to:
1. **Recognize who is using the application** (Face Recognition)
2. **Detect their emotional state** (Emotion Detection)
3. **Monitor their attention level** (Eye gaze, head position, body movement)
4. **Suggest breaks when needed** (Adaptive break recommendations)
5. **Log all data for analysis** (Both JSON and MongoDB)

---

## 🏗️ Project Architecture

### Tech Stack
- **Backend**: FastAPI (Python web framework)
- **Computer Vision**: OpenCV, MediaPipe, face_recognition
- **Database**: MongoDB (for analytics and logging)
- **Frontend**: HTML/CSS/JavaScript (Jinja2 templates)
- **Analytics**: Metabase (can be deployed via Docker)

### Core Components

```
krizzip/
├── main.py                          # FastAPI application entry point
├── services/                        # Business logic
│   ├── camera_service.py           # Webcam frame capture and streaming
│   ├── face_service.py             # Face recognition with known faces
│   ├── attention_service.py        # Attention tracking algorithms
│   ├── mongo_service.py            # MongoDB connection & operations
│   ├── mongo_adapter.py            # Bridge between attention & MongoDB
│   ├── user_service.py             # User session management
│   └── emotion_service.py          # (Empty placeholder)
├── routers/                        # API endpoints
│   ├── video.py                    # Video streaming endpoint
│   ├── pages.py                    # Web pages (HTML responses)
│   ├── status.py                   # System status & monitoring
│   ├── logs.py                     # Log retrieval endpoints
│   └── auth.py                     # Authentication endpoints
├── templates/                      # HTML files
│   ├── index.html                  # Main monitoring page
│   ├── waiting.html                # Login/waiting page
│   └── break.html                  # Break suggestion page
├── static/                         # CSS and static assets
│   └── style.css                   # Styling
├── known_faces/                    # Reference images for face recognition
│   ├── Krishnaa_S.png
│   └── madhu.png
└── logs/                          # JSON logs (file-based backup)
```

---

## 🔄 Data Flow Diagram

```
User at Webcam
     ↓
[Camera Service] → Captures video frames
     ↓
[Attention Service] → Analyzes:
     ├─ Eye closure duration (EAR)
     ├─ Head position/movement
     ├─ Gaze direction
     └─ Distraction patterns
     ↓
[Dual Logging System]:
     ├─ JSON files (logs/)
     └─ MongoDB (via mongo_service)
     ↓
[API Endpoints] → Return data to frontend
     ↓
[Frontend] → Display monitoring UI & suggest breaks
```

---

## 🎬 Key Features

### 1. **Face Recognition**
- **Service**: `services/face_service.py`
- **Method**: Loads reference images from `known_faces/` directory
- **Technology**: face_recognition library (deep learning)
- **Output**: Identifies current user by name

### 2. **Attention Tracking**
- **Service**: `services/attention_service.py`
- **Metrics Tracked**:
  - **Eye Aspect Ratio (EAR)**: Detects eye closure and blink patterns
  - **Head Pose**: Uses MediaPipe face mesh (468 facial landmarks)
  - **Gaze Direction**: Determines where user is looking
  - **Distraction Counter**: Increments when attention drops
  - **Adaptation Factor**: Personalizes thresholds per user

### 3. **Adaptive Thresholds**
- **Initial Threshold**: 75 frames (~10-15 seconds at 5-7 FPS)
- **Dynamic Adjustment**: Based on individual movement patterns
- **Range**: 50-125 frames (adjusts for different user behaviors)
- **Neurodivergent-Friendly**: More forgiving thresholds for users with ADHD, autism, etc.

### 4. **Break Suggestions**
- **Trigger**: When distraction counter exceeds threshold
- **Behavior**: 
  - Suggests a break to the user
  - Offers options (continue monitoring or take a break)
  - Logs the decision
  - Adapts future thresholds based on user behavior

### 5. **Dual Logging System**
- **JSON Logs**: `logs/` directory (backup, human-readable)
  - `logs/users/{username}_log.json`
  - `logs/distraction_log.json`
- **MongoDB**: Production database (queryable, scalable)
  - Collections: `users`, `sessions`, `distraction_events`, `app_logs`
  - Enables analytics and pattern analysis

---

## 📊 API Endpoints

### Page Routes
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home/waiting page (authentication check) |
| `/waiting` | GET | Waiting for face detection |
| `/watch` | GET | Main monitoring page (requires auth) |
| `/break_page` | GET | Break suggestion page |

### Video & Status Routes
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video_feed` | GET | Real-time video stream with overlays |
| `/status` | GET | Current emotion, attention, face recognition, break suggestion |
| `/face_recognition` | GET | Face recognition details only |
| `/emotion_only` | GET | Emotion detection results only |

### Authentication Routes
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | Authenticate user |
| `/api/auth/logout` | POST | End authentication |
| `/api/auth/status` | GET | Check authentication status |

### Logging & Data Routes
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/logs/distraction` | GET | Get distraction logs |
| `/api/logs/user/{username}` | GET | Get user-specific logs |
| `/api/mongodb/status` | GET | MongoDB connection status |

---

## 🔐 Authentication Flow

1. **User accesses `/`** → Shown waiting.html (login page)
2. **User enters username** → Triggers face detection
3. **Face is recognized** → User authenticated
4. **User redirected to `/watch`** → Full monitoring begins
5. **MongoDB session created** → Tracking starts
6. **User leaves monitoring** → Session ends, data logged

---

## 📦 MongoDB Collections Structure

### 1. **users**
```json
{
  "username": "Krishnaa S",
  "registered_date": "2025-09-18T10:00:00Z",
  "total_sessions": 5,
  "notes": "Neurodivergent user"
}
```

### 2. **sessions**
```json
{
  "username": "Krishnaa S",
  "session_id": "ObjectId",
  "start_time": "2025-09-18T10:00:00Z",
  "end_time": "2025-09-18T10:30:00Z",
  "duration_minutes": 30,
  "distraction_events": 3
}
```

### 3. **distraction_events**
```json
{
  "username": "Krishnaa S",
  "session_id": "ObjectId",
  "start_time": "2025-09-18T10:05:00Z",
  "end_time": "2025-09-18T10:08:00Z",
  "duration_seconds": 180,
  "threshold_reached": true,
  "threshold_used": 75,
  "adaptation_factor": 1.0
}
```

### 4. **app_logs**
```json
{
  "event": "startup",
  "version": "1.0",
  "timestamp": "2025-09-18T10:00:00Z",
  "description": "Application started"
}
```

---

## 🧠 Attention Algorithm Details

### Eye Aspect Ratio (EAR)
```
EAR = ||p2 - p6|| + ||p3 - p5|| / (2 × ||p1 - p4||)
```
- Detects when eyes are closed
- Blink detection: EAR < 0.2
- Sustained closure: triggers attention loss

### Distraction Detection
```
if EAR < threshold AND head_not_neutral AND gaze_unfocused:
    distraction_counter += 1
else:
    distraction_counter -= RECOVERY_RATE

if distraction_counter > DISTRACTION_THRESHOLD:
    → Suggest break
```

### Neurodivergent-Friendly Adaptations
- **Larger attention buffer**: 15 frames instead of 10 (less reactive)
- **Slower recovery rate**: 4 frames per second (more forgiving)
- **Pattern recognition**: Adjusts to individual's natural movement
- **Dynamic threshold**: Personalizes based on behavior patterns

---

## 🚀 Deployment Options

### 1. **Local Development**
```bash
# Activate virtual environment
source projenv/bin/activate

# Run the application
uvicorn main:app --reload

# Access at http://localhost:8000
```

### 2. **Docker (for Metabase)**
```bash
# Start Metabase
docker run -d -p 3000:3000 \
  --add-host=host.docker.internal:host-gateway \
  --name metabase metabase/metabase

# Access Metabase at http://localhost:3000
```

### 3. **Production Considerations**
- Use gunicorn instead of uvicorn
- Enable MongoDB authentication (currently enabled with user: `krizzuser`)
- Use HTTPS/SSL certificates
- Set up reverse proxy (nginx/Apache)
- Configure environment variables securely

---

## 📈 Analytics with Metabase

### Connecting MongoDB to Metabase
1. Open Metabase (http://localhost:3000)
2. Admin Settings → Databases → Add database
3. **Configuration**:
   - Name: "Krizzip MongoDB"
   - Type: MongoDB
   - Host: `host.docker.internal` (or your host IP)
   - Port: `27017`
   - Database: `krizzip`
   - Username: `krizzuser`
   - Password: `secure_password`
   - Auth Database: `krizzip`

### Sample Analytics Queries
- Average distraction duration per user
- Most common distraction times
- Break effectiveness (did user stay focused after break?)
- Adaptation factor trends
- Session patterns over time

---

## 🔧 Configuration

### Environment Variables (.env)
```properties
# MongoDB Connection Settings
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=krizzip
MONGO_USER=krizzuser
MONGO_PASSWORD=secure_password
```

### Attention Parameters (attention_service.py)
- `INITIAL_THRESHOLD`: Starting distraction threshold
- `MAX_THRESHOLD`: Maximum for very active users
- `MIN_THRESHOLD`: Minimum for users needing frequent breaks
- `ATTENTION_RECOVERY_RATE`: How fast attention recovers
- `attention_buffer_size`: Smoothing for state changes

---

## 📝 Logging & Testing

### Test Files
- `test_mongo.py`: Tests MongoDB connectivity
- `test_simple_mongo.py`: Simple authentication test
- `test_full_application.py`: Full application test with dual logging
- `test_dual_logging.py`: Tests JSON + MongoDB logging

### Running Tests
```bash
source projenv/bin/activate
python test_mongo.py
```

---

## 🎯 Use Cases

### 1. **ADHD Support**
- Helps users with ADHD maintain focus
- Adaptive thresholds accommodate natural movement
- Regular break suggestions improve productivity

### 2. **Student Focus Management**
- Monitor attention during study sessions
- Track patterns to optimize study schedules
- Generate productivity reports

### 3. **Workplace Wellness**
- Prevent burnout by suggesting breaks
- Create aggregated productivity reports
- Analyze team focus patterns

### 4. **Accessibility Tool**
- Designed with neurodiversity in mind
- Customizable for different users
- Respects individual differences in attention

---

## 📚 Key Files to Understand

1. **main.py** - Application setup and routing
2. **services/attention_service.py** - Core attention algorithm
3. **services/mongo_adapter.py** - MongoDB integration (your focus file)
4. **services/mongo_service.py** - MongoDB operations
5. **services/camera_service.py** - Video capture
6. **routers/pages.py** - Web interface
7. **templates/index.html** - Frontend monitoring page

---

## 🔗 Next Steps

1. **Analytics**: Set up Metabase to visualize your MongoDB data
2. **Customization**: Adjust attention thresholds in `attention_service.py`
3. **Face Recognition**: Add more faces to `known_faces/` directory
4. **Frontend**: Enhance UI in templates and static CSS
5. **Scaling**: Deploy to production with proper security

---

## 📞 Support

For troubleshooting:
- Check MongoDB connection: `python test_mongo.py`
- Monitor application logs: Check console output and `app.log`
- Review MongoDB collections: Use Metabase or MongoDB shell
- Verify face recognition: Ensure faces are in `known_faces/` directory

