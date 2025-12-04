# KRIZZIP System Architecture

## 🏗️ Complete System Overview

```
                            ┌─────────────────────────────────────┐
                            │   USER at COMPUTER                  │
                            │   (Looking at webcam)               │
                            └──────────────┬──────────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────────────┐
                            │   WEBCAM / CAMERA DEVICE            │
                            │   (Physical hardware)               │
                            └──────────────┬──────────────────────┘
                                           │
                                           ▼
                      ┌────────────────────────────────────────────┐
                      │         KRIZZIP APPLICATION               │
                      │        (FastAPI on Port 8000)            │
                      └────────────────────────────────────────────┘
                                           │
                ┌──────────────────────────┼──────────────────────────┐
                │                          │                          │
                ▼                          ▼                          ▼
        ┌──────────────┐          ┌──────────────┐        ┌──────────────┐
        │ CAMERA SRV   │          │ FACE SRV     │        │ ATTENTION    │
        │              │          │              │        │ SRV          │
        │ • Capture    │          │ • Recognition │        │              │
        │ • Preprocess │          │   (ML model) │        │ • Eye aspect │
        │ • Send frames│          │ • ID user    │        │   ratio      │
        └──────┬───────┘          └──────┬───────┘        │ • Head pose  │
               │                         │                │ • Gaze direction
               │                         │                │ • Distraction
               └─────────────────────────┼────────────────┤   counter    │
                                         │                │ • Threshold  │
                                         │                │   adaptive   │
                                         │                └──────┬───────┘
                                         │                       │
                                         └───────────────┬───────┘
                                                         │
                                    ┌────────────────────┴─────────────────┐
                                    │                                      │
                                    ▼                                      ▼
                            ┌──────────────────┐              ┌──────────────────┐
                            │  MONGO ADAPTER   │              │   JSON LOGGER    │
                            │                  │              │                  │
                            │ • Bridges services               │ • Local backup   │
                            │   to MongoDB     │              │ • Human readable │
                            └────────┬─────────┘              │ • File-based     │
                                     │                        └────────┬─────────┘
                                     │                                 │
                    ┌────────────────┴──────────────┐                 │
                    │                               │                 │
                    ▼                               ▼                 │
            ┌──────────────────┐        ┌──────────────────┐         │
            │  MONGO SERVICE   │        │                  │         │
            │                  │        │  JSON FILES      │         │
            │ • Database ops   │        │                  │         │
            │ • Collections    │        │ logs/            │         │
            │ • Connections    │        │ ├── distraction  │         │
            └────────┬─────────┘        │ │   _log.json    │         │
                     │                  │ └── users/       │         │
                     │                  │     └── *.json   │         │
                     │                  └──────────────────┘         │
                     │                                               │
                     ▼                                               ▼
            ┌──────────────────────────────────────────────────────────┐
            │            DATA PERSISTENCE LAYER                        │
            └──────────────────────────────────────────────────────────┘
                     │                                               │
                     ▼                                               ▼
            ┌──────────────────┐                        ┌──────────────────┐
            │    MONGODB       │                        │   FILESYSTEM     │
            │  (localhost:27017)                        │   (Disk storage) │
            │                  │                        │                  │
            │ Collections:     │                        │ Formats: JSON    │
            │ • users          │                        │ Structure: Dated │
            │ • sessions       │                        │ folders & files  │
            │ • distraction_   │                        └──────────────────┘
            │   events         │
            │ • app_logs       │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────────────────────────┐
            │      API ENDPOINTS                   │
            │  (Return data to Frontend)           │
            │                                      │
            │ • /video_feed                        │
            │ • /status                            │
            │ • /face_recognition                  │
            │ • /api/mongodb/status                │
            │ • /api/logs/                         │
            └──────────────┬───────────────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │   FRONTEND (Browser)     │
                │                          │
                │ • HTML Templates         │
                │ • CSS Styling            │
                │ • JavaScript (optional)  │
                │                          │
                │ Pages:                   │
                │ • /index.html            │
                │ • /waiting.html          │
                │ • /break.html            │
                └──────────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  USER SEES:          │
                    │                      │
                    │ • Live video feed    │
                    │ • Attention status   │
                    │ • Break suggestions  │
                    │ • Emotion detection  │
                    └──────────────────────┘


                          ANALYTICS LAYER
                    (Optional - Metabase Docker)

            ┌────────────────────────────────────────┐
            │    METABASE (Docker Container)         │
            │    (Port 3000)                         │
            │                                        │
            │ Connects to MongoDB at:                │
            │ host.docker.internal:27017             │
            └────────────────┬───────────────────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │  ANALYTICS QUERIES │
                    │  & DASHBOARDS      │
                    │                    │
                    │ • User stats       │
                    │ • Focus patterns   │
                    │ • Trends           │
                    │ • Reports          │
                    └────────────────────┘
```

---

## 📊 Data Flow Detailed

### Real-Time Processing Loop (5-7 FPS)

```
FRAME n
│
├─ Camera Service
│  └─ Capture frame from webcam
│
├─ Face Service
│  ├─ Detect face in frame
│  ├─ Compare to known faces
│  └─ Return username & confidence
│
├─ Attention Service
│  ├─ Extract 468 facial landmarks (MediaPipe)
│  ├─ Calculate Eye Aspect Ratio (EAR)
│  ├─ Determine head pose
│  ├─ Detect gaze direction
│  ├─ Update distraction counter
│  ├─ Compare to threshold
│  └─ Return: Attention state + counter + recommendation
│
├─ Dual Logging
│  ├─ JSON Logger
│  │  └─ Append to logs/users/{username}_log.json
│  │
│  └─ MongoDB Logger (via mongo_adapter)
│     └─ Insert/update MongoDB collections
│
├─ API Response
│  └─ Format data for frontend
│
├─ Frontend Render
│  ├─ Display video feed
│  ├─ Show attention indicator
│  ├─ Display counter
│  └─ Suggest break if needed
│
└─ FRAME n+1
   (Repeat 5-7 times per second)
```

---

## 🔄 Session Lifecycle

```
USER OPENS APP (Port 8000)
│
├─ Route: "/" (waiting.html)
│  └─ User sees: "Waiting for face detection..."
│
├─ User shows face to camera
│  │
│  ├─ Camera captures frames
│  ├─ Face service detects face
│  ├─ Face recognized? ✓ YES
│  │
│  └─ User authenticated as: "Krishnaa S"
│
├─ redirect to "/watch?username=Krishnaa_S"
│  │
│  └─ mongo_service.start_user_session("Krishnaa S")
│     ├─ Create session document in MongoDB
│     ├─ Set session_id = newly created ObjectId
│     └─ Return session_id to caller
│
├─ MONITORING PAGE (index.html)
│  │
│  └─ Real-time loop starts:
│     └─ Capture frames → Process → Log → Display
│
│        For each frame:
│        ├─ Capture video
│        ├─ Track attention
│        ├─ Log to both:
│        │  ├─ JSON file (logs/users/Krishnaa_S_log.json)
│        │  └─ MongoDB (via mongo_adapter)
│        └─ Send to frontend
│
│        When distracted too long:
│        ├─ Log distraction event
│        ├─ Suggest break
│        └─ Show break.html
│
├─ USER TAKES BREAK or LEAVES APP
│  │
│  └─ mongo_service.end_user_session("Krishnaa S")
│     ├─ Update session document
│     ├─ Set end_time = now
│     ├─ Calculate duration
│     └─ Close session
│
├─ Route "/" again (back to waiting.html)
│
└─ Session data available for analytics:
   ├─ JSON logs
   ├─ MongoDB queries
   └─ Metabase dashboards
```

---

## 🗄️ MongoDB Collections Schema

### Collection: `users`
```
{
  _id: ObjectId("..."),
  username: "Krishnaa S",
  registered_date: ISODate("2025-09-18"),
  total_sessions: 5,
  total_attention_time: 14400,        // seconds
  average_attention_span: 2880,       // seconds (48 min)
  most_common_distraction_time: 1500, // seconds (25 min)
  notes: "Neurodivergent user"
}
```

### Collection: `sessions`
```
{
  _id: ObjectId("..."),
  username: "Krishnaa S",
  session_id: ObjectId("..."),
  start_time: ISODate("2025-09-18T10:00:00Z"),
  end_time: ISODate("2025-09-18T10:45:00Z"),
  duration_minutes: 45,
  duration_seconds: 2700,
  distraction_events: 2,
  break_suggestions_given: 2,
  breaks_taken: 1,
  final_attention_counter: 45,
  final_threshold_used: 75,
  device: "Laptop",
  location: "Home Office"
}
```

### Collection: `distraction_events`
```
{
  _id: ObjectId("..."),
  username: "Krishnaa S",
  session_id: ObjectId("..."),
  start_time: ISODate("2025-09-18T10:05:00Z"),
  end_time: ISODate("2025-09-18T10:08:00Z"),
  duration_seconds: 180,
  duration_minutes: 3,
  threshold_reached: true,
  threshold_used: 75,
  adaptation_factor: 1.0,
  break_suggested: true,
  break_accepted: false,
  eye_closure_primary: true,
  head_movement_secondary: false,
  gaze_unfocused: true
}
```

### Collection: `app_logs`
```
{
  _id: ObjectId("..."),
  timestamp: ISODate("2025-09-18T10:00:00Z"),
  event_type: "startup" | "shutdown" | "error",
  event: "Application started",
  version: "1.0.0",
  description: "Krizzip face attention monitoring system",
  status: "success" | "error",
  error_details: null
}
```

---

## 🌐 Network Topology

```
┌─────────────────────────────────────────────────────────────┐
│                      LOCALHOST (127.0.0.1)                  │
│                    Developer Environment                    │
│                                                             │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │   Krizzip App    │        │    MongoDB       │          │
│  │   Port: 8000     │◄──────►│   Port: 27017    │          │
│  │                  │        │                  │          │
│  │ • FastAPI        │        │ • krizzip db     │          │
│  │ • Services       │        │ • 4 collections  │          │
│  │ • Routers        │        │ • Authenticated  │          │
│  └────────┬─────────┘        └────────┬─────────┘          │
│           │                           │                    │
│           │                           │                    │
│           │ (Browser)                 │ (Backup)           │
│           ▼                           ▼                    │
│    ┌────────────────┐          ┌──────────────┐            │
│    │  Browser       │          │  JSON Files  │            │
│    │ :8000/         │          │  logs/       │            │
│    └────────────────┘          └──────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (Docker Network)
                            │
                ┌───────────────────────────┐
                │   Docker Container        │
                │   (Separate Environment)  │
                │                           │
                │  ┌─────────────────────┐  │
                │  │  Metabase          │  │
                │  │  Port: 3000        │  │
                │  │                     │  │
                │  │ Uses:               │  │
                │  │ host.docker.internal:27017
                │  │ (connects to local MongoDB)
                │  └─────────────────────┘  │
                │                           │
                └───────────────────────────┘
```

---

## 🔐 Authentication Flow

```
USER VISITS http://localhost:8000
│
├─ GET /  (Home page)
│  └─ Return waiting.html
│
├─ waiting.html loaded in browser
│  ├─ Shows: "Waiting for face detection..."
│  └─ JavaScript starts polling /face_recognition
│
├─ get /face_recognition (every 500ms)
│  │
│  └─ camera_service processes each frame:
│     ├─ Capture from webcam
│     ├─ Detect faces
│     ├─ For each face:
│     │  ├─ Extract face encoding
│     │  ├─ Compare to known_faces/* encodings
│     │  ├─ If match found:
│     │  │  └─ Return {"username": "Krishnaa S", "confidence": 0.95}
│     │  └─ If no match:
│     │     └─ Return {"username": "Unknown", "confidence": 0.5}
│     └─ Send JSON response
│
├─ Browser receives: {"username": "Krishnaa S", "confidence": 0.95}
│  │
│  ├─ Check: confidence > 0.8? ✓ YES
│  │
│  └─ Call: POST /api/auth/login
│     │
│     └─ user_service.set_authenticated_user("Krishnaa S")
│        ├─ Store username in session
│        ├─ Mark as authenticated
│        └─ Return {"success": true}
│
├─ Browser redirects to /watch?username=Krishnaa_S
│
├─ pages.py:verify_authentication() checks:
│  ├─ Is session authenticated? ✓ YES
│  └─ Return username
│
├─ Serve index.html with username embedded
│  │
│  └─ Monitoring page loads
│     ├─ start_user_session() called
│     ├─ MongoDB session created
│     └─ Real-time monitoring begins
│
└─ User is now authenticated & monitoring active
```

---

## 📈 Data Processing Pipeline

```
INPUT: Raw video frame (1920x1080 pixels, RGB)
│
├─ STEP 1: Face Detection
│  ├─ Input: Full frame
│  ├─ Process: MediaPipe face detection
│  ├─ Output: Face bounding box + 468 landmarks
│  └─ If no face: Skip frame, continue
│
├─ STEP 2: Face Recognition
│  ├─ Input: Face region from frame
│  ├─ Process: 
│  │  ├─ Extract face encoding (128-D vector)
│  │  ├─ Compare to known_faces/ encodings
│  │  ├─ Find closest match (Euclidean distance)
│  │  └─ Check confidence threshold
│  ├─ Output: {"username": "...", "confidence": 0.95}
│  └─ If no match: "Unknown"
│
├─ STEP 3: Attention Analysis
│  ├─ Input: 468 facial landmarks
│  ├─ Process:
│  │  ├─ Extract eye landmarks (left & right)
│  │  ├─ Calculate Eye Aspect Ratio (EAR)
│  │  │  EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
│  │  ├─ Extract head pose (3D angles)
│  │  ├─ Calculate gaze direction
│  │  ├─ Determine attention state
│  │  │  ├─ If EAR < 0.2 AND head_angle > threshold:
│  │  │  │  └─ DISTRACTED
│  │  │  └─ Else:
│  │  │     └─ ATTENTIVE
│  │  └─ Update counters
│  │
│  ├─ Output:
│  │  ├─ attention_state: "Attentive" | "Distracted"
│  │  ├─ distraction_counter: int (0-200)
│  │  ├─ threshold_exceeded: bool
│  │  └─ break_suggested: bool
│  │
│  └─ Special: Pattern Recognition
│     ├─ Track movement_patterns (last 50 frames)
│     ├─ Calculate adaptation_factor
│     └─ Adjust thresholds dynamically
│
├─ STEP 4: Dual Logging
│  │
│  ├─ JSON Logger
│  │  ├─ Create log entry:
│  │  │  {
│  │  │    "timestamp": "2025-09-18T10:00:01Z",
│  │  │    "state": "Attentive",
│  │  │    "counter": 45,
│  │  │    "threshold": 75,
│  │  │    "break_needed": false
│  │  │  }
│  │  └─ Append to logs/users/{username}_log.json
│  │
│  └─ MongoDB Logger (via mongo_adapter)
│     ├─ Call: mongo_log_distraction_event() if threshold hit
│     ├─ Insert document in distraction_events collection
│     └─ Update session document
│
├─ STEP 5: API Response
│  ├─ Format response JSON:
│  │  {
│  │    "username": "Krishnaa S",
│  │    "state": "Attentive",
│  │    "counter": 45,
│  │    "break_needed": false,
│  │    "emotion": "Neutral",
│  │    "confidence": 0.95
│  │  }
│  └─ Return to frontend
│
├─ STEP 6: Frontend Display
│  ├─ Update video feed (MJPEG stream)
│  ├─ Update status indicators
│  ├─ Show counter progress
│  ├─ Display break suggestion if needed
│  └─ Play notification sound (if enabled)
│
└─ OUTPUT: Monitoring results in browser + Data in MongoDB
```

---

## 🎯 Critical Attention Threshold Logic

```
DISTRACTION_THRESHOLD = 75 (adaptive, ranges 50-125)

Each Frame:
│
├─ Is user distracted?
│  ├─ EAR < 0.2 (eyes closed)?
│  ├─ Head tilted >30°?
│  └─ Gaze unfocused?
│
├─ If ALL/MOST true:
│  ├─ distraction_counter += 1
│  └─ Reached: counter (75)
│
├─ If NONE true:
│  ├─ distraction_counter -= 4 (recovery)
│  └─ Decreases faster than increases
│
├─ Smoothing (attention_buffer):
│  ├─ Uses 15-frame buffer
│  ├─ Requires 66% majority to change state
│  └─ Prevents jitter
│
└─ When counter >= 75:
   ├─ Suggest break
   ├─ Log distraction event
   ├─ Update MongoDB
   ├─ Calculate adaptation_factor
   └─ Adjust future threshold (50-125 range)
```

---

## 🔗 Service Interconnections

```
attention_service.py
├─ Uses:
│  ├─ camera_service (get frames)
│  ├─ mongo_adapter (log events)
│  └─ MediaPipe (eye/head detection)
│
└─ Called by:
   └─ camera_service (gen_frames)

mongo_adapter.py
├─ Uses:
│  └─ mongo_service (actual DB ops)
│
└─ Called by:
   ├─ attention_service
   ├─ user_service
   └─ Other services

mongo_service.py
├─ Uses:
│  └─ PyMongo (MongoDB driver)
│
└─ Called by:
   └─ mongo_adapter

face_service.py
├─ Uses:
│  ├─ camera_service (get frames)
│  ├─ SimpleFacerec (recognition)
│  └─ OpenCV (image processing)
│
└─ Called by:
   ├─ video.py (video_feed route)
   └─ routers/status.py

camera_service.py
├─ Uses:
│  ├─ OpenCV (camera access)
│  ├─ face_service (recognition)
│  └─ attention_service (tracking)
│
└─ Called by:
   └─ video.py (video_feed route)
```

---

**This architecture is designed to be:**
- **Modular**: Each service has clear responsibilities
- **Scalable**: Easy to add new features
- **Resilient**: Dual logging ensures no data loss
- **Adaptable**: Learns from user behavior
- **Accessible**: Designed for neurodivergent users

