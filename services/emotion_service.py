# Emotion detection alternatives

# Option 1: DeepFace (more accurate, actively maintained)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("Using DeepFace for emotion detection")
except ImportError:
    print("DeepFace not available. Install with: pip install deepface")
    DEEPFACE_AVAILABLE = False

# Option 2: Mediapipe (faster, works well with the face_mesh we're already using)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Fall back to FER if needed
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    print("Warning: FER module could not be imported.")
    FER_AVAILABLE = False

# Set which library to use (change this to select your preferred method)
EMOTION_LIBRARY = "FER"  # Options: "DEEPFACE", "FER", "MEDIAPIPE"

# Initialize emotion detectors
emotion_detector = None
if EMOTION_LIBRARY == "DEEPFACE" and DEEPFACE_AVAILABLE:
    # DeepFace doesn't need initialization here
    pass
elif EMOTION_LIBRARY == "FER" and FER_AVAILABLE:
    try:
        emotion_detector = FER(mtcnn=True)
    except Exception as e:
        print(f"Error initializing FER: {e}")
        FER_AVAILABLE = False
elif EMOTION_LIBRARY == "MEDIAPIPE" and MEDIAPIPE_AVAILABLE:
    # We'll use the face_mesh from attention_service
    pass

# Add smoothing for emotion detection
emotion_buffer_size = 10
emotion_buffer = ["Neutral 😐"] * emotion_buffer_size
current_emotion = "Neutral 😐"

def detect_emotion(frame):
    """Detects emotion from a frame using the selected library."""
    global emotion_buffer, current_emotion
    
    try:
        raw_emotion = "Neutral 😐"
        
        # DeepFace approach - more accurate but can be slower
        if EMOTION_LIBRARY == "DEEPFACE" and DEEPFACE_AVAILABLE:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                if result and 'emotion' in result:
                    emotion = max(result['emotion'].items(), key=lambda x: x[1])[0]
                    raw_emotion = format_emotion(emotion)
            except Exception as e:
                print(f"DeepFace error: {e}")
        
        # Original FER approach
        elif EMOTION_LIBRARY == "FER" and FER_AVAILABLE and emotion_detector:
            try:
                result = emotion_detector.detect_emotions(frame)
                if result and len(result) > 0:
                    emotions = result[0]["emotions"]
                    if emotions:
                        # Get the emotion with the highest score
                        emotion, score = max(emotions.items(), key=lambda x: x[1])
                        raw_emotion = format_emotion(emotion)
            except Exception as e:
                print(f"Emotion detection error: {e}")
        
        # MediaPipe approach - faster, works with existing face_mesh
        elif EMOTION_LIBRARY == "MEDIAPIPE" and MEDIAPIPE_AVAILABLE:
            # This is simplified - would need a proper emotion model with MediaPipe
            # For now it returns neutral or a guess based on face landmarks
            raw_emotion = "Neutral 😐"
            
        # Apply smoothing to emotion detection
        emotion_buffer.pop(0)
        emotion_buffer.append(raw_emotion)
        
        # Get most common emotion in buffer
        emotions_count = {}
        for e in emotion_buffer:
            emotions_count[e] = emotions_count.get(e, 0) + 1
        
        # Only change if we have a new dominant emotion
        most_common = max(emotions_count.items(), key=lambda x: x[1])
        if most_common[1] > emotion_buffer_size // 2:
            current_emotion = most_common[0]
            
        return current_emotion
    
    except Exception as e:
        print(f"Emotion detection general error: {e}")
        return "Neutral 😐"

def format_emotion(emotion):
    """Format emotion string with emoji"""
    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "surprise": "😲", 
        "surprised": "😲",
        "neutral": "😐",
        "fear": "😨",
        "disgust": "🤢",
    }
    emoji = emoji_map.get(emotion.lower(), "")
    return f"{emotion.capitalize()} {emoji}"
