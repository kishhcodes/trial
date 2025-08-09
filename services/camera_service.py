import cv2
import time
import numpy as np
import threading
from services.face_service import recognize_faces
from services.emotion_service import detect_emotion
from services.attention_service import detect_attention, check_distraction

# Global state variables
attention_state = "Attentive"
emotion_state = "Neutral 😐"
distraction_start_time = None
distraction_threshold = 10  # seconds
needs_redirect = False

# Initialize camera - use a singleton pattern to ensure only one instance
class CameraManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                # Initialize the camera here
                cls._instance.camera = cv2.VideoCapture(0)
                
                # Check if camera opened successfully
                if not cls._instance.camera.isOpened():
                    print("Error: Could not open camera.")
                    # Try another camera index as fallback
                    cls._instance.camera = cv2.VideoCapture(1)
                    if not cls._instance.camera.isOpened():
                        print("Error: Could not open backup camera.")
                        
                # Set camera properties for better performance
                cls._instance.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cls._instance.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
        return cls._instance

    def get_frame(self):
        """Get a single frame from the camera"""
        if not self.camera.isOpened():
            print("Warning: Camera not opened. Trying to reopen...")
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Error: Could not reopen camera.")
                return None
            
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Could not read frame.")
            return None
        return frame
    
    def release(self):
        """Release the camera resources"""
        if self.camera.isOpened():
            self.camera.release()

# Get the camera manager instance
camera_manager = CameraManager()

def get_frame():
    """Get a single processed frame from the camera"""
    return camera_manager.get_frame()

def process_frame(frame):
    """Process a frame with all detection services"""
    global attention_state, emotion_state, distraction_start_time, needs_redirect
    
    if frame is None:
        return None
    
    try:
        # Get emotion from frame
        emotion_label = detect_emotion(frame)
        emotion_state = emotion_label
        
        # Face recognition
        frame_with_faces = recognize_faces(frame, emotion_label)
        
        # Attention detection
        current_attention = detect_attention(frame)
        attention_state = current_attention
        
        # Check distraction status and update frame
        needs_redirect, distraction_start_time, seconds_distracted = check_distraction(
            current_attention, distraction_start_time, distraction_threshold)
        
        # Add UI elements
        if emotion_label.startswith(("Sad", "Angry")) or attention_state == "Distracted":
            cv2.putText(frame_with_faces, "💤 Break Time!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        
            # Add distraction timer if we're tracking distraction
            if distraction_start_time is not None:
                cv2.putText(frame_with_faces, f"Distracted for: {seconds_distracted}s", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
        # Add status text
        cv2.putText(frame_with_faces, f"Attention: {attention_state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame_with_faces, f"Emotion: {emotion_state}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
        if needs_redirect:
            cv2.putText(frame_with_faces, "Redirecting to break page...", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame_with_faces
    except Exception as e:
        print(f"Error processing frame: {e}")
        # Return the original frame if processing fails
        return frame

def gen_frames():
    """Generate frames for streaming"""
    while True:
        try:
            frame = get_frame()
            if frame is None:
                # If no frame, wait a bit then try again
                time.sleep(0.1)
                continue
                
            processed_frame = process_frame(frame)
            if processed_frame is None:
                continue
                
            ret, buf = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in frame generation: {e}")
            # Wait a bit before trying again
            time.sleep(0.5)
            continue

def get_status():
    """Get current status data"""
    global attention_state, emotion_state, needs_redirect
    
    # Get current face data
    face_data = {"name": "Unknown", "recognized": False}
    
    try:
        frame = get_frame()
        if frame is not None:
            from services.face_service import get_face_data
            face_data = get_face_data(frame)
    except Exception as e:
        print(f"Error in status detection: {e}")
    
    # Determine if a break should be suggested
    should_break = False
    if attention_state == "Distracted" or emotion_state.startswith(("Sad", "Angry")):
        should_break = True
    
    redirect_status = needs_redirect
    if needs_redirect:
        needs_redirect = False  # Reset after checking
        
    return {
        "attention": attention_state,
        "emotion": emotion_state,
        "face": face_data,
        "should_break": should_break,
        "redirect": redirect_status
    }
