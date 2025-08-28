import cv2
from simple_facerec import SimpleFacerec

# Initialize face recognition from SimpleFacerec
sfr = SimpleFacerec()
try:
    sfr.load_encoding_images("known_faces/")
except Exception as e:
    print(f"Warning: Could not load face encodings: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Add frame counter
frame_count = 0
# Add variables to track face detection and recognition
last_face_count = 0
known_face_detected = False

# Add variable to track current recognized user
current_recognized_user = "Unknown"

def recognize_faces(frame, attention_state=None):
    """Recognize faces in frame and add labels"""
    global frame_count, last_face_count, known_face_detected, current_recognized_user
    frame_count += 1
    
    # Make a copy of the frame to avoid modifying the original
    display_frame = frame.copy()
    
    try:
        # Try to use face recognition
        face_locations, face_names = sfr.detect_known_faces(frame)
        
        # Update the face count
        last_face_count = len(face_locations)
        
        # Reset current user for this frame
        current_recognized_user = "Unknown"
        
        if len(face_locations) > 0:
            # Check if any known faces are detected
            for i, name in enumerate(face_names):
                if name != "Unknown":
                    known_face_detected = True
                    # Use the first recognized face as current user
                    if current_recognized_user == "Unknown":
                        current_recognized_user = name
                    break
            
            if frame_count % 30 == 0:  # Reduce logging
                print(f"Found {len(face_locations)} faces with recognition")
            
            for (y1, x2, y2, x1), name in zip(face_locations, face_names):
                # Safety checks to make sure coordinates are valid
                h, w = display_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                
                # Only draw rectangle and name, not attention state
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw name but NOT attention
                    try:
                        text_y_pos = max(30, y1-10)
                        cv2.putText(display_frame, name, (x1, text_y_pos), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 200), 2)
                    except Exception as text_err:
                        pass
    except Exception as recog_err:
        # Fallback to simple face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # Draw rectangle but not attention state
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return display_frame

def get_face_data(frame):
    """Get information about faces in the frame"""
    face_data = {"name": "Unknown", "recognized": False}
    
    try:
        face_locations, face_names = sfr.detect_known_faces(frame)
        if face_locations and face_names:
            # Use the first face detected
            face_data = {"name": face_names[0], "recognized": face_names[0] != "Unknown"}
    except Exception as e:
        print(f"Error in face data detection: {e}")
    
    return face_data

def is_known_face_detected():
    """Check if a known face is currently detected"""
    global known_face_detected
    return known_face_detected

def get_current_user(frame=None):
    """Get the name of the currently recognized user"""
    global current_recognized_user
    
    # If frame provided, try to recognize face first
    if frame is not None:
        try:
            face_locations, face_names = sfr.detect_known_faces(frame)
            if face_locations and face_names:
                for name in face_names:
                    if name != "Unknown":
                        current_recognized_user = name
                        break
        except Exception:
            pass
    
    return current_recognized_user