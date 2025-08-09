import cv2
from simple_facerec import SimpleFacerec

# Initialize face recognition from SimpleFacerec
sfr = SimpleFacerec()
try:
    sfr.load_encoding_images("known_faces/")
except Exception as e:
    print(f"Warning: Could not load face encodings: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def recognize_faces(frame, emotion_label="Neutral"):
    """Recognize faces in frame and add labels"""
    try:
        face_locations, face_names = sfr.detect_known_faces(frame)
        
        for (y1, x2, y2, x1), name in zip(face_locations, face_names):
            # Draw face box & name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 200), 2)
            cv2.putText(frame, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
    except Exception as e:
        # Fallback to simple face detection if face recognition fails
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

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
