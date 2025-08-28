import cv2
import time
import numpy as np
from services.face_service import recognize_faces
from services.attention_service import detect_attention, check_distraction
# Import only the needed functions to avoid circular imports
from services.user_service import check_user_authentication, get_authentication_status
from services.user_service import is_monitoring_active, get_locked_user

# Global state variables
attention_state = "Attentive"
distraction_start_time = None
distraction_threshold = 10  # seconds
needs_redirect = False

# Add a variable to track if a face is detected
face_detected = False

# Add a variable to track if a known face is detected
known_face_detected = False

# Initialize camera
camera = None

# Add a more explicit redirect trigger
redirect_triggered = False
redirect_trigger_time = None
REDIRECT_DELAY_SECONDS = 3  # Wait this many seconds before actually redirecting

def initialize_camera():
    """Initialize the camera if not already done"""
    global camera
    if camera is None or not camera.isOpened():
        print("Initializing camera...")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Warning: Failed to open camera index 0, trying index 1...")
            camera = cv2.VideoCapture(1)
        
        if camera.isOpened():
            print("Camera initialized successfully")
        else:
            print("ERROR: Could not initialize any camera!")
    return camera.isOpened()

def get_frame():
    """Get a single processed frame from the camera"""
    if not initialize_camera():
        print("No camera available")
        return None
    
    ret, frame = camera.read()
    if not ret:
        print("Failed to read frame from camera")
        return None
    return frame

def process_frame(frame):
    """Process a frame with attention detection services"""
    global attention_state, distraction_start_time, needs_redirect
    global known_face_detected, redirect_triggered, redirect_trigger_time
    
    try:
        if frame is None:
            return None
        
        # Check if we're on the monitoring page or still authenticating
        on_monitoring_page = is_monitoring_active()
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        if on_monitoring_page:
            # On watch page - skip authentication and use locked user
            is_authenticated = True
            current_user = get_locked_user() or "Unknown"
            
            # Always proceed with attention tracking on watch page
            current_attention = detect_attention(frame)
            attention_state = current_attention
            
            # Check distraction status
            needs_redirect_now, distraction_start_time, seconds_distracted = check_distraction(
                current_attention, distraction_start_time, distraction_threshold, current_user)
            
            if needs_redirect_now:
                # Set a stronger redirect trigger that won't be reset accidentally
                if not redirect_triggered:
                    redirect_triggered = True
                    redirect_trigger_time = time.time()
                    print(f"⚠️ REDIRECT TRIGGERED for user {current_user}! Waiting {REDIRECT_DELAY_SECONDS}s before redirect.")
                
                # After a small delay, set the actual redirect flag (this gives time to see the alert on screen)
                if redirect_trigger_time and time.time() - redirect_trigger_time > REDIRECT_DELAY_SECONDS:
                    needs_redirect = True
                    print(f"🚨 REDIRECT FLAG ACTIVATED for user {current_user}!")
            
            # Face recognition just for display (no authentication logic)
            display_frame = recognize_faces(display_frame, None)
            
            # Add attention UI elements
            cv2.putText(display_frame, f"User: {current_user}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                       
            cv2.putText(display_frame, f"Attention: {attention_state}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                   
            if attention_state == "Distracted":
                cv2.putText(display_frame, "💤 Break Time!", (50, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                
                # Add distraction timer
                if distraction_start_time is not None:
                    cv2.putText(display_frame, f"Distracted for: {seconds_distracted}s", (50, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add very clear redirect visual when triggered
            if redirect_triggered:
                # Draw a red border around the entire frame
                border_thickness = 20
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (0, 0), (w, h), (0, 0, 255), border_thickness)
                
                # Add countdown to redirect
                if redirect_trigger_time:
                    time_until_redirect = max(0, REDIRECT_DELAY_SECONDS - (time.time() - redirect_trigger_time))
                    cv2.putText(display_frame, f"BREAK TIME IN: {time_until_redirect:.1f}s", 
                               (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            # Authentication page - full authentication logic
            is_authenticated, current_user = check_user_authentication(frame)
            known_face_detected = is_authenticated
            
            # Reset attention variables when not on monitoring page
            attention_state = "Attentive"
            distraction_start_time = None
            
            # Face recognition for authentication
            display_frame = recognize_faces(display_frame, None)
            
            # Show authentication UI elements
            auth_status = get_authentication_status()
            if auth_status["authenticated"]:
                cv2.putText(display_frame, f"User: {auth_status['user']} (Authenticated)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, "Redirecting to monitoring...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif auth_status["authentication_in_progress"]:
                progress = f"{auth_status['detection_progress']}/{auth_status['detection_needed']}"
                cv2.putText(display_frame, f"Authenticating: {progress}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Please look at the camera to authenticate", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
        return display_frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        # Return a blank frame with error message if something went wrong
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, f"Error: {str(e)}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return blank_frame

def gen_frames():
    """Generate frames for streaming with improved error handling"""
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
            print(f"Error generating frames: {str(e)}")
            # Generate an error frame rather than crashing
            try:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Camera error: {str(e)}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(error_frame, "Attempting to recover...", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                ret, buf = cv2.imencode('.jpg', error_frame)
                if ret:
                    frame_bytes = buf.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except:
                pass
            time.sleep(1)  # Longer delay to avoid rapid error loops

def get_status():
    """Get current status data with more explicit redirect info"""
    global attention_state, needs_redirect, redirect_triggered
    
    # Get authentication status - simpler for monitoring page
    if is_monitoring_active():
        locked_user = get_locked_user() or "Unknown"
        face_data = {"name": locked_user, "recognized": True}
        should_break = (attention_state == "Distracted")
        
        return {
            "attention": attention_state,
            "face": face_data,
            "should_break": should_break,
            "redirect": needs_redirect,
            "redirect_triggered": redirect_triggered,  # Add this for more debugging
            "authenticated": True,
            "monitoring_active": True
        }
    else:
        # Authentication page status
        auth_status = get_authentication_status()
        face_data = {"name": auth_status["user"], "recognized": auth_status["authenticated"]}
        
        return {
            "attention": "Not Tracking",
            "face": face_data,
            "should_break": False,
            "redirect": False,
            "authenticated": auth_status["authenticated"],
            "authentication_in_progress": auth_status["authentication_in_progress"],
            "detection_progress": auth_status["detection_progress"],
            "detection_needed": auth_status["detection_needed"],
            "monitoring_active": False
        }

# Add a function to explicitly check if a face is detected
def is_face_detected():
    """Check if a face is currently detected"""
    global face_detected
    return face_detected

# Update the is_known_face_detected function to use authentication status
def is_known_face_detected():
    """Check if a known face is currently detected and authenticated"""
    auth_status = get_authentication_status()
    return auth_status["authenticated"]

def reset_redirect_flag():
    """Reset the redirect flag after successful redirection"""
    global needs_redirect, distraction_start_time, redirect_triggered, redirect_trigger_time
    was_redirecting = needs_redirect
    
    # Reset all redirect-related variables
    needs_redirect = False
    redirect_triggered = False
    redirect_trigger_time = None
    distraction_start_time = None
    
    print(f"🔄 All redirect flags reset! Was redirecting: {was_redirecting}")
    return was_redirecting

def release_camera():
    """Release the camera when app is shutting down"""
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        print("Camera released")