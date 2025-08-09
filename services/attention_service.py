import cv2
import numpy as np
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Missing variable declarations
last_attention_time = time.time()
attention_timeout = 5

# Global counter for tracking distraction events
distraction_counter = 0
# Maximum counter value that triggers a redirect
DISTRACTION_THRESHOLD = 10

# Add state smoothing variables
attention_buffer_size = 10  # Number of frames to consider for smoothing
attention_buffer = ["Attentive"] * attention_buffer_size  # Initialize with "Attentive"
attention_change_threshold = 7  # Require 70% agreement to change state
current_smoothed_state = "Attentive"  # Smoothed state (what we actually use)

def detect_attention(frame):
    """Detect if user is attentive or distracted"""
    global last_attention_time, attention_buffer, current_smoothed_state
    
    # Default to current state if something goes wrong
    raw_current_state = current_smoothed_state
    
    try:
        # Detect raw attention state for this frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        # Determine raw state for current frame
        raw_current_state = "Distracted"  # Default state
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            def get_point(idx):
                return int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            left_eye_outer = get_point(33)
            left_eye_inner = get_point(133)
            right_eye_outer = get_point(362)
            right_eye_inner = get_point(263)
            left_iris = get_point(468)
            right_iris = get_point(473)
            left_eye_top = get_point(159)
            left_eye_bottom = get_point(145)
            right_eye_top = get_point(386)
            right_eye_bottom = get_point(374)
            def eye_aspect_ratio(top, bottom, left, right):
                vertical = np.linalg.norm(np.array(top) - np.array(bottom))
                horizontal = np.linalg.norm(np.array(left) - np.array(right))
                return vertical / horizontal if horizontal != 0 else 0
            left_ear = eye_aspect_ratio(left_eye_top, left_eye_bottom, left_eye_outer, left_eye_inner)
            right_ear = eye_aspect_ratio(right_eye_top, right_eye_bottom, right_eye_outer, right_eye_inner)
            def gaze_ratio(iris, outer, inner):
                return np.linalg.norm(np.array(iris) - np.array(inner)) / (np.linalg.norm(np.array(outer) - np.array(inner)) + 1e-6)
            left_ratio = gaze_ratio(left_iris, left_eye_outer, left_eye_inner)
            right_ratio = gaze_ratio(right_iris, right_eye_outer, right_eye_inner)
            vertical_iris_shift = abs(left_iris[1] - left_eye_top[1]) + abs(right_iris[1] - right_eye_top[1])
            if (
                left_ear < 0.15 and right_ear < 0.15 or
                abs(left_ratio - 0.5) > 0.25 or abs(right_ratio - 0.5) > 0.25 or
                vertical_iris_shift > 20
            ):
                if time.time() - last_attention_time > attention_timeout:
                    raw_current_state = "Distracted"
            else:
                last_attention_time = time.time()
                raw_current_state = "Attentive"
    except Exception as e:
        print(f"Error in attention detection: {e}")
        # In case of error, don't change state
        return current_smoothed_state
    
    try:
        # Update buffer with the new state (remove oldest, add newest)
        attention_buffer.pop(0)
        attention_buffer.append(raw_current_state)
        
        # Count occurrences of each state in the buffer
        attentive_count = attention_buffer.count("Attentive")
        distracted_count = attention_buffer.count("Distracted")
        
        # Only change state if we have a strong majority
        if current_smoothed_state == "Attentive" and distracted_count >= attention_change_threshold:
            current_smoothed_state = "Distracted"
            print("State changed to: Distracted")
        elif current_smoothed_state == "Distracted" and attentive_count >= attention_change_threshold:
            current_smoothed_state = "Attentive"
            print("State changed to: Attentive")
    except Exception as e:
        print(f"Error in attention state management: {e}")
    
    return current_smoothed_state

def check_distraction(current_attention, distraction_start_time, threshold):
    """
    Check if user has been distracted enough times to trigger a redirect
    Instead of timing continuous distraction, we now count distraction events
    """
    global distraction_counter
    needs_redirect = False
    seconds_distracted = 0  # We'll still return this for UI purposes
    
    try:
        if current_attention == "Distracted":
            # Increment the counter when distracted
            distraction_counter += 1
            
            # For UI purposes, still track how long current distraction has lasted
            if distraction_start_time is None:
                distraction_start_time = time.time()
            else:
                seconds_distracted = int(time.time() - distraction_start_time)
                
            # Check if counter threshold is reached
            if distraction_counter >= DISTRACTION_THRESHOLD:
                needs_redirect = True
                # Reset counter after triggering redirect
                distraction_counter = 0
                
            print(f"Distraction counter: {distraction_counter}/{DISTRACTION_THRESHOLD}")
        else:
            # When attentive, decrease the counter (but not below 0)
            if distraction_counter > 0:
                distraction_counter -= 1
            
            # Reset the distraction timer
            distraction_start_time = None
            
    except Exception as e:
        print(f"Error in distraction checking: {e}")
        
    return needs_redirect, distraction_start_time, seconds_distracted
