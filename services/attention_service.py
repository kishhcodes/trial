import cv2
import numpy as np
import mediapipe as mp
import time
import json
import os
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh
# Use static_image_mode=False for better tracking
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Base state tracking
last_attention_time = time.time()
attention_timeout = 3  # Responsive but not too sensitive

# Neurodivergent-friendly parameters - allow more movement and different patterns
# Dynamic threshold that starts stricter but adapts based on user patterns
distraction_counter = 0
INITIAL_THRESHOLD = 75  # ~10-15 seconds at 5-7 FPS
DISTRACTION_THRESHOLD = INITIAL_THRESHOLD
MAX_THRESHOLD = 125  # Upper limit for very active users (~20 seconds)
MIN_THRESHOLD = 50   # Lower limit for users who need more frequent breaks (~8-10 seconds)

# Pattern recognition variables
movement_patterns = []
pattern_window_size = 50
adaptation_factor = 1.0  # Will be adjusted based on detected patterns

# Recovery settings - gentler for neurodivergent users
ATTENTION_RECOVERY_RATE = 4  # Slower recovery - more forgiving of brief movements

# State smoothing with larger buffer for neurodivergent users 
attention_buffer_size = 15  # Increased from 10
attention_buffer = ["Attentive"] * attention_buffer_size
attention_change_threshold = 10  # 66% majority - more stable state changes
current_smoothed_state = "Attentive"  # This variable needs to be defined globally

# Recovery rate counter
recovery_frame_count = 0

# Calibration variables adapted for neurodivergent users
baseline_ear = None
baseline_ear_std = None  # Standard deviation to account for wider range of movements
baseline_gaze_range = None  # Track typical gaze movement range
baseline_samples = []
CALIBRATION_SAMPLES = 45  # Longer calibration to capture more varied behavior
is_calibrated = False
calibration_message = "Getting to know you... keep looking at the screen naturally"
calibration_start_time = None

# Break timing variables - adaptive based on usage patterns
last_break_time = time.time()
break_frequency_minutes = 15  # Default - will adapt
break_duration_seconds = 120  # Default - will adapt
attention_span_history = []

# Make sure frame_count is properly initialized and incremented
frame_count = 0

# Add logging configuration
LOGS_DIR = "/home/krizz/projects/krizzip/logs"
DISTRACTION_LOG_FILE = os.path.join(LOGS_DIR, "distraction_log.json")
USER_LOGS_DIR = os.path.join(LOGS_DIR, "users")

# Create logs directories
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(USER_LOGS_DIR, exist_ok=True)

# Enhanced tracking variables
active_user_sessions = {}  # Track separate user sessions
session_start_times = {}   # When each user started their session

# Initialize or load existing log data
def init_distraction_log():
    """Initialize or load the distraction log file"""
    if os.path.exists(DISTRACTION_LOG_FILE):
        try:
            with open(DISTRACTION_LOG_FILE, 'r') as f:
                log_data = json.load(f)
                print(f"Loaded existing distraction log with {len(log_data['sessions'])} sessions")
                
                # Add users field if it doesn't exist in older logs
                if 'users' not in log_data:
                    log_data['users'] = {}
                    
                # Ensure all sessions have user_events field
                for session in log_data.get('sessions', []):
                    if 'user_events' not in session:
                        session['user_events'] = {}
                        
                # Add user_sessions field if not present
                if 'user_sessions' not in log_data:
                    log_data['user_sessions'] = {}
                
                return log_data
        except Exception as e:
            print(f"Error loading distraction log: {e}, creating new log")
    
    # Create new log structure if file doesn't exist or has errors
    log_data = {
        "sessions": [],
        "users": {},  # Track per-user distraction data
        "user_sessions": {},  # Track per-user sessions
        "last_updated": datetime.now().isoformat()
    }
    
    # Start a new session
    new_session = {
        "start_time": datetime.now().isoformat(),
        "distraction_events": [],
        "user_events": {}  # Maps username -> list of event indices
    }
    log_data["sessions"].append(new_session)
    
    # Save initial log
    save_distraction_log(log_data)
    print("Created new distraction log file with user tracking")
    
    return log_data

# Save log data to file
def save_distraction_log(log_data):
    """Save distraction log data to JSON file"""
    log_data["last_updated"] = datetime.now().isoformat()
    try:
        with open(DISTRACTION_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving distraction log: {e}")
        return False

# Get or create user profile
def get_user_profile(username):
    """Get or create a user's distraction profile"""
    global distraction_log
    
    users = distraction_log.get('users', {})
    
    # Create user if doesn't exist
    if username not in users:
        users[username] = {
            "first_seen": datetime.now().isoformat(),
            "total_distraction_events": 0,
            "total_distraction_seconds": 0,
            "average_distraction_duration": 0,
            "breaks_taken": 0,
            "last_session": datetime.now().isoformat()
        }
        distraction_log['users'] = users
        save_distraction_log(distraction_log)
        print(f"Created new user profile for '{username}'")
    
    return users[username]

# Update user statistics
def update_user_stats(username, distraction_seconds=0, break_taken=False):
    """Update a user's distraction statistics"""
    if not username or username == "Unknown":
        return
        
    user = get_user_profile(username)
    
    if distraction_seconds > 0:
        # Update distraction statistics
        user["total_distraction_events"] += 1
        user["total_distraction_seconds"] += distraction_seconds
        user["average_distraction_duration"] = (
            user["total_distraction_seconds"] / user["total_distraction_events"]
        )
    
    if break_taken:
        user["breaks_taken"] += 1
        
    user["last_session"] = datetime.now().isoformat()
    
    # Save the updated profile
    distraction_log['users'][username] = user
    save_distraction_log(distraction_log)

# Load or initialize log on module import
distraction_log = init_distraction_log()
current_session = distraction_log["sessions"][-1]
last_log_save_time = time.time()

def calibrate_eye_metrics(left_ear, right_ear, left_ratio, right_ratio, vertical_shift):
    """Calibrate eye metrics with neurodivergent-friendly approach"""
    global baseline_ear, baseline_ear_std, baseline_gaze_range, baseline_samples, is_calibrated
    global calibration_message, calibration_start_time, adaptation_factor
    
    # Initialize calibration start time
    if calibration_start_time is None:
        calibration_start_time = time.time()
        print("Starting calibration process - please look naturally at the screen")
    
    # Collect comprehensive metrics including movement variations
    if len(baseline_samples) < CALIBRATION_SAMPLES:
        baseline_samples.append((left_ear, right_ear, left_ratio, right_ratio, vertical_shift))
        progress = int((len(baseline_samples) / CALIBRATION_SAMPLES) * 100)
        
        # More encouraging messages for neurodivergent users
        if progress < 33:
            calibration_message = f"Getting to know you... {progress}%"
        elif progress < 66:
            calibration_message = f"You're doing great! {progress}%"
        else:
            calibration_message = f"Almost there! {progress}%"
            
        return False
    
    elif not is_calibrated:
        # Calculate baseline values with statistics for better adaptation
        ears = [(s[0] + s[1])/2 for s in baseline_samples]
        gaze_variations = [abs(s[2] - 0.5) + abs(s[3] - 0.5) for s in baseline_samples]
        vertical_movements = [s[4] for s in baseline_samples]
        
        # Calculate mean and standard deviation for more nuanced thresholds
        baseline_ear = sum(ears) / len(ears)
        baseline_ear_std = np.std(ears)
        
        # Calculate typical gaze movement range
        baseline_gaze_range = np.percentile(gaze_variations, 80)
        
        # Set adaptation factor based on movement patterns
        movement_variability = np.std(vertical_movements)
        if movement_variability > 8:  # High variability
            adaptation_factor = 1.3  # More lenient
            print("Detected high movement variability - adapting thresholds")
        elif movement_variability < 3:  # Low variability
            adaptation_factor = 0.9  # More strict
        else:
            adaptation_factor = 1.0  # Default
            
        # Adjust thresholds based on calibration
        global DISTRACTION_THRESHOLD
        DISTRACTION_THRESHOLD = int(INITIAL_THRESHOLD * adaptation_factor)
        DISTRACTION_THRESHOLD = max(MIN_THRESHOLD, min(MAX_THRESHOLD, DISTRACTION_THRESHOLD))
        
        # Log calibration results
        print(f"✅ Eye tracking calibrated for neurodivergent patterns:")
        print(f"   Baseline EAR: {baseline_ear:.4f} ± {baseline_ear_std:.4f}")
        print(f"   Gaze variation range: {baseline_gaze_range:.4f}")
        print(f"   Adaptation factor: {adaptation_factor:.2f}")
        print(f"   Adjusted threshold: {DISTRACTION_THRESHOLD} frames")
        
        calibration_message = "All set! Attention tracking ready"
        is_calibrated = True
        return True
    
    return True

def detect_attention(frame):
    """Detect attention with sensitivity to neurodivergent attention patterns"""
    global last_attention_time, attention_buffer, current_smoothed_state, is_calibrated
    global movement_patterns, adaptation_factor, DISTRACTION_THRESHOLD, frame_count
    
    # Increment frame counter each time this function is called
    frame_count += 1
    
    # Detect raw attention state for this frame
    raw_current_state = "Distracted"  # Default state
    
    try:
        # Convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            
            # Get key points for attention tracking
            def get_point(idx):
                return int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                
            # Eye landmarks
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
            
            # Calculate eye aspect ratio (height/width)
            def eye_aspect_ratio(top, bottom, left, right):
                vertical = np.linalg.norm(np.array(top) - np.array(bottom))
                horizontal = np.linalg.norm(np.array(left) - np.array(right))
                return vertical / horizontal if horizontal != 0 else 0
                
            # Eye aspect ratios
            left_ear = eye_aspect_ratio(left_eye_top, left_eye_bottom, left_eye_outer, left_eye_inner)
            right_ear = eye_aspect_ratio(right_eye_top, right_eye_bottom, right_eye_outer, right_eye_inner)
            
            # Gaze detection - where is iris relative to eye corners
            def gaze_ratio(iris, outer, inner):
                iris_to_inner = np.linalg.norm(np.array(iris) - np.array(inner))
                outer_to_inner = np.linalg.norm(np.array(outer) - np.array(inner))
                return iris_to_inner / (outer_to_inner + 1e-6)
                
            left_ratio = gaze_ratio(left_iris, left_eye_outer, left_eye_inner)
            right_ratio = gaze_ratio(right_iris, right_eye_outer, right_eye_inner)
            
            # Vertical gaze - how high/low is iris in eye socket
            vertical_iris_shift = abs(left_iris[1] - left_eye_top[1]) + abs(right_iris[1] - right_eye_top[1])
            
            # Dynamically adjust thresholds based on calibration
            if not is_calibrated:
                # During calibration, assume attentive with encouraging feedback
                calibrated = calibrate_eye_metrics(left_ear, right_ear, left_ratio, right_ratio, vertical_iris_shift)
                if not calibrated:
                    # Still calibrating - show friendly message
                    raw_current_state = "Attentive"
                    
                    # Add calibration message with friendly, colorful style
                    cv2.putText(frame, calibration_message, (20, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)
                    
                    # Add progress bar
                    progress = len(baseline_samples) / CALIBRATION_SAMPLES
                    bar_width = int(200 * progress)
                    cv2.rectangle(frame, (20, 100), (20 + bar_width, 115), (50, 200, 50), -1)
                    cv2.rectangle(frame, (20, 100), (220, 115), (200, 200, 200), 2)
                    
                    return "Attentive"  # Special case during calibration
            
            # Record movement pattern for adaptation
            if len(movement_patterns) >= pattern_window_size:
                movement_patterns.pop(0)
            movement_patterns.append((left_ear, right_ear, left_ratio, right_ratio, vertical_iris_shift))
            
            # Neurodivergent-friendly thresholds using calibrated baselines and patterns
            if baseline_ear and baseline_ear_std:
                # More personalized thresholds based on the user's natural patterns
                # Lower threshold means eyes are more closed than their baseline
                ear_threshold = max(0.1, baseline_ear - (1.5 * baseline_ear_std * adaptation_factor))
                
                # Gaze threshold based on their natural movement patterns
                gaze_threshold = min(0.4, baseline_gaze_range * 1.3)
                
                # Vertical threshold based on observed patterns
                vertical_threshold = 22 * adaptation_factor  # Higher for users with more movement
            else:
                # Default fallback thresholds
                ear_threshold = 0.18
                gaze_threshold = 0.25
                vertical_threshold = 20
            
            # Check for signs of distraction with personalized thresholds:
            if (
                (left_ear < ear_threshold and right_ear < ear_threshold) or  # Eyes too closed
                abs(left_ratio - 0.5) > gaze_threshold or abs(right_ratio - 0.5) > gaze_threshold or  # Looking away
                vertical_iris_shift > vertical_threshold  # Looking up/down
            ):
                if time.time() - last_attention_time > attention_timeout:
                    raw_current_state = "Distracted"
            else:
                # Actively looking at screen
                last_attention_time = time.time()
                raw_current_state = "Attentive"
                
            # Periodically update adaptation factor based on patterns (every 100 frames)
            if is_calibrated and len(movement_patterns) == pattern_window_size and time.time() % 100 < 1:
                # Calculate recent movement variability
                recent_movements = [m[4] for m in movement_patterns[-20:]]  # Last 20 samples
                recent_variability = np.std(recent_movements)
                
                # Adjust adaptation factor if significant change in patterns
                if recent_variability > 10 and adaptation_factor < 1.3:
                    adaptation_factor += 0.05
                    DISTRACTION_THRESHOLD = min(MAX_THRESHOLD, int(DISTRACTION_THRESHOLD * 1.1))
                    print(f"⚙️ Adapting to higher movement patterns: {adaptation_factor:.2f}")
                elif recent_variability < 5 and adaptation_factor > 0.9:
                    adaptation_factor -= 0.05
                    DISTRACTION_THRESHOLD = max(MIN_THRESHOLD, int(DISTRACTION_THRESHOLD * 0.95))
                    print(f"⚙️ Adapting to calmer movement patterns: {adaptation_factor:.2f}")
    except Exception as e:
        # More tolerant error handling
        raw_current_state = "Distracted"
    
    # Update buffer with new raw state
    attention_buffer.pop(0)
    attention_buffer.append(raw_current_state)
    
    # Count states in buffer
    attentive_count = attention_buffer.count("Attentive")
    distracted_count = attention_buffer.count("Distracted")
    
    # Only change overall state with strong majority
    if current_smoothed_state == "Attentive" and distracted_count >= attention_change_threshold:
        current_smoothed_state = "Distracted"
        print("⚠️ State changed to: Distracted")
    elif current_smoothed_state == "Distracted" and attentive_count >= attention_change_threshold:
        current_smoothed_state = "Attentive"
        print("✅ State changed to: Attentive")
    
    return current_smoothed_state

# Adjust the distraction counter logic to be more stable
def check_distraction(current_attention, distraction_start_time, threshold, current_user="Unknown"):
    """
    Check if user has been distracted with neurodivergent-friendly timing logic
    More adaptive to different attention patterns
    Now tracks distraction per user session
    """
    try:
        global distraction_counter, recovery_frame_count, frame_count
        global last_break_time, break_frequency_minutes, distraction_log, current_session
        global last_log_save_time
        
        needs_redirect = False
        seconds_distracted = 0
        current_time = time.time()
        
        # Fix: Create distraction_start_time_dt for logging
        distraction_start_time_dt = (datetime.fromtimestamp(distraction_start_time) 
                                    if distraction_start_time is not None else None)
        
        # If current_session doesn't have user_events, add it
        if 'user_events' not in current_session:
            current_session['user_events'] = {}
        
        # If this is a new user session, initialize it
        if current_user != "Unknown" and current_user not in active_user_sessions:
            start_user_session(current_user)
        
        # Debug the current state - add counter vs threshold info
        print(f"Attention: {current_attention}, Counter: {distraction_counter}/{DISTRACTION_THRESHOLD}")
        
        if current_attention == "Distracted":
            # Don't increment too quickly - only every 3rd frame
            if frame_count % 3 == 0:
                old_counter = distraction_counter
                distraction_counter += 1
                
                # Log every 5th count or when close to threshold
                remaining = DISTRACTION_THRESHOLD - distraction_counter
                if (distraction_counter % 5 == 0) or (0 < remaining < 10):
                    print(f"⬆️ Counter: {old_counter} → {distraction_counter}, Threshold: {DISTRACTION_THRESHOLD}, Remaining: {remaining}")
            
            # Track timing for UI
            if distraction_start_time is None:
                distraction_start_time = current_time
                # Log the start of a distraction event
                event_data = {
                    "start_time": datetime.fromtimestamp(current_time).isoformat(),
                    "end_time": None,
                    "duration_seconds": None,
                    "threshold_used": DISTRACTION_THRESHOLD,
                    "threshold_reached": False,
                    "adaptation_factor": adaptation_factor,
                    "user": current_user
                }
                current_session["distraction_events"].append(event_data)
                
                # Add to user-specific tracking
                if current_user != "Unknown":
                    event_index = len(current_session["distraction_events"]) - 1
                    if current_user not in current_session["user_events"]:
                        current_session["user_events"][current_user] = []
                    current_session["user_events"][current_user].append(event_index)
                    
            else:
                seconds_distracted = int(current_time - distraction_start_time)
                
                # Update the latest distraction event
                if current_session["distraction_events"]:
                    current_event = current_session["distraction_events"][-1]
                    current_event["duration_seconds"] = seconds_distracted
                    # Also update the user name in case recognition improved mid-event
                    if current_event["user"] == "Unknown" and current_user != "Unknown":
                        current_event["user"] = current_user
                        # Add to user-specific tracking if not already there
                        event_index = len(current_session["distraction_events"]) - 1
                        if current_user not in current_session["user_events"]:
                            current_session["user_events"][current_user] = []
                        if event_index not in current_session["user_events"][current_user]:
                            current_session["user_events"][current_user].append(event_index)
            
            # Check threshold and trigger redirect if needed
            if distraction_counter >= DISTRACTION_THRESHOLD:
                needs_redirect = True
                print(f"🚨 THRESHOLD REACHED! {distraction_counter} >= {DISTRACTION_THRESHOLD}")
                last_break_time = current_time
                
                # Update log that threshold was reached
                if current_session["distraction_events"]:
                    current_event = current_session["distraction_events"][-1]
                    current_event["threshold_reached"] = True
                    current_event["counter_value"] = distraction_counter
                    
                    # Also log to user session with enhanced data
                    if current_user != "Unknown":
                        log_distraction_event(
                            current_user, 
                            distraction_start_time_dt if 'distraction_start_time_dt' in locals() else datetime.fromtimestamp(distraction_start_time),
                            datetime.fromtimestamp(current_time),
                            seconds_distracted,
                            True  # threshold reached
                        )
                
                # Update user statistics
                if current_user != "Unknown":
                    update_user_stats(current_user, 
                                     distraction_seconds=seconds_distracted, 
                                     break_taken=True)
                
                print(f"🚨 BREAK TIME for {current_user}! Counter: {distraction_counter}")
        else:
            # When attentive, decrease counter but log more clearly
            if recovery_frame_count >= ATTENTION_RECOVERY_RATE and distraction_counter > 0:
                old_counter = distraction_counter
                distraction_counter = max(0, distraction_counter - 1)
                recovery_frame_count = 0
                print(f"⬇️ Counter decreased: {old_counter} → {distraction_counter}")
            else:
                recovery_frame_count += 1
                
            # When attentive, finalize any active distraction event
            if distraction_start_time is not None:
                # End the current distraction event
                if current_session["distraction_events"]:
                    current_event = current_session["distraction_events"][-1]
                    if current_event["end_time"] is None:  # Only update if not already ended
                        current_event["end_time"] = datetime.fromtimestamp(current_time).isoformat()
                        seconds_distracted = int(current_time - distraction_start_time)
                        current_event["duration_seconds"] = seconds_distracted
                        
                        # Also log to user session
                        if current_user != "Unknown":
                            log_distraction_event(
                                current_user,
                                distraction_start_time_dt if 'distraction_start_time_dt' in locals() else datetime.fromtimestamp(distraction_start_time),
                                datetime.fromtimestamp(current_time),
                                seconds_distracted
                            )
                        
                        # Update user statistics
                        user = current_event.get("user", "Unknown")
                        if user != "Unknown":
                            update_user_stats(user, distraction_seconds=seconds_distracted)
        
        # Periodic log
        if distraction_counter > 0 and distraction_counter % 10 == 0:
            print(f"Distraction counter: {distraction_counter}/{DISTRACTION_THRESHOLD}")
            
        # Save log periodically (every 30 seconds) to avoid losing data
        if time.time() - last_log_save_time > 30:
            save_distraction_log(distraction_log)
            last_log_save_time = time.time()
        
        return needs_redirect, distraction_start_time, seconds_distracted
    except Exception as e:
        print(f"Error in distraction check: {e}")
        import traceback
        traceback.print_exc()
        return False, distraction_start_time, 0

# Fix: Make reset more thorough
def reset_distraction_counter():
    """Reset the distraction counter after successful redirect"""
    global distraction_counter, recovery_frame_count, distraction_start_time
    old_value = distraction_counter
    
    # Reset all distraction tracking variables
    distraction_counter = 0
    recovery_frame_count = 0
    distraction_start_time = None
    
    print(f"✅ Distraction counter FULLY RESET from {old_value} to 0")
    return old_value

# Add user-specific log file path generation
def get_user_log_path(username):
    """Get the path to a user's specific log file"""
    if not username or username == "Unknown":
        return None
    # Create a safe filename from username
    safe_username = "".join(c if c.isalnum() or c in "._- " else "_" for c in username)
    safe_username = safe_username.replace(" ", "_")
    return os.path.join(USER_LOGS_DIR, f"{safe_username}_log.json")

# Initialize or load existing user log data
def init_user_log(username):
    """Initialize or load a specific user's log file"""
    if not username or username == "Unknown":
        return None
        
    log_path = get_user_log_path(username)
    if not log_path:
        return None
        
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                user_log = json.load(f)
                print(f"Loaded existing log for user {username} with {len(user_log.get('sessions', []))} sessions")
                return user_log
        except Exception as e:
            print(f"Error loading user log for {username}: {e}, creating new log")
    
    # Create new user log structure
    user_log = {
        "username": username,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "sessions": [],
        "stats": {
            "total_sessions": 0,
            "total_distraction_events": 0,
            "total_distraction_seconds": 0,
            "average_distraction_duration": 0,
            "breaks_taken": 0
        }
    }
    
    # Save initial log
    save_user_log(username, user_log)
    print(f"Created new log file for user: {username}")
    
    return user_log

def save_user_log(username, user_log):
    """Save a specific user's log data to file"""
    if not username or username == "Unknown" or not user_log:
        return False
        
    log_path = get_user_log_path(username)
    if not log_path:
        return False
        
    user_log["last_updated"] = datetime.now().isoformat()
    try:
        with open(log_path, 'w') as f:
            json.dump(user_log, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving user log for {username}: {e}")
        return False

# Track loaded user logs
user_logs = {}

def get_user_log(username, force_reload=False):
    """Get a user's log data, loading it if necessary"""
    global user_logs
    
    if not username or username == "Unknown":
        return None
        
    if username not in user_logs or force_reload:
        user_logs[username] = init_user_log(username)
    
    return user_logs[username]

def start_user_session(username):
    """Start a new tracking session for a specific user"""
    global active_user_sessions, session_start_times, distraction_log
    
    # Skip for unknown users
    if not username or username == "Unknown":
        return False
    
    # If user already has an active session, end it first
    if username in active_user_sessions:
        end_user_session(username)
    
    # Start a new session
    now = datetime.now()
    session_id = f"{username}_{now.strftime('%Y%m%d_%H%M%S')}"
    
    # Create session data
    session = {
        "session_id": session_id,
        "start_time": now.isoformat(),
        "distraction_events": [],
        "total_distraction_time": 0,
        "breaks_taken": 0
    }
    
    # Store in active sessions
    active_user_sessions[username] = session
    session_start_times[username] = now
    
    # Also add to main log
    if username not in distraction_log['user_sessions']:
        distraction_log['user_sessions'][username] = []
    
    # Get user-specific log and add session
    user_log = get_user_log(username)
    if user_log:
        user_log["sessions"].append(session.copy())
        user_log["stats"]["total_sessions"] += 1
        save_user_log(username, user_log)
    
    # Save main log too
    save_distraction_log(distraction_log)
    
    print(f"📊 Started tracking session for user: {username}")
    return True

def end_user_session(username):
    """End tracking session for a user and save session data"""
    global active_user_sessions, session_start_times, distraction_log
    
    # Skip if no active session
    if not username or username not in active_user_sessions:
        return False
    
    # Finalize session data
    session = active_user_sessions[username]
    end_time = datetime.now()
    
    # Add end time and calculate duration
    session["end_time"] = end_time.isoformat()
    
    if username in session_start_times:
        start_time = session_start_times[username]
        duration_seconds = (end_time - start_time).total_seconds()
        session["duration_seconds"] = duration_seconds
    
    # Finalize any incomplete distraction events
    for event in session["distraction_events"]:
        if "end_time" not in event or not event["end_time"]:
            event["end_time"] = end_time.isoformat()
            if "start_time" in event:
                try:
                    start_time = datetime.fromisoformat(event["start_time"])
                    event["duration_seconds"] = int((end_time - start_time).total_seconds())
                except:
                    event["duration_seconds"] = 0
    
    # Add to main log's user sessions
    distraction_log['user_sessions'][username].append(session.copy())
    
    # Update main log's user profile stats
    if username in distraction_log['users']:
        user_profile = distraction_log['users'][username]
        user_profile["sessions_count"] = user_profile.get("sessions_count", 0) + 1
        user_profile["total_session_time"] = user_profile.get("total_session_time", 0) + session.get("duration_seconds", 0)
        user_profile["last_session"] = end_time.isoformat()
    
    # Update user-specific log
    user_log = get_user_log(username)
    if user_log:
        # Find the matching session and update it
        session_updated = False
        for i, s in enumerate(user_log["sessions"]):
            if s.get("session_id") == session.get("session_id"):
                user_log["sessions"][i] = session.copy()
                session_updated = True
                break
                
        # If not found, append it
        if not session_updated:
            user_log["sessions"].append(session.copy())
            
        # Update user stats
        total_distraction_seconds = sum(
            event.get("duration_seconds", 0) for s in user_log["sessions"] 
            for event in s.get("distraction_events", [])
        )
        total_events = sum(len(s.get("distraction_events", [])) for s in user_log["sessions"])
        total_breaks = sum(s.get("breaks_taken", 0) for s in user_log["sessions"])
        
        user_log["stats"].update({
            "total_distraction_events": total_events,
            "total_distraction_seconds": total_distraction_seconds,
            "average_distraction_duration": (
                total_distraction_seconds / total_events if total_events > 0 else 0
            ),
            "breaks_taken": total_breaks
        })
        
        # Save the user log
        save_user_log(username, user_log)
    
    # Save main log too
    save_distraction_log(distraction_log)
    
    # Clean up
    del active_user_sessions[username]
    if username in session_start_times:
        del session_start_times[username]
    
    print(f"📊 Ended tracking session for user: {username}")
    return True

def log_distraction_event(current_user, start_time, end_time, duration_seconds, threshold_reached=False):
    """Log a distraction event to the user's session"""
    global active_user_sessions
    
    if not current_user or current_user == "Unknown" or current_user not in active_user_sessions:
        return False
    
    # Create event data
    event = {
        "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
        "end_time": end_time.isoformat() if isinstance(end_time, datetime) and end_time else None,
        "duration_seconds": duration_seconds,
        "threshold_used": DISTRACTION_THRESHOLD,
        "threshold_reached": threshold_reached,
        "adaptation_factor": adaptation_factor
    }
    
    # Add to active session
    active_user_sessions[current_user]["distraction_events"].append(event)
    active_user_sessions[current_user]["total_distraction_time"] += duration_seconds if duration_seconds else 0
    
    if threshold_reached:
        active_user_sessions[current_user]["breaks_taken"] += 1
    
    # Also update the user-specific log file
    user_log = get_user_log(current_user)
    if user_log:
        # Find the current session
        session_id = active_user_sessions[current_user].get("session_id")
        for i, session in enumerate(user_log["sessions"]):
            if session.get("session_id") == session_id:
                # Add event to this session
                user_log["sessions"][i]["distraction_events"].append(event.copy())
                user_log["sessions"][i]["total_distraction_time"] += duration_seconds if duration_seconds else 0
                
                if threshold_reached:
                    user_log["sessions"][i]["breaks_taken"] = user_log["sessions"][i].get("breaks_taken", 0) + 1
                
                # Update user stats
                user_log["stats"]["total_distraction_events"] += 1
                user_log["stats"]["total_distraction_seconds"] += duration_seconds if duration_seconds else 0
                
                if user_log["stats"]["total_distraction_events"] > 0:
                    user_log["stats"]["average_distraction_duration"] = (
                        user_log["stats"]["total_distraction_seconds"] / 
                        user_log["stats"]["total_distraction_events"]
                    )
                    
                if threshold_reached:
                    user_log["stats"]["breaks_taken"] += 1
                
                # Save the updated log
                save_user_log(current_user, user_log)
                break
    
    return True

def get_calibration_status():
    """Get calibration status for display"""
    if not is_calibrated:
        progress = len(baseline_samples) / CALIBRATION_SAMPLES * 100
        return {
            "is_calibrated": False,
            "progress": progress,
            "message": calibration_message
        }
    else:
        return {
            "is_calibrated": True,
            "adaptation_factor": adaptation_factor,
            "threshold": DISTRACTION_THRESHOLD,
            "message": "Calibration complete"
        }
