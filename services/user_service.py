import time

# User session state
current_user = {
    "name": "Unknown",
    "authenticated": False,
    "timestamp": time.time(),
    "consecutive_detections": 0
}

# Authentication settings
AUTHENTICATION_THRESHOLD = 5  # Reduced threshold to make authentication faster
authentication_in_progress = False

# Add a flag to track if we're on the main monitoring page
on_monitoring_page = False

# Add locked user tracking for monitoring mode
locked_monitoring_user = None

# Move this function to avoid circular import
def get_current_user_from_frame(frame):
    """Get the current recognized user from a frame - separate from face_service import"""
    # Avoid circular import by importing here
    from services.face_service import get_current_user
    return get_current_user(frame)

def check_user_authentication(frame):
    """
    Check if a known user is present and authenticate them.
    This is completely separate from attention tracking.
    """
    global current_user, authentication_in_progress, locked_monitoring_user, on_monitoring_page
    
    # If we're on monitoring page and have a locked user, use that instead of detection
    if on_monitoring_page and locked_monitoring_user:
        return True, locked_monitoring_user
    
    # Try to authenticate a user
    try:
        # Use our new function to avoid circular import
        detected_user = get_current_user_from_frame(frame)
        
        # Debug output
        if detected_user != "Unknown":
            print(f"Face recognition: {detected_user}, Auth progress: {current_user['consecutive_detections']}/{AUTHENTICATION_THRESHOLD}")
        
        # If we detect a known user
        if detected_user != "Unknown":
            # Same user as we've been tracking
            if detected_user == current_user["name"]:
                current_user["consecutive_detections"] += 1
                
                # If we've seen the same user enough times, authenticate them
                if current_user["consecutive_detections"] >= AUTHENTICATION_THRESHOLD:
                    if not current_user["authenticated"]:
                        print(f"🔐 User {detected_user} authenticated!")
                    current_user["authenticated"] = True
                    current_user["timestamp"] = time.time()
                    authentication_in_progress = False
                else:
                    authentication_in_progress = True
            # Different user, start over
            else:
                if current_user["name"] != "Unknown":
                    print(f"👤 Detected new user: {detected_user}")
                current_user["name"] = detected_user
                current_user["consecutive_detections"] = 1
                current_user["timestamp"] = time.time()
                current_user["authenticated"] = False
                authentication_in_progress = True
        # No known user detected
        else:
            # Reset progress if we lose the user
            if current_user["consecutive_detections"] > 0:
                current_user["consecutive_detections"] = max(0, current_user["consecutive_detections"] - 1)
                
            # If we completely lost track of the user, reset authentication
            if current_user["consecutive_detections"] == 0:
                if current_user["name"] != "Unknown":
                    print("❌ Lost track of user, resetting authentication")
                current_user["name"] = "Unknown"
                current_user["authenticated"] = False
                authentication_in_progress = False
    
    except Exception as e:
        print(f"Error in user authentication: {e}")
    
    return current_user["authenticated"], current_user["name"]

def get_authentication_status():
    """Get the current authentication status"""
    global current_user, authentication_in_progress
    return {
        "authenticated": current_user["authenticated"],
        "user": current_user["name"],
        "authentication_in_progress": authentication_in_progress,
        "detection_progress": current_user["consecutive_detections"],
        "detection_needed": AUTHENTICATION_THRESHOLD
    }

def reset_authentication():
    """Reset the authentication state"""
    global current_user, authentication_in_progress
    current_user = {
        "name": "Unknown",
        "authenticated": False,
        "timestamp": time.time(),
        "consecutive_detections": 0
    }
    authentication_in_progress = False
    print("🔄 Authentication reset")

def set_monitoring_active(active=True):
    """Set whether we're on the main monitoring page"""
    global on_monitoring_page, locked_monitoring_user, current_user
    
    on_monitoring_page = active
    
    if active and current_user["authenticated"]:
        # Lock in the current user when entering monitoring mode
        locked_monitoring_user = current_user["name"]
        print(f"Monitoring page active - locked user: {locked_monitoring_user}")
        
        # Start a session for this user - use a try/except to avoid circular imports
        try:
            from services.attention_service import start_user_session
            start_user_session(locked_monitoring_user)
        except Exception as e:
            print(f"Error starting user session: {e}")
    elif not active:
        # End the session when leaving monitoring mode
        if locked_monitoring_user:
            try:
                from services.attention_service import end_user_session
                end_user_session(locked_monitoring_user)
            except Exception as e:
                print(f"Error ending user session: {e}")
            
        # Reset locked user
        locked_monitoring_user = None
        print("Monitoring page inactive - user unlocked")
    
    print(f"Monitoring page {'active' if active else 'inactive'}")

def is_monitoring_active():
    """Check if we're on the main monitoring page"""
    global on_monitoring_page
    return on_monitoring_page
    
def get_locked_user():
    """Get the currently locked user for monitoring"""
    global locked_monitoring_user
    return locked_monitoring_user
