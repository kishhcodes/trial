"""
MongoDB integration for attention_service.py

This patch module enhances the existing attention tracking functions 
with MongoDB logging while maintaining the JSON file logging for backward compatibility.
"""

# Import standard MongoDB adapter functions
from services.mongo_adapter import (
    mongo_start_user_session,
    mongo_end_user_session,
    mongo_log_distraction_event
)

# Store MongoDB session IDs
mongo_session_ids = {}

# Enhanced functions that wrap the existing functionality

def enhanced_start_user_session(original_func, username):
    """
    Enhance the existing start_user_session function with MongoDB logging
    
    Args:
        original_func: The original start_user_session function
        username: The username to start a session for
        
    Returns:
        The result of the original function
    """
    global mongo_session_ids
    
    # Run the original function first
    result = original_func(username)
    
    # Then add MongoDB logging
    if result and username and username != "Unknown":
        mongo_session_id = mongo_start_user_session(username)
        if mongo_session_id:
            mongo_session_ids[username] = mongo_session_id
            print(f"Started MongoDB session for user {username}")
    
    return result

def enhanced_end_user_session(original_func, username):
    """
    Enhance the existing end_user_session function with MongoDB logging
    
    Args:
        original_func: The original end_user_session function
        username: The username to end a session for
        
    Returns:
        The result of the original function
    """
    global mongo_session_ids
    
    # Run the original function first
    result = original_func(username)
    
    # Then add MongoDB logging
    if username and username != "Unknown":
        if mongo_end_user_session(username):
            if username in mongo_session_ids:
                del mongo_session_ids[username]
            print(f"Ended MongoDB session for user {username}")
    
    return result

def enhanced_log_distraction_event(original_func, current_user, start_time, end_time, duration_seconds, threshold_reached=False):
    """
    Enhance the existing log_distraction_event function with MongoDB logging
    
    Args:
        original_func: The original log_distraction_event function
        current_user: The username to log an event for
        start_time: The start time of the distraction
        end_time: The end time of the distraction
        duration_seconds: The duration of the distraction in seconds
        threshold_reached: Whether the distraction threshold was reached
        
    Returns:
        The result of the original function
    """
    from services.attention_service import DISTRACTION_THRESHOLD, adaptation_factor
    
    # Run the original function first
    result = original_func(current_user, start_time, end_time, duration_seconds, threshold_reached)
    
    # Then add MongoDB logging
    if result and current_user and current_user != "Unknown":
        mongo_log_distraction_event(
            current_user,
            start_time,
            end_time,
            duration_seconds,
            threshold_reached,
            DISTRACTION_THRESHOLD,
            adaptation_factor
        )
    
    return result

# Patch function to apply the enhancements
def apply_mongodb_patches():
    """Apply MongoDB enhancements to the existing attention service functions"""
    import services.attention_service as attention_service
    
    # Store original functions
    original_start_session = attention_service.start_user_session
    original_end_session = attention_service.end_user_session
    original_log_distraction = attention_service.log_distraction_event
    
    # Replace with enhanced versions
    def patched_start_session(username):
        return enhanced_start_user_session(original_start_session, username)
    
    def patched_end_session(username):
        return enhanced_end_user_session(original_end_session, username)
    
    def patched_log_distraction(current_user, start_time, end_time, duration_seconds, threshold_reached=False):
        return enhanced_log_distraction_event(
            original_log_distraction, 
            current_user, 
            start_time, 
            end_time, 
            duration_seconds, 
            threshold_reached
        )
    
    # Apply patches
    attention_service.start_user_session = patched_start_session
    attention_service.end_user_session = patched_end_session
    attention_service.log_distraction_event = patched_log_distraction
    
    print("MongoDB integration applied to attention service")
