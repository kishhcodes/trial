"""
MongoDB integration helper functions for attention_service.py
This module provides adapter functions to bridge the existing attention tracking system with MongoDB
"""
from services import mongo_service
from datetime import datetime
import time
import traceback

def mongo_log_app_startup(version: str, description: str) -> str:
    """Log application startup to MongoDB"""
    try:
        return mongo_service.log_app_startup(version, description)
    except Exception as e:
        print(f"MongoDB startup logging error: {e}")
        return None

def mongo_log_app_shutdown(startup_id: str = None) -> bool:
    """Log application shutdown to MongoDB"""
    try:
        return mongo_service.log_app_shutdown(startup_id)
    except Exception as e:
        print(f"MongoDB shutdown logging error: {e}")
        return False

def mongo_start_user_session(username: str) -> str:
    """Start a MongoDB user session and return the session ID"""
    try:
        return mongo_service.start_user_session(username)
    except Exception as e:
        print(f"MongoDB start session error for {username}: {e}")
        traceback.print_exc()
        return None

def mongo_end_user_session(username: str) -> bool:
    """End a MongoDB user session"""
    try:
        return mongo_service.end_user_session(username)
    except Exception as e:
        print(f"MongoDB end session error for {username}: {e}")
        return False

def mongo_log_distraction_event(
    username: str,
    start_time: datetime,
    end_time: datetime,
    duration_seconds: int,
    threshold_reached: bool = False,
    threshold_used: int = 75,
    adaptation_factor: float = 1.0
) -> str:
    """Log a distraction event to MongoDB"""
    try:
        return mongo_service.log_distraction_event(
            username,
            start_time,
            end_time,
            duration_seconds,
            threshold_reached,
            threshold_used,
            adaptation_factor
        )
    except Exception as e:
        print(f"MongoDB distraction logging error for {username}: {e}")
        return None

def mongo_get_user_sessions(username: str) -> list:
    """Get all sessions for a user from MongoDB"""
    try:
        return mongo_service.get_user_sessions(username)
    except Exception as e:
        print(f"MongoDB get sessions error for {username}: {e}")
        return []

def mongo_get_user_distraction_events(username: str, session_id: str = None) -> list:
    """Get distraction events for a user from MongoDB"""
    try:
        return mongo_service.get_user_distraction_events(username, session_id)
    except Exception as e:
        print(f"MongoDB get events error for {username}: {e}")
        return []
