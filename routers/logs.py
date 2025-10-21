from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import json
from typing import Optional
from services.attention_service import DISTRACTION_LOG_FILE, USER_LOGS_DIR
# Import MongoDB adapters
from services.mongo_adapter import (
    mongo_get_user_sessions,
    mongo_get_user_distraction_events
)

router = APIRouter(
    prefix="/api",
    tags=["Logs"]
)

@router.get("/distraction_log")
async def get_distraction_log(user: Optional[str] = None):
    """Get the distraction log data, optionally filtered by user"""
    try:
        if os.path.exists(DISTRACTION_LOG_FILE):
            with open(DISTRACTION_LOG_FILE, 'r') as f:
                log_data = json.load(f)
                
                # If user specified, filter the data
                if user:
                    # Get user profile
                    if user in log_data.get('users', {}):
                        user_data = log_data['users'][user]
                        
                        # Get user-specific events
                        user_events = []
                        for session in log_data.get('sessions', []):
                            session_user_events = []
                            for idx in session.get('user_events', {}).get(user, []):
                                if idx < len(session.get('distraction_events', [])):
                                    session_user_events.append(session['distraction_events'][idx])
                            
                            if session_user_events:
                                user_events.extend(session_user_events)
                        
                        # Include user sessions if they exist
                        user_sessions = log_data.get('user_sessions', {}).get(user, [])
                        
                        # Return user-specific data
                        return {
                            "user": user,
                            "profile": user_data,
                            "events": user_events,
                            "sessions": user_sessions
                        }
                    else:
                        return {"error": f"User '{user}' not found"}
                
                # Return all data if no user specified
                return log_data
        else:
            return {"error": "Log file not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

@router.get("/users")
async def get_users():
    """Get list of all users with distraction profiles"""
    try:
        if os.path.exists(DISTRACTION_LOG_FILE):
            with open(DISTRACTION_LOG_FILE, 'r') as f:
                log_data = json.load(f)
                users = log_data.get('users', {})
                return {"users": list(users.keys())}
        return {"users": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

@router.get("/user_sessions/{username}")
async def get_user_sessions(username: str):
    """Get session history for a specific user"""
    try:
        if os.path.exists(DISTRACTION_LOG_FILE):
            with open(DISTRACTION_LOG_FILE, 'r') as f:
                log_data = json.load(f)
                
                # Get user sessions
                if username in log_data.get('user_sessions', {}):
                    sessions = log_data['user_sessions'][username]
                    user_profile = log_data.get('users', {}).get(username, {})
                    
                    # Return user-specific data
                    return {
                        "user": username,
                        "profile": user_profile,
                        "sessions": sessions,
                        "total_sessions": len(sessions)
                    }
                else:
                    return {"error": f"No sessions found for user '{username}'"}
                
        else:
            return {"error": "Log file not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

@router.get("/session_summary")
async def get_session_summary():
    """Get summary of all user sessions"""
    try:
        if os.path.exists(DISTRACTION_LOG_FILE):
            with open(DISTRACTION_LOG_FILE, 'r') as f:
                log_data = json.load(f)
                
                # Prepare summary data
                summary = {
                    "total_users": len(log_data.get('users', {})),
                    "users": [],
                    "total_sessions": 0
                }
                
                # Add user summaries
                for username, profile in log_data.get('users', {}).items():
                    session_count = len(log_data.get('user_sessions', {}).get(username, []))
                    summary["users"].append({
                        "username": username,
                        "sessions": session_count,
                        "breaks_taken": profile.get("breaks_taken", 0),
                        "total_distraction_time": profile.get("total_distraction_seconds", 0),
                        "last_session": profile.get("last_session", "")
                    })
                    summary["total_sessions"] += session_count
                
                return summary
        else:
            return {"error": "Log file not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

@router.get("/user_log/{username}")
async def get_specific_user_log(username: str):
    """Get the complete log file for a specific user"""
    try:
        from services.attention_service import get_user_log_path, get_user_log
        
        user_log = get_user_log(username, force_reload=True)
        if user_log:
            return user_log
        else:
            return {"error": f"No log found for user '{username}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading user log: {str(e)}")

@router.get("/user_logs_list")
async def list_user_logs():
    """Get a list of all users with separate log files"""
    try:
        from services.attention_service import USER_LOGS_DIR
        
        log_files = []
        if os.path.exists(USER_LOGS_DIR):
            for file in os.listdir(USER_LOGS_DIR):
                if file.endswith("_log.json"):
                    username = file.replace("_log.json", "").replace("_", " ")
                    log_path = os.path.join(USER_LOGS_DIR, file)
                    
                    # Get basic stats
                    try:
                        with open(log_path, 'r') as f:
                            log_data = json.load(f)
                            stats = log_data.get("stats", {})
                            last_updated = log_data.get("last_updated", "")
                    except:
                        stats = {}
                        last_updated = ""
                    
                    log_files.append({
                        "username": username,
                        "file": file,
                        "stats": stats,
                        "last_updated": last_updated
                    })
        
        return {"user_logs": log_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing user logs: {str(e)}")

# MongoDB specific endpoints
@router.get("/mongodb/users/{username}/sessions")
async def get_mongodb_user_sessions(username: str):
    """Get user session data from MongoDB"""
    try:
        sessions = mongo_get_user_sessions(username)
        if sessions and len(sessions) > 0:
            return {
                "username": username,
                "sessions": sessions,
                "total_sessions": len(sessions),
                "source": "mongodb"
            }
        else:
            return {"error": f"No MongoDB sessions found for user '{username}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")

@router.get("/mongodb/users/{username}/events")
async def get_mongodb_user_events(username: str, session_id: Optional[str] = None):
    """Get user distraction events from MongoDB, optionally filtered by session ID"""
    try:
        events = mongo_get_user_distraction_events(username, session_id)
        if events and len(events) > 0:
            return {
                "username": username,
                "events": events,
                "total_events": len(events),
                "session_id": session_id,
                "source": "mongodb"
            }
        else:
            return {"error": f"No MongoDB events found for user '{username}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")

@router.get("/mongodb/status")
async def get_mongodb_status():
    """Check MongoDB connection status"""
    try:
        from services import mongo_service
        
        # Try to connect and get status
        is_connected = mongo_service.connect_to_mongodb()
        
        if is_connected:
            # Get some basic stats if connected
            users_collection = mongo_service.get_collection(mongo_service.USERS_COLLECTION)
            sessions_collection = mongo_service.get_collection(mongo_service.SESSIONS_COLLECTION)
            events_collection = mongo_service.get_collection(mongo_service.DISTRACTION_EVENTS_COLLECTION)
            
            return {
                "status": "connected",
                "database": mongo_service.MONGO_DB_NAME,
                "collections": {
                    "users": users_collection.count_documents({}) if users_collection is not None else 0,
                    "sessions": sessions_collection.count_documents({}) if sessions_collection is not None else 0,
                    "events": events_collection.count_documents({}) if events_collection is not None else 0
                }
            }
        else:
            return {
                "status": "disconnected",
                "database": mongo_service.MONGO_DB_NAME,
                "error": "Failed to connect to MongoDB"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
