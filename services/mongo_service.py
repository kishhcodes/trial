"""
MongoDB service for managing user session logging.
Provides parallel logging alongside the JSON file-based system.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'krizzip')
MONGO_USER = os.getenv('MONGO_USER', None)
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD', None)

# MongoDB Collections
USERS_COLLECTION = 'users'
SESSIONS_COLLECTION = 'sessions'
DISTRACTION_EVENTS_COLLECTION = 'distraction_events'
APP_LOGS_COLLECTION = 'app_logs'

# MongoDB client
client: Optional[MongoClient] = None
db: Optional[Database] = None

def connect_to_mongodb() -> bool:
    """
    Connect to MongoDB server using environment variables.
    Returns True if connection was successful, False otherwise.
    """
    global client, db
    
    try:
        if MONGO_USER and MONGO_PASSWORD:
            # Properly format the connection string with authentication
            # Use authSource=krizzip since that's where our user is defined
            if '?' in MONGO_URI:
                base_uri, params = MONGO_URI.split('?', 1)
                connection_string = f'mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{base_uri.split("://")[1]}?authSource=krizzip&{params}'
            else:
                connection_string = f'mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_URI.split("://")[1]}/{MONGO_DB_NAME}?authSource=krizzip'
        else:
            connection_string = MONGO_URI
            
        print(f"Connecting with: {connection_string}")
        client = MongoClient(connection_string)
        db = client[MONGO_DB_NAME]
        
        # Test connection
        db.command('ping')
        print(f"✅ Connected to MongoDB: {MONGO_DB_NAME}")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return False

def get_collection(collection_name: str) -> Optional[Collection]:
    """Get a MongoDB collection if connected"""
    global db
    if db is not None:  # Proper check against None
        return db[collection_name]
    else:
        print("MongoDB not connected")
        return None

def log_app_startup(version: str, description: str) -> Optional[str]:
    """
    Log application startup in MongoDB
    Returns the ObjectId as string if successful
    """
    collection = get_collection(APP_LOGS_COLLECTION)
    if collection is None:  # Proper check against None
        return None
        
    log_data = {
        "event": "startup",
        "timestamp": datetime.now(),
        "app_info": {
            "version": version,
            "description": description
        }
    }
    
    try:
        result = collection.insert_one(log_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error logging app startup to MongoDB: {e}")
        return None

def log_app_shutdown(startup_id: Optional[str] = None) -> bool:
    """
    Log application shutdown in MongoDB
    Updates the startup record if startup_id is provided
    """
    collection = get_collection(APP_LOGS_COLLECTION)
    if collection is None:
        return False
        
    now = datetime.now()
    
    if startup_id:
        try:
            # Update the existing record
            from bson.objectid import ObjectId
            result = collection.update_one(
                {"_id": ObjectId(startup_id)},
                {
                    "$set": {
                        "shutdown_time": now,
                        "duration_seconds": (now - collection.find_one({"_id": ObjectId(startup_id)})["timestamp"]).total_seconds()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating app shutdown in MongoDB: {e}")
    
    # If no startup_id or update failed, create new shutdown record
    try:
        log_data = {
            "event": "shutdown",
            "timestamp": now
        }
        result = collection.insert_one(log_data)
        return True
    except Exception as e:
        print(f"Error logging app shutdown to MongoDB: {e}")
        return False

def get_or_create_user(username: str) -> Optional[str]:
    """
    Get or create a user document in MongoDB.
    Returns the user ID as string if successful.
    """
    if not username or username == "Unknown":
        return None
        
    collection = get_collection(USERS_COLLECTION)
    if collection is None:
        return None
    
    try:
        # Find existing user
        user_doc = collection.find_one({"username": username})
        
        if user_doc is not None:
            # Update last_seen
            collection.update_one(
                {"_id": user_doc["_id"]},
                {"$set": {"last_seen": datetime.now()}}
            )
            return str(user_doc["_id"])
        else:
            # Create new user
            new_user = {
                "username": username,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "total_distraction_events": 0,
                "total_distraction_seconds": 0,
                "average_distraction_duration": 0,
                "breaks_taken": 0,
                "total_sessions": 0
            }
            result = collection.insert_one(new_user)
            print(f"Created new user in MongoDB: {username}")
            return str(result.inserted_id)
    except Exception as e:
        print(f"Error with user document in MongoDB: {e}")
        return None

def start_user_session(username: str) -> Optional[str]:
    """
    Start a new user session in MongoDB.
    Returns the session ID as string if successful.
    """
    if not username or username == "Unknown":
        return None
    
    # Get user ID
    user_id = get_or_create_user(username)
    if not user_id:
        return None
    
    # Get sessions collection
    collection = get_collection(SESSIONS_COLLECTION)
    if collection is None:
        return None
    
    try:
        # Check if user has an active session
        from bson.objectid import ObjectId
        active_session = collection.find_one({
            "user_id": ObjectId(user_id),
            "end_time": {"$exists": False}
        })
        
        # If active session exists, end it first
        if active_session is not None:
            end_user_session(username)
        
        # Create new session
        now = datetime.now()
        session_data = {
            "user_id": ObjectId(user_id),
            "username": username,
            "start_time": now,
            "total_distraction_time": 0,
            "breaks_taken": 0,
            "active": True
        }
        
        result = collection.insert_one(session_data)
        session_id = str(result.inserted_id)
        
        # Update user stats
        users_collection = get_collection(USERS_COLLECTION)
        if users_collection is not None:
            users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$inc": {"total_sessions": 1}}
            )
        
        print(f"Started MongoDB session for user: {username}")
        return session_id
    except Exception as e:
        print(f"Error starting MongoDB session: {e}")
        return None

def end_user_session(username: str) -> bool:
    """
    End a user's active session in MongoDB.
    Returns True if successful.
    """
    if not username or username == "Unknown":
        return False
    
    # Get user ID
    user_id = get_or_create_user(username)
    if not user_id:
        return False
    
    # Get sessions collection
    collection = get_collection(SESSIONS_COLLECTION)
    if collection is None:
        return False
    
    try:
        from bson.objectid import ObjectId
        # Find active session
        active_session = collection.find_one({
            "user_id": ObjectId(user_id),
            "active": True
        })
        
        if active_session is None:
            return False
        
        # Update session with end time and duration
        now = datetime.now()
        duration_seconds = (now - active_session["start_time"]).total_seconds()
        
        collection.update_one(
            {"_id": active_session["_id"]},
            {
                "$set": {
                    "end_time": now,
                    "duration_seconds": duration_seconds,
                    "active": False
                }
            }
        )
        
        print(f"Ended MongoDB session for user: {username}")
        return True
    except Exception as e:
        print(f"Error ending MongoDB session: {e}")
        return False

def log_distraction_event(
    username: str, 
    start_time: datetime, 
    end_time: Optional[datetime], 
    duration_seconds: int,
    threshold_reached: bool = False,
    threshold_used: int = 75,
    adaptation_factor: float = 1.0
) -> Optional[str]:
    """
    Log a distraction event to MongoDB.
    Returns the event ID as string if successful.
    """
    if not username or username == "Unknown":
        return None
    
    # Get user ID
    user_id = get_or_create_user(username)
    if not user_id:
        return None
    
    # Find active session
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    if sessions_collection is None:
        return None
    
    try:
        from bson.objectid import ObjectId
        active_session = sessions_collection.find_one({
            "user_id": ObjectId(user_id),
            "active": True
        })
        
        if active_session is None:
            # Start a new session if none active
            session_id = start_user_session(username)
            if session_id is None:
                return None
            active_session = sessions_collection.find_one({"_id": ObjectId(session_id)})
        
        # Create event
        events_collection = get_collection(DISTRACTION_EVENTS_COLLECTION)
        if events_collection is None:
            return None
        
        event_data = {
            "user_id": ObjectId(user_id),
            "session_id": active_session["_id"],
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "threshold_used": threshold_used,
            "threshold_reached": threshold_reached,
            "adaptation_factor": adaptation_factor
        }
        
        result = events_collection.insert_one(event_data)
        event_id = str(result.inserted_id)
        
        # Update session stats
        sessions_collection.update_one(
            {"_id": active_session["_id"]},
            {
                "$inc": {
                    "total_distraction_time": duration_seconds,
                    "breaks_taken": 1 if threshold_reached else 0
                }
            }
        )
        
        # Update user stats
        users_collection = get_collection(USERS_COLLECTION)
        if users_collection is not None:
            users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$inc": {
                        "total_distraction_events": 1,
                        "total_distraction_seconds": duration_seconds,
                        "breaks_taken": 1 if threshold_reached else 0
                    }
                }
            )
            
            # Recalculate average
            user = users_collection.find_one({"_id": ObjectId(user_id)})
            if user["total_distraction_events"] > 0:
                avg = user["total_distraction_seconds"] / user["total_distraction_events"]
                users_collection.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$set": {"average_distraction_duration": avg}}
                )
        
        return event_id
    except Exception as e:
        print(f"Error logging distraction event to MongoDB: {e}")
        return None

def get_user_sessions(username: str) -> List[Dict]:
    """
    Get all sessions for a user from MongoDB.
    Returns a list of session documents.
    """
    if not username or username == "Unknown":
        return []
    
    # Get user ID
    users_collection = get_collection(USERS_COLLECTION)
    if users_collection is None:
        return []
    
    user = users_collection.find_one({"username": username})
    if user is None:
        return []
    
    # Get sessions
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    if sessions_collection is None:
        return []
    
    try:
        # Convert MongoDB documents to dictionaries
        sessions = []
        for session in sessions_collection.find({"user_id": user["_id"]}):
            # Convert ObjectId to string for serialization
            session["_id"] = str(session["_id"])
            session["user_id"] = str(session["user_id"])
            sessions.append(session)
        
        return sessions
    except Exception as e:
        print(f"Error getting MongoDB sessions: {e}")
        return []

def get_user_distraction_events(username: str, session_id: Optional[str] = None) -> List[Dict]:
    """
    Get all distraction events for a user from MongoDB, optionally filtered by session.
    Returns a list of event documents.
    """
    if not username or username == "Unknown":
        return []
    
    # Get user ID
    users_collection = get_collection(USERS_COLLECTION)
    if users_collection is None:
        return []
    
    user = users_collection.find_one({"username": username})
    if user is None:
        return []
    
    # Get events
    events_collection = get_collection(DISTRACTION_EVENTS_COLLECTION)
    if events_collection is None:
        return []
    
    try:
        # Build query
        from bson.objectid import ObjectId
        query = {"user_id": user["_id"]}
        
        if session_id:
            query["session_id"] = ObjectId(session_id)
        
        # Convert MongoDB documents to dictionaries
        events = []
        for event in events_collection.find(query):
            # Convert ObjectId to string for serialization
            event["_id"] = str(event["_id"])
            event["user_id"] = str(event["user_id"])
            event["session_id"] = str(event["session_id"])
            events.append(event)
        
        return events
    except Exception as e:
        print(f"Error getting MongoDB events: {e}")
        return []

# Connect to MongoDB when the module is imported
connect_to_mongodb()
