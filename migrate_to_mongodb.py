"""
MongoDB Data Migration Tool

This script helps migrate existing JSON logs to MongoDB.
It reads the existing JSON log files and imports them into MongoDB.
"""
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from bson.objectid import ObjectId

# Add project path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MongoDB service
from services.mongo_service import (
    connect_to_mongodb, 
    get_collection, 
    disconnect_mongodb,
    USERS_COLLECTION,
    SESSIONS_COLLECTION,
    DISTRACTION_EVENTS_COLLECTION
)
from services.attention_service import DISTRACTION_LOG_FILE, USER_LOGS_DIR

# Load environment variables
load_dotenv()

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" 🔍 {title}")
    print("=" * 80)

def print_step(step):
    """Print a step header"""
    print(f"\n📌 {step}")

def connect_and_check():
    """Connect to MongoDB and check collections"""
    print_step("Connecting to MongoDB...")
    
    if not connect_to_mongodb():
        print("❌ Failed to connect to MongoDB")
        return False
    
    print("✅ Connected to MongoDB")
    
    # Get collections
    users_collection = get_collection(USERS_COLLECTION)
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    events_collection = get_collection(DISTRACTION_EVENTS_COLLECTION)
    
    if not all([users_collection, sessions_collection, events_collection]):
        print("❌ Failed to get all required collections")
        return False
    
    print("✅ All required collections are available")
    
    # Check collection counts
    print("\nCurrent collection counts:")
    print(f"   Users: {users_collection.count_documents({})}")
    print(f"   Sessions: {sessions_collection.count_documents({})}")
    print(f"   Distraction Events: {events_collection.count_documents({})}")
    
    return True

def load_json_logs():
    """Load and parse JSON logs"""
    print_step("Loading JSON logs...")
    
    if not os.path.exists(DISTRACTION_LOG_FILE):
        print(f"❌ Main log file not found at: {DISTRACTION_LOG_FILE}")
        return None
    
    try:
        with open(DISTRACTION_LOG_FILE, 'r') as f:
            log_data = json.load(f)
            
        print("✅ Successfully loaded main log file")
        print(f"   Users: {len(log_data.get('users', {}))} users")
        print(f"   Sessions: {len(log_data.get('sessions', []))} sessions")
        
        # Also check for individual user logs
        user_logs = {}
        if os.path.exists(USER_LOGS_DIR):
            for file in os.listdir(USER_LOGS_DIR):
                if file.endswith("_log.json"):
                    username = file.replace("_log.json", "").replace("_", " ")
                    log_path = os.path.join(USER_LOGS_DIR, file)
                    
                    try:
                        with open(log_path, 'r') as f:
                            user_log_data = json.load(f)
                            user_logs[username] = user_log_data
                            print(f"   Loaded log for user: {username}")
                    except Exception as e:
                        print(f"   ⚠️ Error loading log for {username}: {e}")
        
        print(f"   Individual user logs: {len(user_logs)}")
        
        return {
            "main_log": log_data,
            "user_logs": user_logs
        }
    except Exception as e:
        print(f"❌ Failed to load logs: {e}")
        return None

def migrate_users(log_data, dry_run=True):
    """Migrate user data to MongoDB"""
    print_step("Migrating users to MongoDB...")
    
    users = log_data.get("main_log", {}).get("users", {})
    users_collection = get_collection(USERS_COLLECTION)
    
    if not users:
        print("⚠️ No users found in logs")
        return 0
    
    if dry_run:
        print("🔍 DRY RUN MODE - No data will be written to MongoDB")
    
    migrated_count = 0
    for username, user_data in users.items():
        # Check if user already exists in MongoDB
        existing_user = users_collection.find_one({"username": username})
        
        if existing_user:
            print(f"   ⚠️ User '{username}' already exists in MongoDB - skipping")
            continue
        
        # Prepare user document
        user_doc = {
            "username": username,
            "created_at": datetime.now(),
            "last_active": user_data.get("last_session", None),
            "distraction_profile": {
                "total_distraction_seconds": user_data.get("total_distraction_seconds", 0),
                "breaks_taken": user_data.get("breaks_taken", 0),
                "total_sessions": user_data.get("total_sessions", 0),
                "avg_distraction_duration": user_data.get("avg_distraction_duration", 0)
            },
            "source": "json_migration"
        }
        
        # Insert user if not dry run
        if not dry_run:
            try:
                result = users_collection.insert_one(user_doc)
                print(f"   ✅ Migrated user '{username}' - ID: {result.inserted_id}")
                migrated_count += 1
            except Exception as e:
                print(f"   ❌ Failed to migrate user '{username}': {e}")
        else:
            print(f"   🔍 Would migrate user '{username}'")
            migrated_count += 1
    
    print(f"✅ Migration complete: {migrated_count} users migrated")
    return migrated_count

def migrate_sessions(log_data, dry_run=True):
    """Migrate session data to MongoDB"""
    print_step("Migrating sessions to MongoDB...")
    
    sessions = log_data.get("main_log", {}).get("sessions", [])
    user_sessions = log_data.get("main_log", {}).get("user_sessions", {})
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    if not sessions:
        print("⚠️ No sessions found in logs")
        return 0
    
    if dry_run:
        print("🔍 DRY RUN MODE - No data will be written to MongoDB")
    
    # Create a mapping of session_ids to MongoDB ObjectIds for later reference
    session_id_map = {}
    
    migrated_count = 0
    for session in sessions:
        session_id = session.get("session_id", None)
        if not session_id:
            print("   ⚠️ Session without ID found - skipping")
            continue
        
        # Generate a MongoDB ObjectId for this session
        mongo_id = ObjectId()
        session_id_map[session_id] = mongo_id
        
        # Find users in this session
        session_users = []
        for username, user_session_ids in user_sessions.items():
            if session_id in user_session_ids:
                session_users.append(username)
        
        if not session_users:
            print(f"   ⚠️ Session {session_id} has no associated users - using 'Unknown'")
            session_users = ["Unknown"]
        
        # Create one session document per user
        for username in session_users:
            # Check if this session already exists for this user
            existing_session = sessions_collection.find_one({
                "original_session_id": session_id,
                "username": username
            })
            
            if existing_session:
                print(f"   ⚠️ Session {session_id} for user '{username}' already exists - skipping")
                continue
            
            # Prepare session document
            start_time = session.get("start_time", None)
            end_time = session.get("end_time", None)
            
            try:
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            except:
                # If date parsing fails, use current time
                if not start_time:
                    start_time = datetime.now()
                if not end_time:
                    end_time = datetime.now()
            
            session_doc = {
                "_id": mongo_id,
                "username": username,
                "original_session_id": session_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": session.get("duration_seconds", 0),
                "total_distractions": len(session.get("distraction_events", [])),
                "source": "json_migration"
            }
            
            # Insert session if not dry run
            if not dry_run:
                try:
                    result = sessions_collection.insert_one(session_doc)
                    print(f"   ✅ Migrated session {session_id} for user '{username}'")
                    migrated_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to migrate session {session_id} for user '{username}': {e}")
            else:
                print(f"   🔍 Would migrate session {session_id} for user '{username}'")
                migrated_count += 1
    
    print(f"✅ Migration complete: {migrated_count} sessions migrated")
    return migrated_count

def migrate_events(log_data, dry_run=True):
    """Migrate distraction events to MongoDB"""
    print_step("Migrating distraction events to MongoDB...")
    
    sessions = log_data.get("main_log", {}).get("sessions", [])
    events_collection = get_collection(DISTRACTION_EVENTS_COLLECTION)
    
    if not sessions:
        print("⚠️ No sessions found in logs")
        return 0
    
    if dry_run:
        print("🔍 DRY RUN MODE - No data will be written to MongoDB")
    
    migrated_count = 0
    for session in sessions:
        session_id = session.get("session_id", None)
        if not session_id:
            continue
            
        events = session.get("distraction_events", [])
        if not events:
            print(f"   ℹ️ Session {session_id} has no distraction events")
            continue
            
        # Find which events belong to which users
        user_events = session.get("user_events", {})
        
        # If no user_events mapping, assign all events to "Unknown"
        if not user_events:
            user_events = {"Unknown": list(range(len(events)))}
        
        # Process each user's events
        for username, event_indices in user_events.items():
            for idx in event_indices:
                if idx >= len(events):
                    print(f"   ⚠️ Invalid event index {idx} for user '{username}' - skipping")
                    continue
                    
                event = events[idx]
                
                # Generate a unique identifier for this event
                event_id = f"{session_id}_event_{idx}"
                
                # Check if this event already exists
                existing_event = events_collection.find_one({
                    "original_event_id": event_id
                })
                
                if existing_event:
                    print(f"   ⚠️ Event {event_id} already exists - skipping")
                    continue
                
                # Parse dates
                start_time = event.get("start_time", None)
                end_time = event.get("end_time", None)
                
                try:
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    
                    if isinstance(end_time, str):
                        end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                except:
                    # If date parsing fails, use current time
                    if not start_time:
                        start_time = datetime.now()
                    if not end_time:
                        end_time = datetime.now()
                
                # Prepare event document
                event_doc = {
                    "username": username,
                    "session_id": session_id,
                    "original_event_id": event_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": event.get("duration_seconds", 0),
                    "threshold_reached": event.get("threshold_reached", False),
                    "source": "json_migration"
                }
                
                # Insert event if not dry run
                if not dry_run:
                    try:
                        result = events_collection.insert_one(event_doc)
                        print(f"   ✅ Migrated event {event_id} for user '{username}'")
                        migrated_count += 1
                    except Exception as e:
                        print(f"   ❌ Failed to migrate event {event_id}: {e}")
                else:
                    print(f"   🔍 Would migrate event {event_id} for user '{username}'")
                    migrated_count += 1
    
    print(f"✅ Migration complete: {migrated_count} events migrated")
    return migrated_count

def run_migration(dry_run=True):
    """Run the complete migration process"""
    print_header("MONGODB DATA MIGRATION")
    
    # Connect to MongoDB
    if not connect_and_check():
        return
    
    # Load JSON logs
    logs = load_json_logs()
    if not logs:
        return
    
    # Confirm migration
    if not dry_run:
        print("\n⚠️ WARNING: This will migrate data from JSON logs to MongoDB")
        confirm = input("Are you sure you want to proceed? (yes/no): ")
        if confirm.lower() not in ["yes", "y"]:
            print("Migration cancelled")
            return
    
    # Migrate users
    users_migrated = migrate_users(logs, dry_run)
    
    # Migrate sessions
    sessions_migrated = migrate_sessions(logs, dry_run)
    
    # Migrate events
    events_migrated = migrate_events(logs, dry_run)
    
    # Summary
    print_header("MIGRATION SUMMARY")
    print(f"Users migrated: {users_migrated}")
    print(f"Sessions migrated: {sessions_migrated}")
    print(f"Events migrated: {events_migrated}")
    
    if dry_run:
        print("\n🔍 This was a DRY RUN - No data was written to MongoDB")
        print("To perform the actual migration, run the script with --execute")
    
    # Disconnect from MongoDB
    disconnect_mongodb()

if __name__ == "__main__":
    # Check if this is a dry run
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        dry_run = False
    
    run_migration(dry_run)
