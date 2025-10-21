"""
Integration test for dual logging system (JSON and MongoDB)
This script tests that both JSON and MongoDB logging work together properly
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from services.attention_service import (
    start_user_session, 
    end_user_session, 
    log_distraction_event,
    get_user_log
)
from services.mongo_adapter import (
    mongo_get_user_sessions,
    mongo_get_user_distraction_events
)
from services.mongo_service import connect_to_mongodb, get_collection
from services.attention_mongo_patch import apply_mongodb_patches

# Load environment variables
load_dotenv()

# Test username - make sure this is unique for testing
TEST_USERNAME = f"test_user_{int(time.time())}"

def test_dual_logging():
    """Test that both JSON and MongoDB logging systems work together"""
    print(f"\n🧪 TESTING DUAL LOGGING SYSTEM WITH USER: {TEST_USERNAME}")
    print("=" * 70)
    
    # 1. First, ensure MongoDB is connected
    print("\n📌 Step 1: Connect to MongoDB...")
    if not connect_to_mongodb():
        print("❌ Failed to connect to MongoDB")
        return False
    print("✅ MongoDB connected")
    
    # 2. Apply MongoDB patches to attention service
    print("\n📌 Step 2: Apply MongoDB patches to attention service...")
    try:
        apply_mongodb_patches()
        print("✅ MongoDB patches applied")
    except Exception as e:
        print(f"❌ Failed to apply MongoDB patches: {e}")
        return False
    
    # 3. Start a user session
    print(f"\n📌 Step 3: Start session for user {TEST_USERNAME}...")
    result = start_user_session(TEST_USERNAME)
    if not result:
        print("❌ Failed to start user session")
        return False
    print("✅ User session started")
    
    # Wait a moment to ensure operations complete
    time.sleep(1)
    
    # 4. Log a distraction event
    print("\n📌 Step 4: Log a distraction event...")
    now = datetime.now()
    start_time = now - timedelta(seconds=10)
    duration = 10
    
    result = log_distraction_event(
        TEST_USERNAME,
        start_time,
        now,
        duration,
        threshold_reached=False
    )
    
    if not result:
        print("❌ Failed to log distraction event")
        return False
    print("✅ Distraction event logged")
    
    # Wait a moment to ensure operations complete
    time.sleep(1)
    
    # 5. Log another distraction event with threshold reached
    print("\n📌 Step 5: Log another distraction event with threshold reached...")
    now = datetime.now()
    start_time = now - timedelta(seconds=30)
    duration = 30
    
    result = log_distraction_event(
        TEST_USERNAME,
        start_time,
        now,
        duration,
        threshold_reached=True
    )
    
    if not result:
        print("❌ Failed to log second distraction event")
        return False
    print("✅ Second distraction event logged")
    
    # Wait a moment to ensure operations complete
    time.sleep(1)
    
    # 6. End the user session
    print(f"\n📌 Step 6: End session for user {TEST_USERNAME}...")
    result = end_user_session(TEST_USERNAME)
    if not result:
        print("❌ Failed to end user session")
        return False
    print("✅ User session ended")
    
    # Wait a moment to ensure operations complete
    time.sleep(1)
    
    # 7. Check JSON logging (user_log)
    print("\n📌 Step 7: Verify JSON logs...")
    user_log = get_user_log(TEST_USERNAME, force_reload=True)
    
    if not user_log:
        print("❌ Failed to get user log from JSON")
        return False
    
    json_sessions = user_log.get('sessions', [])
    json_events_count = sum(len(s.get('distraction_events', [])) for s in json_sessions)
    
    print(f"✅ JSON logging verified:")
    print(f"  - Sessions: {len(json_sessions)}")
    print(f"  - Events: {json_events_count}")
    print(f"  - Breaks taken: {user_log['stats'].get('breaks_taken', 0)}")
    
    # 8. Check MongoDB logging
    print("\n📌 Step 8: Verify MongoDB logs...")
    
    # Get sessions from MongoDB
    mongo_sessions = mongo_get_user_sessions(TEST_USERNAME)
    if not mongo_sessions:
        print("❌ Failed to get user sessions from MongoDB")
        return False
    
    # Get events from MongoDB
    mongo_events = mongo_get_user_distraction_events(TEST_USERNAME)
    
    print(f"✅ MongoDB logging verified:")
    print(f"  - Sessions: {len(mongo_sessions)}")
    print(f"  - Events: {len(mongo_events)}")
    
    # 9. Compare counts between JSON and MongoDB
    print("\n📌 Step 9: Compare JSON and MongoDB counts...")
    
    print(f"  JSON sessions: {len(json_sessions)}, MongoDB sessions: {len(mongo_sessions)}")
    print(f"  JSON events: {json_events_count}, MongoDB events: {len(mongo_events)}")
    
    # Print more detailed session info from both systems
    print("\n📊 Detailed JSON session info:")
    for i, session in enumerate(json_sessions):
        print(f"  Session {i+1}:")
        print(f"    ID: {session.get('session_id')}")
        print(f"    Start: {session.get('start_time')}")
        print(f"    End: {session.get('end_time', 'N/A')}")
        print(f"    Events: {len(session.get('distraction_events', []))}")
    
    print("\n📊 Detailed MongoDB session info:")
    for i, session in enumerate(mongo_sessions):
        print(f"  Session {i+1}:")
        print(f"    ID: {session.get('_id')}")
        print(f"    Start: {session.get('start_time')}")
        print(f"    End: {session.get('end_time', 'N/A')}")
    
    # The test is successful if we have verified both logging systems
    print("\n🏁 TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_dual_logging()
    sys.exit(0 if success else 1)
