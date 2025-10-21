"""
MongoDB connection test script.
Run this to verify that your MongoDB Docker container is accessible.
"""
import os
import sys
from services.mongo_service import connect_to_mongodb, get_collection
from services.mongo_adapter import mongo_log_app_startup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mongo_connection():
    """Test MongoDB connection and basic operations"""
    print("Testing MongoDB connection...")
    
    # Test connection
    is_connected = connect_to_mongodb()
    if not is_connected:
        print("❌ Failed to connect to MongoDB")
        return False
    
    print("✅ Connected to MongoDB successfully!")
    
    # Test writing to collections
    try:
        # Log a test startup event
        startup_id = mongo_log_app_startup("test", "MongoDB connection test")
        if startup_id:
            print(f"✅ Successfully wrote test data to MongoDB (ID: {startup_id})")
        else:
            print("❌ Failed to write test data to MongoDB")
            return False
            
        # Get collection stats
        users_collection = get_collection('users')
        sessions_collection = get_collection('sessions')
        events_collection = get_collection('distraction_events')
        
        users_count = users_collection.count_documents({})
        sessions_count = sessions_collection.count_documents({})
        events_count = events_collection.count_documents({})
        
        print("\nCollection statistics:")
        print(f"- Users: {users_count}")
        print(f"- Sessions: {sessions_count}")
        print(f"- Distraction events: {events_count}")
        
        return True
    except Exception as e:
        print(f"❌ Error during MongoDB operations: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mongo_connection()
    sys.exit(0 if success else 1)
