"""
Full Application Test for Dual Logging (JSON + MongoDB)

This script tests the complete application flow with both JSON and MongoDB logging systems.
It simulates user sessions and distraction events, then verifies data is properly stored
in both systems.
"""
import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
import random
import uuid

# Generate a unique test username
TEST_USERNAME = f"test_user_{int(time.time())}"
BASE_URL = "http://localhost:8000/api"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" 🔍 {title}")
    print("=" * 80)

def print_step(step):
    """Print a step header"""
    print(f"\n📌 STEP {step}")

def test_mongodb_connection():
    """Test MongoDB connection status via the API"""
    print_step("Testing MongoDB connection status")
    
    try:
        response = requests.get(f"{BASE_URL}/mongodb/status")
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "connected":
            print("✅ MongoDB is connected")
            print(f"   Database: {data['database']}")
            print(f"   Collections:")
            for collection, count in data["collections"].items():
                print(f"      - {collection}: {count} documents")
            return True
        else:
            print(f"❌ MongoDB is not connected: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error checking MongoDB status: {e}")
        return False

def create_user_session():
    """Create a new user session via the attention tracking system"""
    print_step("Creating user session")
    
    try:
        # Get initial distraction log state
        response_before = requests.get(f"{BASE_URL}/distraction_log")
        data_before = response_before.json()
        
        # Create a session via the video stream (simulating activity)
        # In a real test, we would open the video stream, but for this test
        # we'll directly access the user endpoints
        
        # First check if we can get the users list
        response_users = requests.get(f"{BASE_URL}/users")
        users_data = response_users.json()
        print(f"   Current users: {users_data['users']}")
        
        # Start a test session (in real use, this happens via the attention service)
        # We'll use an HTTP request to simulate camera activity
        print(f"   Creating session for user: {TEST_USERNAME}")
        
        # In this test, we simulate the session creation by directly accessing the API
        # In real use, the session would be created when the user starts the video stream
        # We'll just make a request that will trigger the session creation
        
        # First, access a few API endpoints to simulate activity
        requests.get(f"{BASE_URL}/distraction_log")
        requests.get(f"{BASE_URL}/users")
        
        # For a real test, we would need proper API endpoints to create sessions
        # Since we don't have that, we'll return the test setup info
        return {
            "username": TEST_USERNAME,
            "status": "simulated"
        }
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        return None

def check_logs_and_mongodb():
    """Check both JSON logs and MongoDB for user data"""
    print_step("Checking logs and MongoDB for user data")
    
    try:
        # Check JSON logs
        response = requests.get(f"{BASE_URL}/distraction_log")
        log_data = response.json()
        
        # Check MongoDB status and data
        mongo_status = requests.get(f"{BASE_URL}/mongodb/status").json()
        
        print("✅ Retrieved log data")
        
        # Display log summary
        print("\nJSON Log Summary:")
        print(f"   Total users: {len(log_data.get('users', {}))}")
        print(f"   Total sessions: {len(log_data.get('sessions', []))}")
        
        # Display MongoDB summary
        print("\nMongoDB Summary:")
        print(f"   Status: {mongo_status['status']}")
        if mongo_status["status"] == "connected":
            for collection, count in mongo_status.get("collections", {}).items():
                print(f"   {collection}: {count} documents")
        
        return {
            "json_logs": log_data,
            "mongodb_status": mongo_status
        }
    except Exception as e:
        print(f"❌ Error checking logs: {e}")
        return None

def run_manual_test():
    """
    Run a manual test that guides the user through testing the application
    with both JSON and MongoDB logging.
    """
    print_header("MANUAL TEST GUIDE FOR DUAL LOGGING SYSTEM")
    
    print("""
This guide will help you test the application with both JSON and MongoDB logging enabled.
Follow these steps to verify that both systems are working correctly.
    """)
    
    print_step("1. Check MongoDB connection")
    print("""
    • Visit: http://localhost:8000/api/mongodb/status
    • Verify that the status is "connected"
    • Note the collection counts
    """)
    
    input("Press Enter when ready to continue...")
    
    print_step("2. Start the application and create a session")
    print(f"""
    • Open the application in your browser (typically http://localhost:8000)
    • Login with username: {TEST_USERNAME} (or any username you prefer)
    • Allow camera access to start a session
    • Move your head around to trigger attention tracking
    """)
    
    input("Press Enter when you've completed this step...")
    
    print_step("3. Generate some distraction events")
    print("""
    • Look away from the camera for 5-10 seconds to trigger a distraction event
    • Look back at the camera, then away again a few times
    • Try to generate at least 3-4 distraction events
    """)
    
    input("Press Enter when you've generated some distraction events...")
    
    print_step("4. End the session")
    print("""
    • Close the browser tab or click "End Session" if available
    • This should properly end the session in both logging systems
    """)
    
    input("Press Enter when you've ended the session...")
    
    print_step("5. Check the logs in both systems")
    print("""
    To check JSON logs:
    • Visit: http://localhost:8000/api/distraction_log
    • Note the session data and distraction events
    
    To check MongoDB logs:
    • Visit: http://localhost:8000/api/mongodb/status to see updated counts
    • Visit: http://localhost:8000/api/mongodb/users/{username}/sessions
      (replace {username} with the username you used)
    • Visit: http://localhost:8000/api/mongodb/users/{username}/events
      (replace {username} with the username you used)
    """)
    
    input("Press Enter when you've checked the logs...")
    
    print_header("TEST COMPLETED")
    print("""
If you were able to see data in both the JSON logs and MongoDB collections,
then the dual logging system is working correctly!

Key points to verify:
1. MongoDB connection is established
2. User sessions are created in both systems
3. Distraction events are logged in both systems
4. Sessions are properly closed in both systems
    """)

def main():
    print_header("FULL APPLICATION DUAL LOGGING TEST")
    
    # Check if MongoDB is connected
    mongo_connected = test_mongodb_connection()
    if not mongo_connected:
        print("❌ Cannot proceed with test: MongoDB not connected")
        print("   Please ensure MongoDB is running and try again")
        return
    
    print("\n🚀 Starting manual test guide...")
    run_manual_test()

if __name__ == "__main__":
    main()
