"""
MongoDB Query Tool

This script lets you run custom queries against your MongoDB database
for the face recognition and attention tracking application.
"""
import os
import sys
import pprint
from datetime import datetime, timedelta
from services.mongo_service import connect_to_mongodb, get_collection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pretty printer for MongoDB results
pp = pprint.PrettyPrinter(indent=2)

def query_users():
    """Query all users in the database"""
    users_collection = get_collection('users')
    if users_collection is None:
        return []
    
    print("\n==== USERS ====")
    users = list(users_collection.find())
    for user in users:
        print(f"\nUsername: {user['username']}")
        print(f"First seen: {user.get('first_seen')}")
        print(f"Last seen: {user.get('last_seen')}")
        print(f"Total sessions: {user.get('total_sessions', 0)}")
        print(f"Total distraction events: {user.get('total_distraction_events', 0)}")
        print(f"Total distraction seconds: {user.get('total_distraction_seconds', 0)}")
    
    return users

def query_sessions(username=None, limit=5):
    """
    Query user sessions, optionally filtered by username
    
    Args:
        username: Optional username to filter by
        limit: Maximum number of sessions to return
    """
    sessions_collection = get_collection('sessions')
    if sessions_collection is None:
        return []
    
    # Build query
    query = {}
    if username:
        query["username"] = username
    
    print(f"\n==== SESSIONS {f'FOR {username}' if username else ''} ====")
    sessions = list(sessions_collection.find(query).sort("start_time", -1).limit(limit))
    
    for session in sessions:
        start_time = session.get("start_time")
        end_time = session.get("end_time", "Active")
        duration = session.get("duration_seconds", "N/A")
        
        print(f"\nSession ID: {session['_id']}")
        print(f"User: {session.get('username', 'Unknown')}")
        print(f"Time: {start_time} to {end_time}")
        print(f"Duration: {duration} seconds")
        print(f"Distraction time: {session.get('total_distraction_time', 0)} seconds")
        print(f"Breaks taken: {session.get('breaks_taken', 0)}")
    
    return sessions

def query_distraction_events(username=None, session_id=None, threshold_reached=None, limit=10):
    """
    Query distraction events with various filters
    
    Args:
        username: Optional username to filter by
        session_id: Optional session ID to filter by
        threshold_reached: Optional boolean to filter by threshold_reached flag
        limit: Maximum number of events to return
    """
    events_collection = get_collection('distraction_events')
    users_collection = get_collection('users')
    
    if events_collection is None or users_collection is None:
        return []
    
    # Build query
    query = {}
    
    if username:
        user = users_collection.find_one({"username": username})
        if user:
            query["user_id"] = user["_id"]
    
    if session_id:
        from bson.objectid import ObjectId
        try:
            query["session_id"] = ObjectId(session_id)
        except:
            print(f"Invalid session_id format: {session_id}")
    
    if threshold_reached is not None:
        query["threshold_reached"] = threshold_reached
    
    print(f"\n==== DISTRACTION EVENTS {f'FOR {username}' if username else ''} ====")
    events = list(events_collection.find(query).sort("start_time", -1).limit(limit))
    
    for event in events:
        start_time = event.get("start_time")
        end_time = event.get("end_time", "Ongoing")
        duration = event.get("duration_seconds", "N/A")
        
        print(f"\nEvent ID: {event['_id']}")
        print(f"Time: {start_time} to {end_time}")
        print(f"Duration: {duration} seconds")
        print(f"Threshold reached: {event.get('threshold_reached', False)}")
        print(f"Adaptation factor: {event.get('adaptation_factor', 1.0)}")
    
    return events

def analyze_distraction_patterns(username, days=7):
    """
    Analyze distraction patterns for a specific user
    
    Args:
        username: The username to analyze
        days: Number of days to analyze
    """
    events_collection = get_collection('distraction_events')
    users_collection = get_collection('users')
    
    if events_collection is None or users_collection is None:
        return
    
    user = users_collection.find_one({"username": username})
    if user is None:
        print(f"User {username} not found")
        return
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Query events in date range
    events = list(events_collection.find({
        "user_id": user["_id"],
        "start_time": {"$gte": start_date, "$lte": end_date}
    }).sort("start_time", 1))
    
    if not events:
        print(f"No events found for {username} in the last {days} days")
        return
    
    # Analyze patterns
    print(f"\n==== DISTRACTION PATTERN ANALYSIS FOR {username} (Last {days} days) ====")
    
    # Count events by day
    events_by_day = {}
    total_duration = 0
    threshold_reached_count = 0
    
    for event in events:
        day = event["start_time"].strftime("%Y-%m-%d")
        if day not in events_by_day:
            events_by_day[day] = {"count": 0, "duration": 0}
        
        events_by_day[day]["count"] += 1
        duration = event.get("duration_seconds", 0)
        events_by_day[day]["duration"] += duration
        total_duration += duration
        
        if event.get("threshold_reached", False):
            threshold_reached_count += 1
    
    # Display results
    print(f"Total distraction events: {len(events)}")
    print(f"Total distraction duration: {total_duration} seconds")
    print(f"Average duration per event: {total_duration / len(events):.2f} seconds")
    print(f"Breaks needed: {threshold_reached_count}")
    
    print("\nDistraction by day:")
    for day, data in events_by_day.items():
        print(f"  {day}: {data['count']} events, {data['duration']} seconds")

def main():
    """Main function to run queries based on command line arguments"""
    # Connect to MongoDB
    if not connect_to_mongodb():
        print("Failed to connect to MongoDB")
        return 1
    
    if len(sys.argv) < 2:
        print("Usage: python mongo_query.py [users|sessions|events|patterns]")
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "users":
        query_users()
    
    elif command == "sessions":
        username = sys.argv[2] if len(sys.argv) > 2 else None
        query_sessions(username)
    
    elif command == "events":
        username = sys.argv[2] if len(sys.argv) > 2 else None
        session_id = sys.argv[3] if len(sys.argv) > 3 else None
        query_distraction_events(username, session_id)
    
    elif command == "patterns":
        if len(sys.argv) < 3:
            print("Usage: python mongo_query.py patterns <username> [days]")
            return 1
        username = sys.argv[2]
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
        analyze_distraction_patterns(username, days)
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: users, sessions, events, patterns")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
