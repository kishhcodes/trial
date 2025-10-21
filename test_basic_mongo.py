"""
Simple MongoDB connection test script.
"""
import os
import sys
import traceback
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

def test_mongo():
    """Test basic MongoDB connectivity"""
    try:
        # Get connection settings
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        mongo_db = os.getenv('MONGO_DB_NAME', 'krizzip')
        
        print(f"Connecting to: {mongo_uri}, DB: {mongo_db}")
        
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[mongo_db]
        
        # Test connection
        server_info = client.server_info()
        print(f"MongoDB version: {server_info['version']}")
        
        # List databases
        databases = client.list_database_names()
        print(f"Available databases: {', '.join(databases)}")
        
        # List collections in our database
        collections = db.list_collection_names()
        print(f"Collections in {mongo_db}: {', '.join(collections or ['none'])}")
        
        # Create a test document
        test_coll = db["test_collection"]
        result = test_coll.insert_one({"test": True, "timestamp": "now"})
        print(f"Inserted document ID: {result.inserted_id}")
        
        # Query it back
        doc = test_coll.find_one({"_id": result.inserted_id})
        print(f"Retrieved document: {doc}")
        
        # Clean up
        test_coll.delete_one({"_id": result.inserted_id})
        print("Test document deleted")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mongo()
    sys.exit(0 if success else 1)
