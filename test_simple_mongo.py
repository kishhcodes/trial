"""
Simple MongoDB connection test with error details
"""
import os
import traceback
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB connection details
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'krizzip')
MONGO_USER = os.getenv('MONGO_USER')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')

print(f"MongoDB URI: {MONGO_URI}")
print(f"Database name: {MONGO_DB_NAME}")
print(f"Username: {MONGO_USER}")
print(f"Password: {'*' * len(MONGO_PASSWORD) if MONGO_PASSWORD else None}")

# Create connection string
if MONGO_USER and MONGO_PASSWORD:
    # Explicitly specify authSource=krizzip since that's where our user is defined
    connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_URI.split('://')[1]}/{MONGO_DB_NAME}?authSource=krizzip"
else:
    connection_string = MONGO_URI

print(f"Connection string: {connection_string}")

try:
    # Connect to MongoDB
    client = MongoClient(connection_string)
    
    # Test connection
    db = client[MONGO_DB_NAME]
    db.command('ping')
    
    print("✅ Connected to MongoDB successfully!")
    
    # Print collections
    print(f"Collections in {MONGO_DB_NAME} database:")
    for collection in db.list_collection_names():
        print(f"- {collection}")
    
    # Create a test user for the database if needed
    if input("Create a new 'krizzuser' with password 'secure_password'? (y/n): ").lower() == 'y':
        admin_db = client['admin']
        admin_db.command('createUser', 'krizzuser', pwd='secure_password', 
                        roles=[{'role': 'readWrite', 'db': MONGO_DB_NAME}])
        print("User created successfully!")
    
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    print("\nFull error details:")
    traceback.print_exc()
