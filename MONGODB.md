# MongoDB Integration for Session Logging

This integration adds MongoDB as a parallel logging system alongside the existing JSON file-based logging.

## Features

- Dual logging system - both MongoDB and JSON files are used
- No disruption to existing functionality
- New API endpoints to query MongoDB logs
- MongoDB connection health check
- User sessions and distraction events stored in MongoDB collections

## Configuration

MongoDB connection settings are configured in the `.env` file:

```
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=krizzip
MONGO_USER=krizzuser      # MongoDB username
MONGO_PASSWORD=secure_password  # MongoDB password
```

Authentication is enabled and the user `krizzuser` has been created with `readWrite` and `dbAdmin` roles for the `krizzip` database.

## MongoDB Authentication

The application uses MongoDB with authentication enabled. The following users have been created:

1. **Admin User**:
   - Username: `admin`
   - Has full administrative privileges across all databases
   - Used for MongoDB administration tasks

2. **Application User**:
   - Username: `krizzuser`
   - Has `readWrite` and `dbAdmin` roles for the `krizzip` database
   - Used by the application for all database operations

### Connection String Format

When authentication is enabled, the connection string format is:

```
mongodb://username:password@host:port/database?authSource=database
```

The `authSource` parameter specifies which database contains the user's credentials. In our case, it should be set to `krizzip` since that's where the application user is defined.

## API Endpoints

### MongoDB Status
```
GET /api/mongodb/status
```
Returns the connection status and basic stats about MongoDB collections.

### MongoDB User Sessions
```
GET /api/mongodb/users/{username}/sessions
```
Returns all sessions for a specific user from MongoDB.

### MongoDB User Distraction Events
```
GET /api/mongodb/users/{username}/events?session_id={optional_session_id}
```
Returns distraction events for a specific user, optionally filtered by session ID.

## MongoDB Collections

The system uses the following collections:

- `users`: User profiles and aggregate statistics
- `sessions`: User session tracking
- `distraction_events`: Individual distraction events
- `app_logs`: Application startup and shutdown logs

## Implementation Details

The MongoDB integration is implemented as a non-invasive patch that extends the existing JSON-based logging system. The implementation follows these principles:

1. **Backward Compatibility**: Existing JSON logging continues to work as before
2. **Parallel Logging**: Both systems log the same data simultaneously
3. **Graceful Fallback**: If MongoDB is unavailable, the system falls back to JSON logging only
4. **Enhanced Querying**: MongoDB allows for more advanced queries and aggregations

The integration is composed of these components:

- `mongo_service.py`: Core MongoDB connection and CRUD functions
- `mongo_adapter.py`: Adapter functions that handle errors and provide a clean interface
- `attention_mongo_patch.py`: Non-invasive patches to the existing attention service

## Requirement

- MongoDB 4.0+
- PyMongo package (added to requirements.txt)
