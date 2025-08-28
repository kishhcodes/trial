from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from datetime import datetime

# First, import services that don't depend on routers
from services.camera_service import release_camera
from services.attention_service import save_distraction_log, distraction_log

# Then import all routers
from routers import video, status, pages, logs, auth

app = FastAPI(title="Face Attention API", description="Detects face attention with Swagger UI.")

# CORS (optional, for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling middleware
@app.middleware("http")
async def errors_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"Request error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}"}
        )

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(pages.router)
app.include_router(video.router)
app.include_router(status.router)
app.include_router(logs.router)
app.include_router(auth.router)

@app.on_event("startup")
async def startup_event():
    """Run when the app starts"""
    # Log application startup in the distraction log
    current_session = distraction_log["sessions"][-1] 
    current_session["app_startup"] = datetime.now().isoformat()
    current_session["app_info"] = {
        "version": "1.0",
        "description": "Face attention monitoring system"
    }
    save_distraction_log(distraction_log)
    print("Application started, session logged")

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when app shuts down"""
    # Log application shutdown
    current_session = distraction_log["sessions"][-1]
    current_session["app_shutdown"] = datetime.now().isoformat()
    
    # Calculate session duration
    if "app_startup" in current_session:
        start_time = datetime.fromisoformat(current_session["app_startup"])
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()
        current_session["session_duration_seconds"] = duration_seconds
    
    # Ensure all user sessions are properly ended
    try:
        from services.attention_service import active_user_sessions, end_user_session
        # Create a copy of keys to avoid modification during iteration
        for username in list(active_user_sessions.keys()):
            end_user_session(username)
    except Exception as e:
        print(f"Error closing user sessions: {e}")
    
    save_distraction_log(distraction_log)
    release_camera()
    print("Application shutting down, all sessions finalized in log")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)