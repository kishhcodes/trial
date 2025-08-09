from fastapi import APIRouter
from services.camera_service import get_status

router = APIRouter(
    prefix="",  # Change from "/api" to empty string to match original URL paths
    tags=["Status"]
)

@router.get("/status")
def status():
    """Return the current status of attention and emotion"""
    return get_status()

@router.get("/check_redirect")
def check_redirect():
    """Endpoint for the frontend to check if a redirect is needed"""
    status_data = get_status()
    return {"redirect": status_data.get("redirect", False)}
