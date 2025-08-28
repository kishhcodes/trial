from fastapi import APIRouter
from services.camera_service import get_status, reset_redirect_flag
from services.attention_service import reset_distraction_counter

router = APIRouter(
    prefix="/api",
    tags=["Status"]
)

@router.get("/status")
async def get_current_status():
    return get_status()

@router.get("/reset_redirect")
async def reset_redirect():
    """Reset the redirect flag after user confirms they're ready to continue"""
    was_redirecting = reset_redirect_flag()
    
    # Also reset the distraction counter to prevent immediate redirect
    reset_distraction_counter()
    
    return {"was_redirecting": was_redirecting, "counter_reset": True}