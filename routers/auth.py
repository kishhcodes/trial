from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from services.user_service import get_authentication_status, reset_authentication, set_monitoring_active

router = APIRouter(tags=["Authentication"])
templates = Jinja2Templates(directory="templates")

@router.get("/auth_status")
async def auth_status():
    """Get the current authentication status"""
    return get_authentication_status()

@router.get("/reset_auth")
async def reset_auth():
    """Reset the authentication status"""
    reset_authentication()
    return {"success": True}

@router.get("/auth_check", response_class=HTMLResponse)
async def auth_check(request: Request):
    """Page that checks for user authentication before proceeding"""
    return templates.TemplateResponse("waiting.html", {"request": request})

@router.get("/auth_redirect")
async def auth_redirect():
    """Check auth status and redirect accordingly"""
    auth_status = get_authentication_status()
    if auth_status["authenticated"]:
        # Just signal frontend to redirect
        return {"redirect": True, "auth_status": auth_status}
    else:
        return {"redirect": False, "auth_status": auth_status}

@router.get("/set_monitoring_active")
async def activate_monitoring():
    """Set the monitoring mode active"""
    set_monitoring_active(True)
    return {"success": True}
