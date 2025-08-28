from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from services.user_service import set_monitoring_active, get_authentication_status

router = APIRouter(tags=["Pages"])
templates = Jinja2Templates(directory="templates")

def verify_authentication():
    """Verify that user is authenticated before allowing access to monitoring"""
    auth_status = get_authentication_status()
    if not auth_status["authenticated"]:
        raise HTTPException(status_code=401, detail="Authentication required")
    return auth_status["user"]

@router.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    """Home page now redirects to auth check first"""
    set_monitoring_active(False)  # We're not on the monitoring page yet
    return templates.TemplateResponse("waiting.html", {"request": request})

@router.get("/waiting", response_class=HTMLResponse)
def waiting_page(request: Request):
    """Page that waits for a face to be detected"""
    return templates.TemplateResponse("waiting.html", {"request": request})

@router.get("/watch", response_class=HTMLResponse)
def watch_feed(request: Request, username: str = Depends(verify_authentication)):
    """
    Page that shows the live camera feed with attention tracking.
    Now requires authentication and locks in the user.
    """
    set_monitoring_active(True)  # We're now on the monitoring page with fixed user
    return templates.TemplateResponse("index.html", {
        "request": request,
        "username": username
    })

@router.get("/break_page", response_class=HTMLResponse)
def break_page(request: Request):
    """Page to redirect to when user is distracted for too long"""
    set_monitoring_active(False)  # We're not on the monitoring page during breaks
    return templates.TemplateResponse("break.html", {"request": request})