from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Pages"])
templates = Jinja2Templates(directory="templates")

@router.get("/watch", response_class=HTMLResponse)
def watch_feed(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/break_page", response_class=HTMLResponse)
def break_page(request: Request):
    """Page to redirect to when user is distracted for too long"""
    return templates.TemplateResponse("break.html", {"request": request})
