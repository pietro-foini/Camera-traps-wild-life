from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

home_router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@home_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Create homepage"""
    return templates.TemplateResponse(name="index.html", request=request)
