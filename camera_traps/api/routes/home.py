from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from camera_traps.data import ClassifierClasses

home_router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@home_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    formatted_classes = [cls.value.replace("_", " ").title() for cls in ClassifierClasses]

    return templates.TemplateResponse(name="index.html", request=request, context={"classes": formatted_classes})
