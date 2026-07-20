from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["pages"])
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html")
