"""A FastAPI based webapp to serve a local dashboard."""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mandr.infomander import InfoManderRepository

_DASHBOARD_PATH = Path(__file__).resolve().parent
_STATIC_PATH = _DASHBOARD_PATH / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=_STATIC_PATH), name="static")
templates = Jinja2Templates(directory=_DASHBOARD_PATH / "templates")


def _build_common_context():
    """Return a dict object with element that are needed in all templates."""
    debug = os.getenv("DEBUG", "no").lower() in [
        "true",
        "1",
    ]
    return {"DEBUG": debug}


@app.get("/")
async def index(request: Request):
    """Serve the dashboard index."""
    context = _build_common_context() | {}
    return templates.TemplateResponse(
        request=request, name="index.html.jinja", context=context
    )


@app.get("/mandrs")
async def list_mandrs(request: Request) -> list[str]:
    """Send the list of mandrs path below the current working directory."""
    return [f"{p}" for p in InfoManderRepository.get_all_paths()]


@app.get("/mandrs/{path:path}")
async def get_mandr(request: Request):
    """Return on mandr."""
    return ""
