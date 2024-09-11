"""The definition of API routes to list stores and get them."""

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from skore.project import Project

SKORES_ROUTER = APIRouter(prefix="/skores", deprecated=True)
STORES_ROUTER = APIRouter(prefix="/stores")


def __serialize_project(project: Project):
    serialized = {}
    for key in project.list_keys():
        item = project.get_item(key)
        serialized[key] = {
            "item_type": str(item.item_type),
            "media_type": item.media_type,
            "serialized": json.loads(item.serialized),
        }

    return serialized


@SKORES_ROUTER.get("")
@SKORES_ROUTER.get("/")
async def get_items(request: Request):
    """Serialize a project and send it."""
    project = request.app.state.project
    return __serialize_project(project)


STATIC_FILES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "dashboard" / "static"
)


class LayoutItem(BaseModel):
    """A layout item."""

    key: str
    size: str


@SKORES_ROUTER.post("/share")
@SKORES_ROUTER.post("/share/")
async def share_store(request: Request, layout: list[LayoutItem]):
    """Serve an inlined shareable HTML page."""
    project = request.app.state.project

    # Get static assets to inject them into the report template
    def read_asset_content(path):
        with open(STATIC_FILES_PATH / path) as f:
            return f.read()

    script_content = read_asset_content("skore.umd.cjs")
    styles_content = read_asset_content("style.css")

    # Fill the Jinja context
    context = {
        "project": __serialize_project(project),
        "layout": [{"key": item.key, "size": item.size} for item in layout],
        "script": script_content,
        "styles": styles_content,
    }

    # Render the template and send the result
    templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")
    return templates.TemplateResponse(
        request=request, name="share.html.jinja", context=context
    )
