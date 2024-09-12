"""The definition of API routes to list project items and get them."""

import base64
import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Request
from fastapi.params import Depends
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from skore.project import ItemType, PersistedItem, Project

from .dependencies import get_static_path, get_templates

router = APIRouter()


class LayoutItem(BaseModel):
    """A layout item."""

    key: str
    size: str


def __serialize_project(project: Project) -> dict[str, PersistedItem]:
    serialized = {}
    for key in project.list_keys():
        item = project.get_item(key)
        if item.item_type == ItemType.MEDIA:
            data = base64.b64encode(item.serialized.encode()).decode()
        else:
            data = json.loads(item.serialized)

        serialized[key] = PersistedItem(
            item_type=item.item_type,
            media_type=item.media_type,
            serialized=data,
        )

    return serialized


@router.get("/items")
async def get_items(request: Request):
    """Serialize a project and send it."""
    project = request.app.state.project
    return __serialize_project(project)


@router.post("/report/share")
async def share_store(
    request: Request,
    layout: list[LayoutItem],
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
    static_path: Annotated[Path, Depends(get_static_path)],
):
    """Serve an inlined shareable HTML page."""
    project = request.app.state.project

    # Get static assets to inject them into the report template
    def read_asset_content(filename: str):
        with open(static_path / filename) as f:
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
    return templates.TemplateResponse(
        request=request, name="share.html.jinja", context=context
    )
