"""The definition of API routes to list project items and get them."""

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Request
from fastapi.params import Depends
from fastapi.templating import Jinja2Templates

from skore.layout import Layout
from skore.project import ItemType, PersistedItem, Project

from .dependencies import get_static_path, get_templates

router = APIRouter()


@dataclass
class SerializedProject:
    """Serialized project, to be sent to the frontend."""

    layout: Layout
    items: dict[str, PersistedItem]


def __serialize_project(project: Project) -> SerializedProject:
    try:
        layout = project.get_report_layout()
    except KeyError:
        layout = []

    items = {}
    for key in project.list_keys():
        item = project.get_item(key)
        if item.item_type == ItemType.MEDIA:
            data = base64.b64encode(item.serialized.encode()).decode()
        else:
            data = json.loads(item.serialized)

        items[key] = PersistedItem(
            item_type=item.item_type,
            media_type=item.media_type,
            serialized=data,
            updated_at=item.updated_at.isoformat(),
            created_at=item.created_at.isoformat(),
        )

    return SerializedProject(layout=layout, items=items)


@router.get("/items")
async def get_items(request: Request):
    """Serialize a project and send it."""
    project = request.app.state.project
    return __serialize_project(project)


@router.post("/report/share")
async def share_store(
    request: Request,
    layout: Layout,
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
        "project": asdict(__serialize_project(project)),
        "layout": [{"key": item.key, "size": item.size} for item in layout],
        "script": script_content,
        "styles": styles_content,
    }

    # Render the template and send the result
    return templates.TemplateResponse(
        request=request, name="share.html.jinja", context=context
    )


@router.put("/report/layout", status_code=201)
async def set_report_layout(request: Request, layout: Layout):
    """Set the report layout."""
    project = request.app.state.project
    project.put_report_layout(layout)
    return __serialize_project(project)
