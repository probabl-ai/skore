"""The definition of API routes to list project items and get them."""

import base64
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Request
from fastapi.params import Depends
from fastapi.templating import Jinja2Templates

from skore.item.media_item import MediaItem
from skore.item.numpy_array_item import NumpyArrayItem
from skore.item.pandas_dataframe_item import PandasDataFrameItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem
from skore.project import Project
from skore.view.view import Layout, View

from .dependencies import get_static_path, get_templates

router = APIRouter()


@dataclass
class SerializedItem:
    """Serialized item."""

    media_type: str
    value: Any
    updated_at: str
    created_at: str


@dataclass
class SerializedProject:
    """Serialized project, to be sent to the frontend."""

    layout: Layout
    views: dict[str, Layout]
    items: dict[str, SerializedItem]


def __serialize_project(project: Project) -> SerializedProject:
    views = {}
    for key in project.list_view_keys():
        view = project.get_view(key)

        views[key] = view.layout

    items = {}
    for key in project.list_item_keys():
        item = project.get_item(key)

        media_type = None
        if isinstance(item, PrimitiveItem):
            value = item.primitive
            media_type = "text/markdown"
        elif isinstance(item, NumpyArrayItem):
            value = item.array_list
            media_type = "text/markdown"
        elif isinstance(item, PandasDataFrameItem):
            value = item.dataframe_dict
            media_type = "application/vnd.dataframe+json"
        elif isinstance(item, SklearnBaseEstimatorItem):
            value = item.estimator_html_repr
            media_type = "application/vnd.sklearn.estimator+html"
        elif isinstance(item, MediaItem):
            value = base64.b64encode(item.media_bytes).decode()
            media_type = item.media_type
        else:
            raise ValueError(f"Item {item} is not a known item type.")

        items[key] = SerializedItem(
            media_type=media_type,
            value=value,
            updated_at=item.updated_at,
            created_at=item.created_at,
        )

    try:
        layout = project.get_view("layout").layout
    except KeyError:
        layout = []

    return SerializedProject(
        layout=layout,
        views=views,
        items=items,
    )


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

    # Get static assets to inject them into the view template
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
async def set_view_layout(request: Request, layout: Layout):
    """Set the view layout."""
    project: Project = request.app.state.project

    view = View(layout=layout)
    project.put_view("layout", view)

    return __serialize_project(project)
