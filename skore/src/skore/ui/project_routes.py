"""The definition of API routes to list project items and get them."""

import base64
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from skore.item.cross_validation_item import CrossValidationItem
from skore.item.media_item import MediaItem
from skore.item.numpy_array_item import NumpyArrayItem
from skore.item.pandas_dataframe_item import PandasDataFrameItem
from skore.item.pandas_series_item import PandasSeriesItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem
from skore.project import Project
from skore.view.view import Layout, View

router = APIRouter(prefix="/project")


@dataclass
class SerializedItem:
    """Serialized item."""

    media_type: str
    value: Any
    updated_at: str
    created_at: str


@dataclass
class SerializedProject:
    """Serialized project, to be sent to the skore-ui."""

    items: dict[str, SerializedItem]
    views: dict[str, Layout]


def __serialize_project(project: Project) -> SerializedProject:
    views = {}
    for key in project.list_view_keys():
        views[key] = project.get_view(key).layout

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
        elif isinstance(item, PandasSeriesItem):
            value = item.series_list
            media_type = "text/markdown"
        elif isinstance(item, SklearnBaseEstimatorItem):
            value = item.estimator_html_repr
            media_type = "application/vnd.sklearn.estimator+html"
        elif isinstance(item, MediaItem):
            value = base64.b64encode(item.media_bytes).decode()
            media_type = item.media_type
        elif isinstance(item, CrossValidationItem):
            # Convert plot to MediaItem
            item = MediaItem.factory(item.plot)
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

    return SerializedProject(
        items=items,
        views=views,
    )


@router.get("/items")
async def get_items(request: Request):
    """Serialize a project and send it."""
    project = request.app.state.project
    return __serialize_project(project)


@router.put("/views", status_code=status.HTTP_201_CREATED)
async def put_view(request: Request, key: str, layout: Layout):
    """Set the layout of the view corresponding to `key`.

    If the view corresponding to `key` does not exist, it will be created.
    """
    project: Project = request.app.state.project

    view = View(layout=layout)
    project.put_view(key, view)

    return __serialize_project(project)


@router.delete("/views", status_code=status.HTTP_202_ACCEPTED)
async def delete_view(request: Request, key: str):
    """Delete the view corresponding to `key`."""
    project: Project = request.app.state.project

    try:
        project.delete_view(key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="View not found"
        ) from None

    return __serialize_project(project)
