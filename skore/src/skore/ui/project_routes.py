"""The definition of API routes to list project items and get them."""

import base64
import json
from traceback import format_exc

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from skore.item import (
    CrossValidationItem,
    Item,
    MediaItem,
    NumpyArrayItem,
    PandasDataFrameItem,
    PandasSeriesItem,
    PrimitiveItem,
    SklearnBaseEstimatorItem,
)
from skore.project import Project
from skore.view.view import Layout, View

router = APIRouter(prefix="/project")


class JSONError(Exception):
    """Exception for objects that can't be serialized."""


def default(_):
    """Raise `JSONError` for objects that can't be serialized."""
    raise JSONError


class ProjectResponse(JSONResponse):
    """Project response with our own serialization strategy."""

    @staticmethod
    def serialize_item_to_json(item: Item) -> str:
        """Serialize Item to JSON."""
        if isinstance(item, PrimitiveItem):
            value = item.primitive
            media_type = "text/markdown"
        elif isinstance(item, NumpyArrayItem):
            value = item.array.tolist()
            media_type = "text/markdown"
        elif isinstance(item, PandasDataFrameItem):
            value = item.dataframe.to_dict(orient="tight")
            media_type = "application/vnd.dataframe+json"
        elif isinstance(item, PandasSeriesItem):
            value = item.series.to_list()
            media_type = "text/markdown"
        elif isinstance(item, SklearnBaseEstimatorItem):
            value = item.estimator_html_repr
            media_type = "application/vnd.sklearn.estimator+html"
        elif isinstance(item, MediaItem):
            value = base64.b64encode(item.media_bytes).decode()
            media_type = item.media_type
        elif isinstance(item, CrossValidationItem):
            value = base64.b64encode(item.plot_bytes).decode()
            media_type = "application/vnd.plotly.v1+json"
        else:
            raise ValueError(f"Item {item} is not a known item type.")

        try:
            value = json.dumps(value, default=default)
        except JSONError:
            value = json.dumps(None)
            error = True
            traceback = format_exc()
        else:
            error = False
            traceback = None

        media_type = json.dumps(media_type)
        error = json.dumps(error)
        traceback = json.dumps(traceback)
        created_at = json.dumps(item.created_at)
        updated_at = json.dumps(item.updated_at)

        return (
            "{"
            f'"media_type": {media_type}, '
            f'"value": {value}, '
            f'"error": {error}, '
            f'"traceback": {traceback}, '
            f'"created_at": {created_at}, '
            f'"updated_at": {updated_at}'
            "}"
        )

    @staticmethod
    def serialize_project_to_json(project: Project) -> str:
        """Serialize Project to JSON."""
        # serialize items
        key_to_items_str = []

        for key in project.list_item_keys():
            items = []
            for item in project.get_item_versions(key):
                items.append(ProjectResponse.serialize_item_to_json(item))

            key_to_items_str.append(f'"{key}": [{", ".join(items)}]')

        key_to_items_str = f'{{{", ".join(key_to_items_str)}}}'

        # serialize views/layouts
        key_to_layout_str = []

        for key in project.list_view_keys():
            layout = project.get_view(key).layout
            layout = json.dumps(layout)

            key_to_layout_str.append(f'"{key}": {layout}')

        key_to_layout_str = f'{{{", ".join(key_to_layout_str)}}}'

        # serialize project
        return f'{{"items": {key_to_items_str}, "views": {key_to_layout_str}}}'

    def render(self, content: Project) -> bytes:
        """Render Project response."""
        return ProjectResponse.serialize_project_to_json(content).encode("utf-8")


@router.get("/items")
async def get_items(request: Request):
    """Serialize a project and send it."""
    return ProjectResponse(request.app.state.project)


@router.put("/views")
async def put_view(request: Request, key: str, layout: Layout):
    """Set the layout of the view corresponding to `key`.

    If the view corresponding to `key` does not exist, it will be created.
    """
    project: Project = request.app.state.project

    view = View(layout=layout)
    project.put_view(key, view)

    return ProjectResponse(project, status_code=status.HTTP_201_CREATED)


@router.delete("/views")
async def delete_view(request: Request, key: str):
    """Delete the view corresponding to `key`."""
    project: Project = request.app.state.project

    try:
        project.delete_view(key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="View not found"
        ) from None

    return ProjectResponse(project, status_code=status.HTTP_202_ACCEPTED)
