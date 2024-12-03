"""The definition of API routes to list project items and get them."""

from __future__ import annotations

import base64
import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request, status

from skore.item import (
    Item,
    MediaItem,
    NumpyArrayItem,
    PandasDataFrameItem,
    PandasSeriesItem,
    PolarsDataFrameItem,
    PolarsSeriesItem,
    PrimitiveItem,
    SklearnBaseEstimatorItem,
)
from skore.item.cross_validation_item import (
    CrossValidationAggregationItem,
    CrossValidationItem,
)
from skore.project import Project
from skore.view.view import Layout, View

if TYPE_CHECKING:
    import pandas  # type: ignore

router = APIRouter(prefix="/project")


@dataclass
class SerializableItem:
    """Serialized item."""

    name: str
    media_type: str
    value: Any
    updated_at: str
    created_at: str


@dataclass
class SerializableProject:
    """Serialized project, to be sent to the skore-ui."""

    items: dict[str, list[SerializableItem]]
    views: dict[str, Layout]


def __pandas_dataframe_as_serializable(df: pandas.DataFrame):
    return df.fillna("NaN").to_dict(orient="tight")


def __item_as_serializable(name: str, item: Item) -> SerializableItem:
    if isinstance(item, PrimitiveItem):
        value = item.primitive
        media_type = "text/markdown"
    elif isinstance(item, NumpyArrayItem):
        value = item.array.tolist()
        media_type = "text/markdown"
    elif isinstance(item, PandasDataFrameItem):
        value = __pandas_dataframe_as_serializable(item.dataframe)
        media_type = "application/vnd.dataframe+json"
    elif isinstance(item, PandasSeriesItem):
        value = item.series.fillna("NaN").to_list()
        media_type = "text/markdown"
    elif isinstance(item, PolarsDataFrameItem):
        value = __pandas_dataframe_as_serializable(item.dataframe.to_pandas())
        media_type = "application/vnd.dataframe+json"
    elif isinstance(item, PolarsSeriesItem):
        value = item.series.to_list()
        media_type = "text/markdown"
    elif isinstance(item, SklearnBaseEstimatorItem):
        value = item.estimator_html_repr
        media_type = "application/vnd.sklearn.estimator+html"
    elif isinstance(item, MediaItem):
        if "text" in item.media_type:
            value = item.media_bytes.decode(encoding=item.media_encoding)
            media_type = f"{item.media_type}"
        else:
            value = base64.b64encode(item.media_bytes).decode()
            media_type = f"{item.media_type};base64"
    elif isinstance(item, (CrossValidationItem, CrossValidationAggregationItem)):
        value = base64.b64encode(item.plot_bytes).decode()
        media_type = "application/vnd.plotly.v1+json;base64"
    else:
        raise ValueError(f"Item {item} is not a known item type.")

    return SerializableItem(
        name=name,
        media_type=media_type,
        value=value,
        updated_at=item.updated_at,
        created_at=item.created_at,
    )


def __project_as_serializable(project: Project) -> SerializableProject:
    items = {
        key: [
            __item_as_serializable(key, item) for item in project.get_item_versions(key)
        ]
        for key in project.list_item_keys()
    }

    views = {key: project.get_view(key).layout for key in project.list_view_keys()}

    return SerializableProject(
        items=items,
        views=views,
    )


@router.get("/items")
async def get_items(request: Request):
    """Serialize a project and send it."""
    project = request.app.state.project
    return __project_as_serializable(project)


@router.put("/views", status_code=status.HTTP_201_CREATED)
async def put_view(request: Request, key: str, layout: Layout):
    """Set the layout of the view corresponding to `key`.

    If the view corresponding to `key` does not exist, it will be created.
    """
    project: Project = request.app.state.project

    view = View(layout=layout)
    project.put_view(key, view)

    return __project_as_serializable(project)


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

    return __project_as_serializable(project)


@router.get("/activity")
async def get_activity(
    request: Request,
    after: datetime = datetime(1, 1, 1, 0, 0, 0, 0, timezone.utc),
):
    """Send all recent as a JSON array.

    The activity is composed of all the items and their versions created after the
    datetime `after`.
    """
    project = request.app.state.project
    return sorted(
        (
            __item_as_serializable(key, version)
            for key in project.list_item_keys()
            for version in project.get_item_versions(key)
            if datetime.fromisoformat(version.updated_at) > after
        ),
        key=operator.attrgetter("updated_at"),
        reverse=True,
    )
