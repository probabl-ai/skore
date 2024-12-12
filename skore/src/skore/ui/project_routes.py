"""The definition of API routes to list project items and get them."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from itertools import starmap
from traceback import format_exc

from fastapi import APIRouter, HTTPException, Request, Response, status

from skore.item import (
    CrossValidationAggregationItem,
    CrossValidationItem,
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
from skore.view.view import Layout, View

router = APIRouter(prefix="/project")


class JSONError(Exception):
    """Exception for objects that can't be serialized."""


def default(_):
    """Raise `JSONError` for objects that can't be serialized."""
    raise JSONError


def pandas_dataframe_as_serializable(df):
    return df.fillna("NaN").to_dict(orient="tight")


def serialize_item_to_json(name: str, item: Item) -> str:
    """Serialize Item to JSON."""
    if isinstance(item, PrimitiveItem):
        value = item.primitive
        media_type = "text/markdown"
    elif isinstance(item, NumpyArrayItem):
        value = item.array.tolist()
        media_type = "text/markdown"
    elif isinstance(item, PandasDataFrameItem):
        value = pandas_dataframe_as_serializable(item.dataframe)
        media_type = "application/vnd.dataframe"
    elif isinstance(item, PandasSeriesItem):
        value = item.series.fillna("NaN").to_list()
        media_type = "text/markdown"
    elif isinstance(item, PolarsDataFrameItem):
        value = pandas_dataframe_as_serializable(item.dataframe.to_pandas())
        media_type = "application/vnd.dataframe"
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

    try:
        value = json.dumps(value, default=default)
    except JSONError:
        value = "null"
        error = "true"
        traceback = f'"{format_exc()}"'
    else:
        error = "false"
        traceback = "null"

    return (
        "{"
        f'"name": "{name}",'
        f'"media_type": "{media_type}",'
        f'"value": {value},'
        f'"error": {error},'
        f'"traceback": {traceback},'
        f'"created_at": "{item.created_at}",'
        f'"updated_at": "{item.updated_at}"'
        "}"
    )


@router.put("/views", status_code=status.HTTP_201_CREATED)
async def put_view(request: Request, key: str, layout: Layout):
    project = request.app.state.project
    project.put_view(key, View(layout=layout))


@router.delete("/views", status_code=status.HTTP_202_ACCEPTED)
async def delete_view(request: Request, key: str):
    """Delete the view corresponding to `key`."""
    project = request.app.state.project

    try:
        project.delete_view(key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="View not found",
        )


@router.get("/items")
async def get_items(request: Request):
    """Serialize Project to JSON."""
    project = request.app.state.project

    # serialize items
    key_to_items_str = []

    for key in project.list_item_keys():
        items = []
        for item in project.get_item_versions(key):
            items.append(serialize_item_to_json(key, item))
        key_to_items_str.append(f'"{key}": [{", ".join(items)}]')

    key_to_items_str = f'{{{", ".join(key_to_items_str)}}}'

    # serialize layouts
    key_to_layout_str = []

    for key in project.list_view_keys():
        layout = project.get_view(key).layout
        layout = json.dumps(layout)
        key_to_layout_str.append(f'"{key}": {layout}')

    key_to_layout_str = f'{{{", ".join(key_to_layout_str)}}}'

    # serialize project
    items = str.encode(
        f'{{"items": {key_to_items_str}, "views": {key_to_layout_str}}}',
        "utf-8",
    )

    return Response(content=items, media_type="application/json")


@router.get("/activity")
async def get_activity(
    request: Request,
    after: datetime = datetime(1, 1, 1, 0, 0, 0, 0, timezone.utc),
):
    """Send all recent activity as a JSON array.

    The activity is composed of all the items and their versions created after the
    datetime `after`, sorted from newest to oldest.
    """
    project = request.app.state.project
    versions = sorted(
        (
            (key, version)
            for key in project.list_item_keys()
            for version in project.get_item_versions(key)
            if datetime.fromisoformat(version.updated_at) > after
        ),
        key=lambda x: x[1].updated_at,
        reverse=True,
    )

    # serialize activity
    activity = str.encode(
        f'[{", ".join(starmap(serialize_item_to_json, versions))}]',
        "utf-8",
    )

    return Response(content=activity, media_type="application/json")
