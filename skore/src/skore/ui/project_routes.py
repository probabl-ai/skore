"""The definition of API routes to list project items and get them."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from skore.persistence.item import Item
from skore.persistence.view.view import Layout, View
from skore.project import Project

router = APIRouter(prefix="/project")


@dataclass
class SerializableItem:
    """Serialized item."""

    name: str
    media_type: str
    value: Any
    updated_at: str
    created_at: str
    note: str


@dataclass
class SerializableProject:
    """Serialized project, to be sent to the skore-ui."""

    items: dict[str, list[SerializableItem]]
    views: dict[str, Layout]


def __item_as_serializable(name: str, item: Item) -> SerializableItem:
    d = item.as_serializable_dict()
    return SerializableItem(
        name=name,
        media_type=d.get("media_type"),
        value=d.get("value"),
        updated_at=d.get("updated_at"),
        created_at=d.get("created_at"),
        note=d.get("note"),
    )


def __project_as_serializable(project: Project) -> SerializableProject:
    items = {
        key: [
            __item_as_serializable(key, item)
            for item in project.item_repository.get_item_versions(key)
        ]
        for key in project.item_repository
    }

    views = {
        key: project.view_repository.get_view(key).layout
        for key in project.view_repository
    }

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
    project.view_repository.put_view(key, view)

    return __project_as_serializable(project)


@router.delete("/views", status_code=status.HTTP_202_ACCEPTED)
async def delete_view(request: Request, key: str):
    """Delete the view corresponding to `key`."""
    project: Project = request.app.state.project

    try:
        project.view_repository.delete_view(key)
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
    """Send all recent activity as a JSON array.

    The activity is composed of all the items and their versions created after the
    datetime `after`, sorted from newest to oldest.
    """
    project = request.app.state.project
    return sorted(
        (
            __item_as_serializable(key, version)
            for key in project.item_repository
            for version in project.item_repository.get_item_versions(key)
            if datetime.fromisoformat(version.updated_at) > after
        ),
        key=operator.attrgetter("updated_at"),
        reverse=True,
    )


@dataclass
class NotePayload:
    """Represent a note and the wanted item to attach to."""

    key: str
    message: str
    version: int = -1


@router.put("/note", status_code=status.HTTP_201_CREATED)
async def set_note(request: Request, payload: NotePayload):
    """Add a note to the given item."""
    project = request.app.state.project
    project.set_note(key=payload.key, note=payload.message, version=payload.version)
    return __project_as_serializable(project)
