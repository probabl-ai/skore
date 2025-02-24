"""The definition of API routes to list project items and get them."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import ORJSONResponse

from skore.persistence.item import Item
from skore.project.project import Project
from skore.ui.serializers import item_as_serializable

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
    version: int


def __item_as_serializable(name: str, item: Item, version: int) -> SerializableItem:
    d = item_as_serializable(item)
    return SerializableItem(
        name=name,
        media_type=d.get("media_type"),
        value=d.get("value"),
        updated_at=d.get("updated_at"),
        created_at=d.get("created_at"),
        note=d.get("note"),
        version=version,
    )


@router.get("/activity", response_class=ORJSONResponse)
async def get_activity(
    request: Request,
    after: datetime = datetime(1, 1, 1, 0, 0, 0, 0, timezone.utc),
) -> list[SerializableItem]:
    """Send all recent activity as a JSON array.

    The activity is composed of all the items and their versions created after the
    datetime `after`, sorted from newest to oldest.
    """
    project: Project = request.app.state.project
    return sorted(
        (
            __item_as_serializable(key, version, index)
            for key in project._item_repository
            for index, version in enumerate(
                project._item_repository.get_item_versions(key)
            )
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
    project: Project = request.app.state.project
    project.set_note(key=payload.key, note=payload.message, version=payload.version)
    return {"result": "ok"}


@router.get("/info", response_class=ORJSONResponse)
async def get_info(request: Request):
    """Get the name and path of the current project."""
    project: Project = request.app.state.project
    return {
        "name": project.name,
        "path": project.path,
    }
