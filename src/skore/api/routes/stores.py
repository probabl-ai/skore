"""The definition of API routes to list stores and get them."""

import json

from fastapi import APIRouter, Request

SKORES_ROUTER = APIRouter(prefix="/skores", deprecated=True)
STORES_ROUTER = APIRouter(prefix="/stores")


@SKORES_ROUTER.get("")
@SKORES_ROUTER.get("/")
async def get_items(request: Request):
    """Serialize a project and send it."""
    project = request.app.state.project

    # Serialize project
    serialized = {}
    for key in project.list_keys():
        item = project.get_item(key)
        serialized[key] = {
            "item_type": str(item.item_type),
            "media_type": item.media_type,
            "serialized": json.loads(item.serialized),
        }

    return serialized
