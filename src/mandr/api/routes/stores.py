"""The definition of API routes to list stores and get them."""

import os
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, status

from mandr import registry
from mandr.api import schema
from mandr.api.schema.layout import LayoutItem
from mandr.storage import URI, FileSystem
from mandr.store.store import Store, _get_storage_path

MANDRS_ROUTER = APIRouter(prefix="/mandrs", deprecated=True)
STORES_ROUTER = APIRouter(prefix="/stores")

# FIXME find a better to isolate layotu from users items
LAYOUT_KEY = "__mandr__layout__"


def serialize_store(store: Store):
    """Serialize a Store."""
    payload: dict = {}
    # mypy does not understand union in generator
    user_items: Iterable[tuple[str, Any, dict]] = filter(
        lambda i: i[0] != LAYOUT_KEY,
        store.items(metadata=True),  # type: ignore
    )
    for key, value, metadata in user_items:
        payload[key] = {
            "type": str(metadata["display_type"]),
            "data": value,
        }

    try:
        layout: list[LayoutItem] = store.read(LAYOUT_KEY)  # type: ignore
    except KeyError:
        layout: list[LayoutItem] = []

    model = schema.Store(
        schema="schema:dashboard:v0",
        uri=str(store.uri),
        payload=payload,
        layout=layout,
    )

    return model.model_dump(by_alias=True)


@MANDRS_ROUTER.get("")
@MANDRS_ROUTER.get("/")
@STORES_ROUTER.get("")
@STORES_ROUTER.get("/")
async def list_stores() -> list[str]:
    """Route used to list the URI of stores."""
    directory = _get_storage_path(os.environ.get("MANDR_ROOT"))
    storage = FileSystem(directory=directory)

    return sorted(str(store.uri) for store in registry.stores(storage))


@MANDRS_ROUTER.get("/{uri:path}")
@STORES_ROUTER.get("/{uri:path}")
async def get_store_by_uri(uri: str):
    """Route used to get a store by its URI."""
    directory = _get_storage_path(os.environ.get("MANDR_ROOT"))
    storage = FileSystem(directory=directory)
    store_uri = URI(uri)

    for store in registry.stores(storage):
        if store_uri == store.uri:
            return serialize_store(store)

    raise HTTPException(status_code=404, detail=f"No store found in '{uri}'")


@MANDRS_ROUTER.put("/{uri:path}/layout", status_code=status.HTTP_201_CREATED)
@STORES_ROUTER.put("/{uri:path}/layout")
async def put_layout(uri: str, payload: list[LayoutItem]):
    """Save the report layout configuration."""
    directory = _get_storage_path(os.environ.get("MANDR_ROOT"))
    storage = FileSystem(directory=directory)
    store_uri = URI(uri)

    for store in registry.stores(storage):
        if store_uri == store.uri:
            try:
                store.insert(LAYOUT_KEY, payload)
            except KeyError:
                store.update(LAYOUT_KEY, payload)
            return serialize_store(store)

    raise HTTPException(status_code=404, detail=f"No store found in '{uri}'")
