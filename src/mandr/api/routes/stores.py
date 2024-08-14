"""The definition of API routes to list stores and get them."""

import os

from fastapi import APIRouter, HTTPException

from mandr import registry
from mandr.api import schema
from mandr.storage import URI, FileSystem
from mandr.store.store import _get_storage_path

MANDRS_ROUTER = APIRouter(prefix="/mandrs", deprecated=True)
STORES_ROUTER = APIRouter(prefix="/stores")


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
    parsed_uri = URI(uri)

    for store in registry.stores(storage):
        if parsed_uri == store.uri:
            model = schema.Store(
                uri=uri,
                payload={
                    key: {
                        "type": str(metadata["display_type"]),
                        "data": value,
                    }
                    for key, value, metadata in store.items(metadata=True)
                },
            )

            return model.model_dump(by_alias=True)

    raise HTTPException(status_code=404, detail=f"No store found in '{uri}'")
