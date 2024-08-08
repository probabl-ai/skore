"""The definition of API routes to list stores and get them."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from mandr import registry
from mandr.api import schema
from mandr.item import DisplayType
from mandr.storage import URI, FileSystem

MANDRS_ROUTER = APIRouter(prefix="/mandrs", deprecated=True)
STORES_ROUTER = APIRouter(prefix="/stores")


@MANDRS_ROUTER.get("")
@MANDRS_ROUTER.get("/")
@STORES_ROUTER.get("")
@STORES_ROUTER.get("/")
async def list_stores() -> list[str]:
    """Route used to list the URI of stores."""
    directory = Path(os.environ["MANDR_ROOT"]).resolve()
    storage = FileSystem(directory=directory)

    return sorted(str(store.uri) for store in registry.stores(storage))


@MANDRS_ROUTER.get("/{uri:path}")
@STORES_ROUTER.get("/{uri:path}")
async def get_store_by_uri(uri: str):
    """Route used to get a store by its URI."""
    directory = Path(os.environ["MANDR_ROOT"]).resolve()
    storage = FileSystem(directory=directory)
    uri = URI(uri)

    for store in registry.stores(storage):
        if uri == store.uri:
            payload = {}
            for key, value, metadata in store.items(metadata=True):
                if metadata["display_type"] == DisplayType.CROSS_VALIDATION_RESULTS:
                    payload[key] = {
                        "type": str(metadata["display_type"]),
                        "data": metadata["computed"],
                    }
                else:
                    payload[key] = {
                        "type": str(metadata["display_type"]),
                        "data": value,
                    }

            model = schema.Store(
                uri=str(uri),
                payload=payload,
            )

            return model.model_dump(by_alias=True)

    raise HTTPException(status_code=404, detail=f"No store found in '{uri}'")
