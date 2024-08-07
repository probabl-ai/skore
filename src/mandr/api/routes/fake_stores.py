"""The definition of API routes to get fake store."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

FAKE_MANDRS_ROUTER = APIRouter(prefix="/fake-mandrs", deprecated=True)
FAKE_STORES_ROUTER = APIRouter(prefix="/fake-stores")


@FAKE_MANDRS_ROUTER.get("/{uri:path}", response_class=FileResponse)
@FAKE_STORES_ROUTER.get("/{uri:path}", response_class=FileResponse)
async def get_fake_store_by_uri(uri: str):
    """Route used to get the fake store, regardless of its URI."""
    return Path(__file__).parent / "fake_store.json"
