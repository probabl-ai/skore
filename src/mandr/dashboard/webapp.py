"""A FastAPI based webapp to serve a local dashboard."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mandr import registry
from mandr.api import schema
from mandr.storage import URI, FileSystem
from mandr.store import Store

_DASHBOARD_PATH = Path(__file__).resolve().parent
_STATIC_PATH = _DASHBOARD_PATH / "static"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/mandrs", deprecated=True)
@app.get("/api/stores")
async def list_mandrs(request: Request) -> list[str]:
    """Send the list of mandrs path below the current working directory."""
    directory = Path(os.environ["MANDR_ROOT"]).resolve()
    storage = FileSystem(directory=directory)

    return sorted(str(store.uri) for store in registry.stores(storage))


@app.get("/api/mandrs/{uri:path}", deprecated=True)
@app.get("/api/stores/{uri:path}")
async def get_mandr(request: Request, uri: str):
    """Return one mandr."""
    directory = Path(os.environ["MANDR_ROOT"]).resolve()
    storage = FileSystem(directory=directory)
    uri = URI(uri)

    if uri not in (key.parent for key in storage):
        raise HTTPException(status_code=404, detail=f"No store found in '{uri}'")

    store = Store(uri, storage=storage)
    model = schema.Store(
        uri=str(uri),
        payload={
            key: {
                "type": str(metadata["display_type"]),
                "data": value,
            }
            for key, value, metadata in store.items(metadata=True)
        },
    )

    return model.model_dump(by_alias=True)


@app.get("/api/fake-mandrs/{path:path}", response_class=FileResponse)
async def get_fake_mandr(request: Request, path: str):
    """Return a fake mandr."""
    return _DASHBOARD_PATH / "fixtures" / "mock-mander.json"


# as we mount / this line should be after all route declarations
app.mount("/", StaticFiles(directory=_STATIC_PATH, html=True), name="static")
