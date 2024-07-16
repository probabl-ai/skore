"""A FastAPI based webapp to serve a local dashboard."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mandr import InfoMander

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


@app.get("/api/mandrs")
async def list_mandrs(request: Request) -> list[str]:
    """Send the list of mandrs path below the current working directory."""
    path = os.environ["MANDR_PATH"]
    root = Path(os.environ["MANDR_ROOT"])
    ims = [InfoMander(path, root=root)]
    paths = []

    # Use `ims` as a queue to recursively iterate over children to retrieve path.
    for im in ims:
        ims[len(ims) :] = im.children()
        absolute_path = im.project_path
        relative_path = absolute_path.relative_to(root)

        paths.append(str(relative_path))

    return sorted(paths)


@app.get("/api/mandrs/{path:path}")
async def get_mandr(request: Request, path: str):
    """Return one mandr."""
    root = Path(os.environ["MANDR_ROOT"])

    if not (root / path).exists():
        raise HTTPException(status_code=404, detail=f"No mandr found in '{path}'")

    im = InfoMander(path, root=root)

    return {
        "path": path,
        "views": im[InfoMander.VIEWS_KEY].items(),
        "logs": im[InfoMander.LOGS_KEY].items(),
        "artifacts": im[InfoMander.ARTIFACTS_KEY].items(),
        "info": {
            key: str(value)
            for key, value in im.fetch().items()
            if key not in InfoMander.RESERVED_KEYS
        },
    }


# as we mount / this line should be after all route declarations
app.mount("/", StaticFiles(directory=_STATIC_PATH, html=True), name="static")
