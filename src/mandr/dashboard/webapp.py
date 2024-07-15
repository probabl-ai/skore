"""A FastAPI based webapp to serve a local dashboard."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mandr import InfoMander

_DASHBOARD_PATH = Path(__file__).resolve().parent
_STATIC_PATH = _DASHBOARD_PATH / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=_STATIC_PATH), name="static")
templates = Jinja2Templates(directory=_DASHBOARD_PATH / "templates")


def _build_common_context():
    """Return a dict object with element that are needed in all templates."""
    debug = os.getenv("DEBUG", "no").lower() in [
        "true",
        "1",
    ]
    return {"DEBUG": debug}


@app.get("/")
async def index(request: Request):
    """Serve the dashboard index."""
    context = _build_common_context() | {}
    return templates.TemplateResponse(
        request=request, name="index.html.jinja", context=context
    )


@app.get("/mandrs")
async def list_mandrs(request: Request) -> list[str]:
    """Send the list of mandrs path below the current working directory."""
    path = os.environ["MANDR_PATH"]
    root = Path(os.environ["MANDR_ROOT"])
    ims = [InfoMander(path, root=root)]
    paths = []

    for im in ims:
        ims[len(ims) :] = im.children()
        absolute_path = im.project_path
        relative_path = absolute_path.relative_to(root)

        paths.append(str(relative_path))

    return sorted(paths)


@app.get("/mandrs/{path:path}")
async def get_mandr(request: Request, path: str):
    """Return one mandr."""
    root = Path(os.environ["MANDR_ROOT"])

    if not (Path(root) / path).exists():
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
