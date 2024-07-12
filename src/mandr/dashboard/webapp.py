"""A FastAPI based webapp to serve a local dashboard."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from mandr.infomander import ARTIFACTS_KEY, LOGS_KEY, VIEWS_KEY, InfoManderRepository

_DASHBOARD_PATH = Path(__file__).resolve().parent
_STATIC_PATH = _DASHBOARD_PATH / "static"

app = FastAPI()


@app.get("/api/mandrs")
async def list_mandrs(request: Request) -> list[str]:
    """Send the list of mandrs path below the current working directory."""
    return [f"{p}" for p in InfoManderRepository.get_all_paths()]


@app.get("/api/mandrs/{mander_path:path}")
async def get_mandr(request: Request, mander_path: str):
    """Return one mandr."""
    mander = InfoManderRepository.get(path=mander_path)
    if mander is None:
        raise HTTPException(status_code=404, detail=f"No mandr found in {mander_path}")
    serialized_mander = {
        "path": f"{mander.path}",
        "views": mander[VIEWS_KEY].items(),
        "logs": mander[LOGS_KEY].items(),
        "artifacts": mander[ARTIFACTS_KEY].items(),
        "info": {k: str(v) for k, v in mander.fetch().items() if not k.startswith("_")},
    }
    return serialized_mander


# as we mount / this line should be after all route declarations
app.mount("/", StaticFiles(directory=_STATIC_PATH, html=True), name="static")
