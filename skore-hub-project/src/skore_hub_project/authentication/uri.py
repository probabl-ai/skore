"""URI used for ``skore hub`` authentication."""

from os import environ
from pathlib import Path
from tempfile import gettempdir
from typing import Final

DEFAULT: Final[str] = "https://api.skore.probabl.ai"
ENVARNAME: Final[str] = "SKORE_HUB_URI"


class URIError(RuntimeError):
    """An exception dedicated to ``URI``."""


def Filepath() -> Path:
    """Filepath used to save the URI on disk."""
    return Path(gettempdir(), "skore.uri")


def persist(uri: str):
    """Persist the URI."""
    Filepath().write_text(uri)


def URI() -> str:
    """URI used for ``skore hub`` authentication."""
    filepath = Filepath()
    uri_from_persistence = filepath.read_text() if filepath.exists() else None
    uri_from_environment = environ.get(ENVARNAME)

    if (
        uri_from_persistence
        and uri_from_environment
        and (uri_from_persistence != uri_from_environment)
    ):
        raise URIError(
            f"\nBad condition: the persisted URI is conflicting with the environment:\n"
            f"\tFrom {filepath}: '{uri_from_persistence}'\n"
            f"\tFrom ${ENVARNAME}: '{uri_from_environment}'\n."
            f"Please run `skore-hub-logout`."
        )

    return uri_from_persistence or uri_from_environment or DEFAULT
