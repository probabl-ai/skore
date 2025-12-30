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


def persist(uri: str) -> None:
    """Persist the URI."""
    Filepath().write_text(uri)


def URI() -> str:
    """
    URI used for ``skore hub`` authentication.

    Notes
    -----
    It is discovered by following the rules in order:

    1. If you have setup an API key using the envar ``SKORE_HUB_API_KEY``, returns the
       value of the envar ``SKORE_HUB_URI``, or ``https://api.skore.probabl.ai``.

    2. If you have an active token, extract the content of the envar ``SKORE_HUB_URI``
       and the URI associated with the token:
        2.1. If both the URIs from the environment and the one associated with the token
             are empty, returns ``https://api.skore.probabl.ai``.
        2.2. If the URI from the environment is empty and the one associated with the
             token isn't, returns the URI associated with the token.
        2.3. If the URI associated with the token is empty, and the one from the
             environment isn't, returns the URI from the environment.
        2.4. If both the URIs associated with the token and the one from the
             environment are not empty:
                2.4.1. If both are equal, returns one of them.
                2.4.2. If both aren't equal, raises a conflicting exception.
    """
    uri_from_environment = environ.get(ENVARNAME)

    if environ.get("SKORE_HUB_API_KEY"):
        return uri_from_environment or DEFAULT

    filepath = Filepath()
    uri_from_persistence = filepath.read_text() if filepath.exists() else None

    if (
        uri_from_persistence
        and uri_from_environment
        and (uri_from_persistence != uri_from_environment)
    ):
        raise URIError(
            f"\nBad condition: the persisted URI is conflicting with the environment:\n"
            f"\tFrom {filepath}: '{uri_from_persistence}'\n"
            f"\tFrom ${ENVARNAME}: '{uri_from_environment}'\n"
            f"\nPlease run `skore-hub-logout`."
        )

    return uri_from_persistence or uri_from_environment or DEFAULT
