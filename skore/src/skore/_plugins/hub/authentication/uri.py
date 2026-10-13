"""Hub backend URI used for ``skore hub`` authentication.

The ``SKORE_HUB_URI`` environment variable must be set to the hub's backend URI
(e.g. ``https://api.skore.probabl.ai``), not the frontend one
(e.g. ``https://skore.probabl.ai``).

When debugging with a local instance of the hub, the backend is usually served on
``http://127.0.0.1:8000``:

    import os; os.environ["SKORE_HUB_URI"] = "http://127.0.0.1:8000"
"""

from os import environ
from typing import Final
from urllib.parse import urlsplit, urlunsplit

DEFAULT: Final[str] = "https://api.skore.probabl.ai"
ENV_VAR_NAME: Final[str] = "SKORE_HUB_URI"


def normalize(uri: str) -> str:
    """Return ``uri`` with a stable scheme, host and path for credential lookup."""
    parts = urlsplit(uri.strip())

    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "")
    )


def URI() -> str:
    """Hub backend URI used for ``skore hub`` authentication."""
    return normalize(environ.get(ENV_VAR_NAME, DEFAULT))
