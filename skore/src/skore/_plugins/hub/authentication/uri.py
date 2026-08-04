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

DEFAULT: Final[str] = "https://api.skore.probabl.ai"
ENV_VAR_NAME: Final[str] = "SKORE_HUB_URI"


def URI() -> str:
    """Hub backend URI used for ``skore hub`` authentication."""
    return environ.get(ENV_VAR_NAME, DEFAULT)
