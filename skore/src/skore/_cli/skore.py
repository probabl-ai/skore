"""Resolve a Hub URL, including one published by a frontend."""

from __future__ import annotations

import json
import os
import urllib.request
from urllib.error import HTTPError, URLError

from skore._plugins.hub.authentication import HOST_ENV_VAR_NAME


def _discover_api_url(url: str) -> str | None:
    """Return the API URL published at ``url/.well-known/skore-hub.json``."""
    discovery_url = url.rstrip("/") + "/.well-known/skore-hub.json"
    try:
        with urllib.request.urlopen(discovery_url, timeout=5) as response:
            if response.status == 200:
                api_url = json.loads(response.read().decode()).get("api_url")
                if api_url:
                    return api_url
    except HTTPError as error:
        # ``urlopen`` raises before the ``with`` block owns the response.
        error.close()
    except (URLError, OSError, ValueError):
        pass
    return None


def apply_hub_uri(hub_url: str, *, discover: bool = False) -> str:
    """Record ``hub_url`` in ``SKORE_HUB_URI`` and return the recorded URL.

    Parameters
    ----------
    hub_url : str
        Hub URL to record. A value loaded from ``.skore`` is already resolved.
    discover : bool, default=False
        When True, a frontend that serves ``/.well-known/skore-hub.json``
        contributes its ``api_url`` instead of ``hub_url``.
    """
    if discover:
        hub_url = _discover_api_url(hub_url) or hub_url
    os.environ[HOST_ENV_VAR_NAME] = hub_url
    return hub_url


def resolve_hub_uri(hub_url: str | None) -> str:
    """Resolve ``hub_url`` and return the canonical hub URI.

    An explicit URL is written to ``SKORE_HUB_URI``. A frontend that serves
    ``/.well-known/skore-hub.json`` contributes its ``api_url`` instead.
    When ``hub_url`` is omitted, ``SKORE_HUB_URI`` is used.
    """
    if hub_url:
        apply_hub_uri(hub_url, discover=True)

    host = os.environ.get(HOST_ENV_VAR_NAME)
    if not host:
        raise ValueError(
            "You must provide an host via the parameter `host` or the environment "
            f"variable `{HOST_ENV_VAR_NAME}`"
        )

    from skore._plugins.hub.authentication.host import normalize

    return normalize(host)
