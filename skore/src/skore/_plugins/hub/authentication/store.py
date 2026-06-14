"""Local cache for an interactive ``skore hub`` device-flow token.

This persists **only** the interactive OAuth token so that a separate process
(opencode, via ``skore agent init``) can reuse it without re-authenticating. The
**API key is never stored here** -- it is user-managed through the
``SKORE_HUB_API_KEY`` environment variable, exactly like the Python
authentication (``skore._plugins.hub.authentication.apikey``).

The token is stored as JSON at ``<user_config_dir>/skore/hub.json`` with
``0600`` permissions. The location can be overridden with the
``SKORE_HUB_CREDENTIALS`` environment variable (useful for tests/CI).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import TypedDict, cast

import platformdirs


class Token(TypedDict, total=False):
    """A persisted interactive device-flow token (never an API key)."""

    uri: str
    access_token: str
    refresh_token: str
    expires_at: str


def path() -> Path:
    """Return the path to the token file (honoring the env override)."""
    override = os.environ.get("SKORE_HUB_CREDENTIALS")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_config_dir("skore")) / "hub.json"


def load() -> Token | None:
    """Return the persisted token, or ``None`` if absent/unreadable."""
    file = path()
    if not file.is_file():
        return None
    try:
        data = json.loads(file.read_text() or "{}")
    except (OSError, json.JSONDecodeError):
        return None
    return cast(Token, data) if isinstance(data, dict) else None


def save(token: Token) -> Path:
    """Persist ``token`` to disk with owner-only permissions."""
    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(token, indent=2) + "\n")
    os.chmod(file, stat.S_IRUSR | stat.S_IWUSR)
    return file


def clear() -> Path | None:
    """Remove the token file; return its path if one existed."""
    file = path()
    if file.is_file():
        file.unlink()
        return file
    return None
