"""Logout from ``skore hub``."""

from __future__ import annotations

from .token import Token


def logout():
    """Logout from ``skore hub`` by deleting tokens."""
    if (token := Token()).valid:
        token.filepath.unlink()
