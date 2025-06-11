"""Logout from ``skore hub``."""

from __future__ import annotations

from .token import Token


def logout():
    """Logout from ``skore hub`` by deleting tokens."""
    if Token.exists():
        Token.filepath().unlink()
