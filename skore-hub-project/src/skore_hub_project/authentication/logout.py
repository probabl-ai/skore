"""Logout from ``skore hub``."""

from . import token, uri


def logout() -> None:
    """Logout from ``skore hub`` by deleting the persisted token."""
    token.Filepath().unlink(missing_ok=True)
    uri.Filepath().unlink(missing_ok=True)
