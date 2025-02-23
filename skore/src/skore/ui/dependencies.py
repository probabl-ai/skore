"""Declare all injectable dependencies."""

from pathlib import Path

__UI_MODULE_PATH = Path(__file__).resolve().parent


def get_static_path() -> Path:
    """Injectable static path."""
    return __UI_MODULE_PATH / "static"
