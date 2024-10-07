"""Declare all injectable dependencies."""

from pathlib import Path

from fastapi.templating import Jinja2Templates

__UI_MODULE_PATH = Path(__file__).resolve().parent


def get_templates():
    """Injectable template engine."""
    return Jinja2Templates(directory=__UI_MODULE_PATH / "templates")


def get_static_path():
    """Injectable static path."""
    return __UI_MODULE_PATH / "static"
