"""Alias top level function and class of the project submodule."""

from .create import create
from .load import load
from .open import open
from .project import Project

__all__ = [
    "create",
    "load",
    "open",
    "Project",
]
