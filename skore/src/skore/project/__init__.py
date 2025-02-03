"""Alias top level function and class of the project submodule."""

from ._open import open
from .project import Project

__all__ = [
    "open",
    "Project",
]
