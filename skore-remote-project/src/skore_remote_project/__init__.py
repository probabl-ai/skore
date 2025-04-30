"""Package that provides APIs to communicate between ``skore`` and ``skore hub``."""

from rich.console import Console
from rich.theme import Theme

from .project.project import Project

__all__ = ["Project", "console"]


console = Console(
    width=88,
    theme=Theme(
        {
            "repr.str": "cyan",
            "rule.line": "orange1",
            "repr.url": "orange1",
        }
    ),
)
