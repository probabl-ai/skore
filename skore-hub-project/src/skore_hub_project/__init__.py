"""Package that provides APIs to communicate between ``skore`` and ``skore hub``."""

from pydantic import BaseModel
from rich.console import Console
from rich.theme import Theme

from .project.project import Project

__all__ = ["Payload", "Project", "console"]


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


class Payload(BaseModel):
    class Config:
        frozen = True

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs, exclude_none=True)
