"""Provide the Store class for managing information in a data science project."""

import logging
import typing

import rich.logging

from skore.project import Project, load

__all__ = [
    "load",
    "Project",
]


class Handler(rich.logging.RichHandler):
    """A logging handler that renders output with Rich."""

    def get_level_text(self, record: rich.logging.LogRecord) -> rich.logging.Text:
        """Get the logger name and levelname from the record."""
        levelname = record.levelname
        name = record.name
        txt = f"<Logger {name} ({levelname})>"

        return rich.logging.Text.styled(
            txt.ljust(8),
            f"logging.level.{levelname.lower()}",
        )


formatter = logging.Formatter("%(message)s")
handler = Handler(markup=True)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def lazy_is_instance(object: typing.Any, cls_fullname: str):
    return cls_fullname in {
        f"{cls.__module__}.{cls.__name__}" for cls in object.__class__.__mro__
    }
