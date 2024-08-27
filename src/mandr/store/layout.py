"""Models to store state of a report's UI."""

from enum import StrEnum, auto

from pydantic import BaseModel


class LayoutItemSize(StrEnum):
    """String enum to store report card size."""

    small = auto()
    medium = auto()
    large = auto()


class LayoutItem(BaseModel):
    """A Report display some of its items.

    This class represents displayed item setting.
    """

    key: str
    size: LayoutItemSize


Layout = list[LayoutItem]
