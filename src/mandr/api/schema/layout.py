"""Models to strope reports UI state."""

from enum import StrEnum, auto

from pydantic import BaseModel


class LayoutItemSizeEnum(StrEnum):
    """String enum to store report card size."""

    small = auto()
    medium = auto()
    large = auto()


class LayoutItem(BaseModel):
    """A Report display some of its items.

    This class represents displayed item setting.
    """

    key: str
    size: LayoutItemSizeEnum
