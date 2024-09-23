"""Layout models."""

from dataclasses import dataclass
from enum import StrEnum


class LayoutItemSize(StrEnum):
    """The size of a layout item."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class LayoutItem:
    """A layout item."""

    key: str
    size: LayoutItemSize


Layout = list[LayoutItem]
