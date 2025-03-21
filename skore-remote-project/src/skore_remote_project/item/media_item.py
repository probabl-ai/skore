from __future__ import annotations

from enum import Enum
from typing import Any

from .item import Item, ItemTypeError


class MediaType(Enum):
    """A media type of a string."""

    HTML = "text/html"
    MARKDOWN = "text/markdown"
    SVG = "image/svg+xml"


class MediaItem(Item):
    def __init__(self, media: str, media_type: str):
        self.media = media
        self.media_type = media_type

    @property
    def __raw__(self) -> Any:
        return self.media

    @property
    def __representation__(self) -> dict:
        return {
            "representation": {
                "media_type": self.media_type,
                "value": self.media,
            }
        }

    @classmethod
    def factory(
        cls, media: str, /, media_type: str = MediaType.MARKDOWN.value
    ) -> MediaItem:
        if not isinstance(media, str):
            raise ItemTypeError(f"Type '{media.__class__}' is not supported.")

        # | Before Python 3.12, a TypeError is raised if a non-Enum-member is used in a
        # | containment check.
        #
        # https://docs.python.org/3.12/library/enum.html#enum.EnumType.__contains__
        if media_type not in MediaType._value2member_map_:
            raise ValueError(f"MIME type '{media_type}' is not supported.")

        return cls(media, media_type)
