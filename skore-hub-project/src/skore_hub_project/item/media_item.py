"""
MediaItem.

This module defines the ``MediaType`` class used to serialize string media.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from .item import Item, ItemTypeError


class MediaType(Enum):  # noqa: D101
    HTML = "text/html"
    MARKDOWN = "text/markdown"
    SVG = "image/svg+xml"


class MediaItem(Item):
    """Serialize string media."""

    def __init__(self, media: str, media_type: str):
        """
        Initialize a ``MediaItem``.

        Parameters
        ----------
        media : str
            The media to serialize.
        media_type : str
            The MIME type of the media content.
        """
        self.media = media
        self.media_type = media_type

    @property
    def __raw__(self) -> Any:
        """Get the value from the ``MediaItem`` instance."""
        return self.media

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``MediaItem`` instance."""
        return {"representation": {"media_type": self.media_type, "value": self.media}}

    @classmethod
    def factory(
        cls,
        media: str,
        /,
        media_type: str = MediaType.MARKDOWN.value,
    ) -> MediaItem:
        """
        Create a new ``MediaItem``.

        Parameters
        ----------
        media : str
            The media to serialize.
        media_type : str
            The MIME type of the media content.

        Returns
        -------
        MediaItem
            A new ``MediaItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``media`` is not an instance of ``str``.
        ValueError
            If ``media_type`` is not valid.
        """
        if not isinstance(media, str):
            raise ItemTypeError(f"Type '{media.__class__}' is not supported.")

        # | Before Python 3.12, a TypeError is raised if a non-Enum-member is used in a
        # | containment check.
        #
        # https://docs.python.org/3.12/library/enum.html#enum.EnumType.__contains__
        if media_type not in MediaType._value2member_map_:
            raise ValueError(f"MIME type '{media_type}' is not supported.")

        return cls(media, media_type)
