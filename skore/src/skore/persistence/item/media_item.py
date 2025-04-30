"""MediaItem.

This module defines the MediaItem class, used to persist media.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from .item import Item, ItemTypeError


def lazy_is_instance(object: Any, cls_fullname: str) -> bool:
    """Return True if object is an instance of `cls_fullname`."""
    return cls_fullname in {
        f"{cls.__module__}.{cls.__name__}" for cls in object.__class__.__mro__
    }


class MediaType(Enum):
    """A media type of a string."""

    HTML = "text/html"
    MARKDOWN = "text/markdown"
    SVG = "image/svg+xml"


class MediaItem(Item):
    """A class used to persist media."""

    def __init__(
        self,
        media: str,
        media_type: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a MediaItem.

        Parameters
        ----------
        media : str
            The media content.
        media_type : str
            The MIME type of the media content.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.media = media
        self.media_type = media_type

    @classmethod
    def factory(
        cls,
        media: str,
        /,
        media_type: str = MediaType.MARKDOWN.value,
        **kwargs,
    ) -> MediaItem:
        """
        Create a new MediaItem instance from a string.

        Parameters
        ----------
        media : str
            The media content.
        media_type : str
            The MIME type of the media content.

        Returns
        -------
        MediaItem
            A new MediaItem instance.
        """
        if not isinstance(media, str):
            raise ItemTypeError(f"Type '{media.__class__}' is not supported.")

        # | Before Python 3.12, a TypeError is raised if a non-Enum-member is used in a
        # | containment check.
        #
        # https://docs.python.org/3.12/library/enum.html#enum.EnumType.__contains__
        if media_type not in MediaType._value2member_map_:
            raise ValueError(f"MIME type '{media_type}' is not supported.")

        return cls(media, media_type, **kwargs)
