"""MediaItem.

This module defines the MediaItem class, used to persist media.
"""

from __future__ import annotations

import base64
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any, Union

from .item import Item, ItemTypeError

if TYPE_CHECKING:
    from matplotlib.figure import Figure as Matplotlib


def lazy_is_instance(object: Any, cls_fullname: str) -> bool:
    """Return True if object is an instance of `cls_fullname`."""
    return cls_fullname in {
        f"{cls.__module__}.{cls.__name__}" for cls in object.__class__.__mro__
    }


class MediaType(Enum):
    """Enum used to map aliases and media types."""

    HTML = "text/html"
    MARKDOWN = "text/markdown"
    SVG = "image/svg+xml"


class MediaItem(Item):
    """
    A class to represent a media item.

    This class encapsulates various types of media along with metadata.
    """

    def __init__(
        self,
        media_bytes: bytes,
        media_encoding: str,
        media_type: str,
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
    ):
        """
        Initialize a MediaItem.

        Parameters
        ----------
        media_bytes : bytes
            The raw bytes of the media content.
        media_encoding : str
            The encoding of the media content.
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

        self.media_bytes = media_bytes
        self.media_encoding = media_encoding
        self.media_type = media_type

    def as_serializable_dict(self):
        """Return item as a JSONable dict to export to frontend."""
        if "text" in self.media_type:
            value = self.media_bytes.decode(encoding=self.media_encoding)
            media_type = f"{self.media_type}"
        else:
            value = base64.b64encode(self.media_bytes).decode()
            media_type = f"{self.media_type};base64"

        return super().as_serializable_dict() | {
            "media_type": media_type,
            "value": value,
        }

    @classmethod
    def factory(cls, media, *args, **kwargs):
        """
        Create a new MediaItem instance.

        This is a generic factory method that dispatches to specific
        factory methods based on the type of media provided.

        Parameters
        ----------
        media : Any
            The media content to store.

        Raises
        ------
        NotImplementedError
            If the type of media is not supported.

        Returns
        -------
        MediaItem
            A new MediaItem instance.
        """
        if lazy_is_instance(media, "builtins.bytes"):
            return cls.factory_bytes(media, *args, **kwargs)
        if lazy_is_instance(media, "builtins.str"):
            return cls.factory_str(media, *args, **kwargs)
        if lazy_is_instance(media, "matplotlib.figure.Figure"):
            return cls.factory_matplotlib(media, *args, **kwargs)

        raise ItemTypeError(f"Type '{media.__class__}' is not supported.")

    @classmethod
    def factory_bytes(
        cls,
        media: bytes,
        media_encoding: str = "utf-8",
        media_type: str = "application/octet-stream",
    ) -> MediaItem:
        """
        Create a new MediaItem instance from bytes.

        Parameters
        ----------
        media : bytes
            The raw bytes of the media content.
        media_encoding : str, optional
            The encoding of the media content, by default "utf-8".
        media_type : str, optional
            The MIME type of the media content, by default "application/octet-stream".

        Returns
        -------
        MediaItem
            A new MediaItem instance.
        """
        return cls(
            media_bytes=media,
            media_encoding=media_encoding,
            media_type=media_type,
        )

    @classmethod
    def factory_str(cls, media: str, media_type: str = "text/markdown") -> MediaItem:
        """
        Create a new MediaItem instance from a string.

        Parameters
        ----------
        media : str
            The string content to store.
        media_type : str, optional
            The MIME type of the media content, by default "text/markdown".

        Returns
        -------
        MediaItem
            A new MediaItem instance.
        """
        media_bytes = media.encode("utf-8")

        return cls(
            media_bytes=media_bytes,
            media_encoding="utf-8",
            media_type=media_type,
        )

    @classmethod
    def factory_matplotlib(cls, media: Matplotlib) -> MediaItem:
        """
        Create a new MediaItem instance from a Matplotlib figure.

        Parameters
        ----------
        media : Matplotlib
            The Matplotlib figure to store.

        Returns
        -------
        MediaItem
            A new MediaItem instance.
        """
        with BytesIO() as stream:
            media.savefig(stream, format="svg", bbox_inches="tight")
            media_bytes = stream.getvalue()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/svg+xml",
            )
