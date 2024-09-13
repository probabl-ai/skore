"""MediaItem.

This module defines the MediaItem class, which represents media items.
"""

from __future__ import annotations

from functools import singledispatchmethod
from io import BytesIO

from altair.vegalite.v5.schema.core import TopLevelSpec as Altair
from matplotlib.figure import Figure as Matplotlib
from PIL.Image import Image as Pillow

from skore.item.item import Item


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
        created_at: str | None = None,
        updated_at: str | None = None,
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
        """
        super().__init__(created_at, updated_at)

        self.media_bytes = media_bytes
        self.media_encoding = media_encoding
        self.media_type = media_type

    @singledispatchmethod
    @classmethod
    def factory(cls, media):
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
        raise NotImplementedError(f"Type '{type(media)}' is not yet supported")

    @factory.register(bytes)
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

    @factory.register(str)
    @classmethod
    def factory_str(cls, media: str, media_type: str = "text/markdown") -> MediaItem:
        """
        Create a new MediaItem instance from a string.

        Parameters
        ----------
        media : str
            The string content to store.
        media_type : str, optional
            The MIME type of the media content, by default "text/html".

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

    @factory.register(Altair)
    @classmethod
    def factory_altair(cls, media: Altair) -> MediaItem:
        """
        Create a new MediaItem instance from an Altair chart.

        Parameters
        ----------
        media : Altair
            The Altair chart to store.

        Returns
        -------
        MediaItem
            A new MediaItem instance.
        """
        media_bytes = media.to_json().encode("utf-8")

        return cls(
            media_bytes=media_bytes,
            media_encoding="utf-8",
            media_type="application/vnd.vega.v5+json",
        )

    @factory.register(Matplotlib)
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
            media.savefig(stream, format="svg")
            media_bytes = stream.getvalue()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/svg+xml",
            )

    @factory.register(Pillow)
    @classmethod
    def factory_pillow(cls, media: Pillow) -> MediaItem:
        """
        Create a new MediaItem instance from a Pillow image.

        Parameters
        ----------
        media : Pillow
            The Pillow image to store.

        Returns
        -------
        MediaItem
            A new MediaItem instance.
        """
        with BytesIO() as stream:
            media.save(stream, format="png")
            media_bytes = stream.getvalue()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/png",
            )
