"""
PillowImageItem.

This module defines the ``PillowImageItem`` class used to serialize instances of
``pillow`` images, using binary protocols.
"""

from __future__ import annotations

from functools import cached_property
from io import BytesIO
from typing import TYPE_CHECKING

from .item import (
    Item,
    ItemTypeError,
    b64_str_to_bytes,
    bytes_to_b64_str,
    lazy_is_instance,
)

if TYPE_CHECKING:
    import PIL.Image


class PillowImageItem(Item):
    """Serialize instances of ``pillow`` images, using binary protocols."""

    def __init__(
        self,
        image_b64_str: str,
        image_mode: str,
        image_size: tuple[int, int],
    ):
        """
        Initialize a ``PillowImageItem``.

        Parameters
        ----------
        image_b64_str : str
            The raw bytes of the image in the ``pillow`` serialization format, encoded
            in a base64 string.
        image_mode : str
            The image mode.
        image_size : tuple of int
            The image size.
        """
        self.image_b64_str = image_b64_str
        self.image_mode = image_mode
        self.image_size = image_size

    @cached_property
    def __raw__(self) -> PIL.Image.Image:
        """Get the value from the ``PillowImageItem`` instance."""
        import PIL.Image

        image_bytes = b64_str_to_bytes(self.image_b64_str)

        return PIL.Image.frombytes(
            mode=self.image_mode,
            size=self.image_size,
            data=image_bytes,
        )

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``PillowImageItem`` instance."""
        with BytesIO() as stream:
            self.__raw__.save(stream, format="png")

            png_bytes = stream.getvalue()
            png_b64_str = bytes_to_b64_str(png_bytes)

        return {
            "representation": {
                "media_type": "image/png;base64",
                "value": png_b64_str,
            }
        }

    @classmethod
    def factory(cls, value: PIL.Image.Image, /) -> PillowImageItem:
        """
        Create a new ``PillowImageItem`` from an instance of ``pillow`` image.

        It uses binary protocols.

        Parameters
        ----------
        value: ``pillow`` image.
            The value to serialize.

        Returns
        -------
        PillowImageItem
            A new ``NumpyArrayItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``pillow`` image.
        """
        if not lazy_is_instance(value, "PIL.Image.Image"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        image_bytes = value.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        instance = cls(image_b64_str, value.mode, value.size)
        instance.__raw__ = value

        return instance
