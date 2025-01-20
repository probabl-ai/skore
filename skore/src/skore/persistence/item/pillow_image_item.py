"""PillowImageItem.

This module defines the PillowImageItem class, used to persist Pillow images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .item import Item, ItemTypeError
from .media_item import lazy_is_instance

if TYPE_CHECKING:
    import PIL.Image


class PillowImageItem(Item):
    """A class used to persist a Pillow image."""

    def __init__(
        self,
        image_bytes: bytes,
        image_mode: str,
        image_size: tuple[int],
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a PillowImageItem.

        Parameters
        ----------
        image_bytes : bytes
            The raw bytes of the Pillow image.
        image_mode : str
            The image mode.
        image_size : tuple[int]
            The image size.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.image_bytes = image_bytes
        self.image_mode = image_mode
        self.image_size = image_size

    @classmethod
    def factory(cls, image: PIL.Image.Image, /, **kwargs) -> PillowImageItem:
        """
        Create a new PillowImageItem instance from a Pillow image.

        Parameters
        ----------
        image : PIL.Image.Image
            The Pillow image to store.

        Returns
        -------
        PillowImageItem
            A new PillowImageItem instance.
        """
        if not lazy_is_instance(image, "PIL.Image.Image"):
            raise ItemTypeError(f"Type '{image.__class__}' is not supported.")

        return cls(
            image_bytes=image.tobytes(),
            image_mode=image.mode,
            image_size=image.size,
            **kwargs,
        )

    @property
    def image(self) -> PIL.Image.Image:
        """The image from the persistence."""
        import PIL.Image

        return PIL.Image.frombytes(
            mode=self.image_mode,
            size=self.image_size,
            data=self.image_bytes,
        )

    def as_serializable_dict(self):
        """Convert item to a JSON-serializable dict to used by frontend."""
        import base64
        import io

        with io.BytesIO() as stream:
            self.image.save(stream, format="png")

            png_bytes = stream.getvalue()
            png_b64_str = base64.b64encode(png_bytes).decode()

        return super().as_serializable_dict() | {
            "media_type": "image/png;base64",
            "value": png_b64_str,
        }
