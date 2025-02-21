"""PillowImageItem.

This module defines the PillowImageItem class, used to persist Pillow images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from skore.persistence.item.item import Item, ItemTypeError
from skore.persistence.item.media_item import lazy_is_instance
from skore.utils import b64_str_to_bytes, bytes_to_b64_str

if TYPE_CHECKING:
    import PIL.Image


class PillowImageItem(Item):
    """A class used to persist a Pillow image."""

    def __init__(
        self,
        image_b64_str: str,
        image_mode: str,
        image_size: tuple[int, int],
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a PillowImageItem.

        Parameters
        ----------
        image_b64_str : str
            The raw bytes of the Pillow image.
        image_mode : str
            The image mode.
        image_size : tuple of int
            The image size.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.image_b64_str = image_b64_str
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

        image_bytes = image.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        return cls(
            image_b64_str=image_b64_str,
            image_mode=image.mode,
            image_size=image.size,
            **kwargs,
        )

    @property
    def image(self) -> PIL.Image.Image:
        """The image from the persistence."""
        import PIL.Image

        image_bytes = b64_str_to_bytes(self.image_b64_str)

        return PIL.Image.frombytes(
            mode=self.image_mode,
            size=self.image_size,
            data=image_bytes,
        )
