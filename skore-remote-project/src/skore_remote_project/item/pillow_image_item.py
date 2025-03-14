from __future__ import annotations

from functools import cached_property
from io import BytesIO
from typing import TYPE_CHECKING

from .item import (
    Item,
    ItemTypeError,
    Representation,
    b64_str_to_bytes,
    bytes_to_b64_str,
    lazy_is_instance,
)

if TYPE_CHECKING:
    import PIL.Image


class PillowImageItem(Item):
    """A class used to persist a Pillow image."""

    def __init__(
        self,
        image_b64_str: str,
        image_mode: str,
        image_size: tuple[int, int],
    ):
        self.image_b64_str = image_b64_str
        self.image_mode = image_mode
        self.image_size = image_size

    @cached_property
    def __raw__(self) -> PIL.Image.Image:
        import PIL.Image

        image_bytes = b64_str_to_bytes(self.image_b64_str)

        return PIL.Image.frombytes(
            mode=self.image_mode,
            size=self.image_size,
            data=image_bytes,
        )

    @property
    def __representation__(self) -> Representation:
        with BytesIO() as stream:
            self.__raw__.save(stream, format="png")

            png_bytes = stream.getvalue()
            png_b64_str = bytes_to_b64_str(png_bytes)

        return Representation(media_type="image/png;base64", value=png_b64_str)

    @classmethod
    def factory(cls, image: PIL.Image.Image, /) -> PillowImageItem:
        if not lazy_is_instance(image, "PIL.Image.Image"):
            raise ItemTypeError(f"Type '{image.__class__}' is not supported.")

        image_bytes = image.tobytes()
        image_b64_str = bytes_to_b64_str(image_bytes)

        instance = cls(image_b64_str, image.mode, image.size)
        instance.__raw__ = image

        return instance
