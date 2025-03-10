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
)

if TYPE_CHECKING:
    import numpy


class NumpyArrayItem(Item):
    def __init__(self, array_b64_str: str):
        self.array_b64_str = array_b64_str

    @cached_property
    def __raw__(self) -> numpy.ndarray:
        import numpy

        array_bytes = b64_str_to_bytes(self.array_b64_str)

        with BytesIO(array_bytes) as stream:
            return numpy.load(stream)

    @property
    def __representation__(self) -> Representation:
        return Representation(
            media_type="application/json",
            value=self.__raw__.tolist(),
        )

    @classmethod
    def factory(cls, array: numpy.ndarray, /, **kwargs) -> NumpyArrayItem:
        import numpy

        if not isinstance(array, numpy.ndarray):
            raise ItemTypeError(f"Type '{array.__class__}' is not supported.")

        with BytesIO() as stream:
            numpy.save(stream, array, allow_pickle=False)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)

        instance = cls(array_b64_str, **kwargs)
        instance.__raw__ = array

        return instance
