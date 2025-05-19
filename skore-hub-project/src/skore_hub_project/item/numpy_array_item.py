"""
NumpyArrayItem.

This module defines the ``NumpyArrayItem`` class used to serialize instances of
``numpy.Array``, using binary protocols.
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
    import numpy


class NumpyArrayItem(Item):
    """Serialize instances of ``numpy.Array``, using binary protocols."""

    def __init__(self, array_b64_str: str):
        """
        Initialize a ``NumpyArrayItem``.

        Parameters
        ----------
        array_b64_str : str
            The raw bytes of the array in the ``numpy`` serialization format, encoded in
            a base64 string.
        """
        self.array_b64_str = array_b64_str

    @cached_property
    def __raw__(self) -> numpy.ndarray:
        """Get the value from the ``NumpyArrayItem``."""
        import numpy

        array_bytes = b64_str_to_bytes(self.array_b64_str)

        with BytesIO(array_bytes) as stream:
            return numpy.load(stream)

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``NumpyArrayItem`` instance."""
        return {
            "representation": {
                "media_type": "application/json",
                "value": self.__raw__.tolist(),
            }
        }

    @classmethod
    def factory(cls, value: numpy.ndarray, /) -> NumpyArrayItem:
        """
        Create a new ``NumpyArrayItem`` from an instance of ``numpy.Array``.

        It uses binary protocols.

        Parameters
        ----------
        value: ``numpy.Array``
            The value to serialize.

        Returns
        -------
        NumpyArrayItem
            A new ``NumpyArrayItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``numpy.Array``.
        """
        if not lazy_is_instance(value, "numpy.ndarray"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        import numpy

        with BytesIO() as stream:
            numpy.save(stream, value, allow_pickle=False)

            array_bytes = stream.getvalue()
            array_b64_str = bytes_to_b64_str(array_bytes)

        instance = cls(array_b64_str)
        instance.__raw__ = value

        return instance
