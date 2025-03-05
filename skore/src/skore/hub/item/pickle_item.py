"""PickleItem.

This module defines the PickleItem class, which is used to persist objects that cannot
be otherwise.
"""

from __future__ import annotations

from functools import cached_property
from io import BytesIO
from typing import Any

from joblib import dump, load

from .item import (
    Item,
    Representation,
    b64_str_to_bytes,
    bytes_to_b64_str,
)


class PickleItem(Item):
    """
    An item used to persist objects that cannot be otherwise, using binary protocols.

    It encapsulates the object with its pickle representaton, its creation and update
    timestamps.
    """

    def __init__(self, pickle_b64_str: str):
        self.pickle_b64_str = pickle_b64_str

    @cached_property
    def __raw__(self) -> Any:
        pickle_bytes = b64_str_to_bytes(self.pickle_b64_str)

        with BytesIO(pickle_bytes) as stream:
            return load(stream)

    @property
    def __representation__(self) -> Representation:
        return Representation(
            media_type="text/markdown",
            value=f"```python\n{repr(self.__raw__)}\n```",
        )

    @classmethod
    def factory(cls, value: Any) -> PickleItem:
        """
        Create a new PickleItem from ``object``.

        Parameters
        ----------
        value: Any
            The value to store.

        Returns
        -------
        PickleItem
            A new PickleItem instance.
        """
        with BytesIO() as stream:
            dump(value, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        instance = cls(pickle_b64_str)
        instance.__raw__ = value

        return instance
