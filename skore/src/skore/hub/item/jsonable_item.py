from __future__ import annotations

from json import dumps, loads
from typing import Any

from .item import Item, ItemTypeError, Representation


class JSONableItem(Item):
    def __init__(self, value: Any):
        self.value = value

    @property
    def __raw__(self):
        return self.value

    @property
    def __representation__(self) -> Representation:
        return Representation(media_type="application/json", value=self.__raw__)

    @classmethod
    def factory(cls, value: Any, /, **kwargs) -> JSONableItem:
        try:
            value = loads(dumps(value))
        except TypeError:
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.") from None

        return cls(value, **kwargs)
