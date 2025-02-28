from __future__ import annotations

from json import dumps, loads
from typing import Any, Union

from .item import Item, ItemTypeError, Representation


class JSONableItem(Item):
    def __init__(
        self,
        value_json_str: str,
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
    ):
        super().__init__(created_at, updated_at, note)

        self.value_json_str = value_json_str

    @property
    def __raw__(self) -> Any:
        return loads(self.value_json_str)

    @property
    def __representation__(self) -> Representation:
        return Representation(media_type="application/json", value=self.__raw__)

    @classmethod
    def factory(cls, value: Any, /, **kwargs) -> JSONableItem:
        try:
            value_json_str = dumps(value)
        except TypeError:
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.") from None

        return cls(value_json_str, **kwargs)
