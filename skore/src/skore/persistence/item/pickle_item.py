from functools import cached_property
from pickle import dumps, loads
from typing import Any

from skore.item.item import Item


class PickleItem(Item):
    def __init__(
        self,
        pickle_bytes: bytes,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        super().__init__(created_at, updated_at)

        self.pickle_bytes = pickle_bytes

    @cached_property
    def object(self) -> Any:
        return loads(self.pickle_bytes)

    @classmethod
    def factory(cls, object: Any) -> PickleItem:
        return cls(dumps(object))
