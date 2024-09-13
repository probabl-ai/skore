from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from functools import cached_property


class Item(ABC):
    def __init__(
        self,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        now = datetime.now(tz=UTC).isoformat()

        self.created_at = created_at or now
        self.updated_at = updated_at or now

    @classmethod
    @abstractmethod
    def factory(cls) -> Item: ...

    @cached_property
    def __parameters__(self) -> dict[str, Any]:
        cls = self.__class__
        cls_parameters = inspect.signature(cls).parameters

        return {parameter: getattr(self, parameter) for parameter in cls_parameters}
