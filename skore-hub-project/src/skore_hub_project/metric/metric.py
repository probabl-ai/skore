from __future__ import annotations

from abc import abstractmethod
from contextlib import suppress
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, computed_field


def cast_to_float(value: Any) -> float | None:
    """Cast value to float."""
    with suppress(TypeError):
        if isfinite(value := float(value)):
            return value

    return None


class Metric(BaseModel):
    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None = None
    greater_is_better: bool
    position: int | None = None

    @computed_field
    @property
    @abstractmethod
    def value(self) -> float | None: ...
