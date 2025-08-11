from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, computed_field


class Representation(BaseModel):
    media_type: str
    value: Any

    class Config:
        frozen = True
        arbitrary_types_allowed = True


class Media(ABC, BaseModel):
    key: str
    verbose_name: str | None = None
    category: Literal["performance", "feature_importance", "model", "data"]
    attributes: dict[()] = {}
    parameters: dict[()] = {}

    class Config:
        frozen = True
        arbitrary_types_allowed = True

    @computed_field
    @property
    @abstractmethod
    def representation(self) -> Representation: ...
