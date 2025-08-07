from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, computed_field


class Representation(BaseModel):
    media_type: str
    value: Any

    class Config:
        frozen = True


class Media(ABC, BaseModel):
    key: str
    verbose_name: str | None = None
    category: Literal["performance", "feature_importance", "model", "data"]
    attributes: dict | None = None
    parameters: dict | None = None

    class Config:
        frozen = True

    @computed_field
    @property
    @abstractmethod
    def representation(self) -> Representation | None: ...
