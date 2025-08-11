from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field


class Representation(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    media_type: str
    value: Any


class Media(ABC, BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    key: str
    verbose_name: str
    category: Literal["performance", "feature_importance", "model", "data"]
    attributes: dict = {}
    parameters: dict = {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def representation(self) -> Representation | None: ...
