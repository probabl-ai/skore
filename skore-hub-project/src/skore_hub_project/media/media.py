from abc import ABC, abstractmethod
from typing import Any, Literal, ClassVar
from functools import cached_property

from pydantic import BaseModel, computed_field, Field

from skore_hub_project import Payload


class Representation(Payload):
    media_type: str
    value: str


class Media(ABC, Payload):
    key: str
    verbose_name: str | None = None
    category: Literal["performance", "feature_importance", "model", "data"]
    attributes: dict | None = None
    parameters: dict | None = None

    @computed_field
    @property
    @abstractmethod
    def representation(self) -> Representation | None: ...
