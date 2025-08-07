from abc import ABC, abstractmethod
from typing import Literal, Any
from typing_extensions import Self

from pydantic import computed_field, model_validator

from skore_hub_project import Payload


class Representation(Payload):
    media_type: str
    value: Any


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
