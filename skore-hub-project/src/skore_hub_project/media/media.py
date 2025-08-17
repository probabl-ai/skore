"""Class definition of the payload used to send a media to ``hub``."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field


class Representation(BaseModel):
    """Payload used to send the representation of a media to ``hub``."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    media_type: str
    value: Any


class Media(ABC, BaseModel):
    """
    Payload used to send a media to ``hub``.

    Attributes
    ----------
    key : str
        Key/name of the media.
    verbose_name : str
        Verbose name of the media.
    category: Literal["performance", "feature_importance", "model", "data"]
        Category of the media.
    attributes: dict, optional
        Attributes of the media usually used to customize the calculation function,
        default empty dict.
    parameters: dict
        .. deprecated
          The ``parameters`` property is unused and will be removed.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    key: str
    verbose_name: str
    category: Literal["performance", "feature_importance", "model", "data"]
    attributes: dict = {}
    parameters: dict = {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def representation(self) -> Representation | None:
        """The representation of the media."""
