from __future__ import annotations

from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from inspect import signature as inspect_signature
from typing import Any, Optional

#
# faire un system qui import dynamiquement les modules
# et qui leve une exception si les modules ne sont pas dispos
# pour empecher d'acceder a la factory
# on pourra faire sauter les lazy_is_instance
#
# rajouter un test_ensure_jsonable pour toutes les classes d'item
#
# faire que les projects sont des plugins
#
# on doit considérer skore comme une lib externe, comme Altair etc
#
# faire sauter le stockage local pour un simple gestionnaire qui pickle tout
#


def lazy_is_instance(value: Any, cls_fullname: str) -> bool:
    """Return True if value is an instance of `cls_fullname`."""
    return cls_fullname in {
        f"{cls.__module__}.{cls.__name__}" for cls in value.__class__.__mro__
    }


def bytes_to_b64_str(literal: bytes) -> str:
    """Encode the bytes-like object `literal` in a Base64 str."""
    return b64encode(literal).decode("utf-8")


def b64_str_to_bytes(literal: str) -> bytes:
    """Decode the Base64 str object `literal` in a bytes."""
    return b64decode(literal.encode("utf-8"))


@dataclass
class Representation:
    media_type: str
    value: Any
    raised: bool = False
    traceback: str = None
    schema: int = 1


class ItemTypeError(Exception):
    """"""


class Item(ABC):
    def __init__(
        self,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat()

        self.created_at = created_at or now
        self.updated_at = updated_at or now
        self.note = note

    @property
    def __parameters__(self) -> dict[str, Any]:
        cls = self.__class__
        cls_parameters = inspect_signature(cls).parameters

        return {parameter: getattr(self, parameter) for parameter in cls_parameters}

    @property
    @abstractmethod
    def __raw__(self) -> Any:
        """"""

    @property
    @abstractmethod
    def __representation__(self) -> Representation:
        """"""

    @classmethod
    @abstractmethod
    def factory(cls, *args, **kwargs) -> Item:
        """"""
