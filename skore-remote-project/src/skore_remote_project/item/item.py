from __future__ import annotations

from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from inspect import signature as inspect_signature
from typing import Any


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


class ItemTypeError(Exception):
    """"""


class Item(ABC):
    @property
    def __parameters__(self) -> dict[str, dict[str, Any]]:
        """"""
        cls = self.__class__
        cls_name = cls.__name__
        cls_parameters = inspect_signature(cls).parameters

        return {
            "parameters": {
                "class": cls_name,
                "parameters": {p: getattr(self, p) for p in cls_parameters},
            }
        }

    @property
    def __metadata__(self) -> dict[str, Any]:
        return dict()

    @property
    @abstractmethod
    def __raw__(self) -> Any:
        """"""

    @property
    @abstractmethod
    def __representation__(self) -> dict[str, Any]:
        """"""

    @classmethod
    @abstractmethod
    def factory(cls, *args, **kwargs) -> Item:
        """"""
