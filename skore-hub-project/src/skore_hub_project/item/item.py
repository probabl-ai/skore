"""Abstract base class for all items in the ``skore`` hub project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from contextlib import contextmanager
from inspect import signature as inspect_signature
from typing import Any


@contextmanager
def switch_mpl_backend():
    """
    Context-manager for switching ``matplotlib.backend`` to ``agg``.

    Notes
    -----
    The ``agg`` backend is a non-interactive backend that can only write to files.
    It is used in ``skore-hub-project`` to generate artifacts where we don't need an
    X display.

    https://matplotlib.org/stable/users/explain/figure/backends.html#selecting-a-backend
    """
    import matplotlib

    original_backend = matplotlib.get_backend()

    try:
        matplotlib.use("agg")
        yield
    finally:
        matplotlib.use(original_backend)


def lazy_is_instance(value: Any, cls_fullname: str) -> bool:
    """Return True if value is an instance of ``cls_fullname``."""
    return cls_fullname in {
        f"{cls.__module__}.{cls.__name__}" for cls in value.__class__.__mro__
    }


def bytes_to_b64_str(literal: bytes) -> str:
    """Encode the bytes-like object ``literal`` in a Base64 str."""
    return b64encode(literal).decode("utf-8")


def b64_str_to_bytes(literal: str) -> bytes:
    """Decode the Base64 str object ``literal`` in a bytes."""
    return b64decode(literal.encode("utf-8"))


class ItemTypeError(Exception):
    """
    Item type exception.

    Exception raised when an attempt is made to convert an object to an ``Item`` with an
    unsupported type.
    """


class Item(ABC):
    """
    Abstract base class for all items in the ``skore`` hub project.

    This class provides a common interface for all items, including the serialization of
    the parameters needed to recreate the instance from the hub project.

    ``Item`` is an internal concept that is used as a DTO (Data Transfer Object) to
    exchange python objects between ``skore`` and ``skore hub``.
    """

    @property
    def __parameters__(self) -> dict[str, dict[str, Any]]:
        """
        Get the parameters of the ``Item`` instance.

        These parameters must be sufficient to recreate the instance.
        They are persisted in the ``skore`` hub project and retrieved as needed.
        """
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
        """Get the metadata of the ``Item`` instance."""
        return dict()

    @property
    @abstractmethod
    def __raw__(self) -> Any:
        """Get the raw python object from the ``Item`` instance."""

    @property
    @abstractmethod
    def __representation__(self) -> dict[str, Any]:
        """Get the representation of the ``Item`` instance."""

    @classmethod
    @abstractmethod
    def factory(cls, *args, **kwargs) -> Item:
        """Create and return a new instance of ``Item``."""
