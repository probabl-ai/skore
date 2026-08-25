"""Enhance `sklearn` functions."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["TrainTestSplit"]

__lazy__ = {
    "TrainTestSplit": "skore._sklearn.train_test_split",
}

if TYPE_CHECKING:
    from skore._sklearn.train_test_split import TrainTestSplit


def __getattr__(name: str) -> Any:
    if module_name := __lazy__.get(name):
        value = getattr(import_module(module_name), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
