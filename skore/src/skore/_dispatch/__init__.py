"""Dispatch functions to evaluate and compare reports."""

from typing import TYPE_CHECKING

from skore._externals import lazy_loader

if TYPE_CHECKING:
    from skore._dispatch.compare import compare
    from skore._dispatch.evaluate import evaluate

__all__ = ["compare", "evaluate"]

__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submod_attrs={"compare": ["compare"], "evaluate": ["evaluate"]},
)
