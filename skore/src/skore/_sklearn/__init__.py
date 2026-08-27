"""Enhance `sklearn` functions."""

from typing import TYPE_CHECKING

from skore._externals import lazy_loader

if TYPE_CHECKING:
    from skore._sklearn.train_test_split import TrainTestSplit

__all__ = ["TrainTestSplit"]

__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submod_attrs={"train_test_split": ["TrainTestSplit"]},
)
