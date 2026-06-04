"""Enhance `sklearn` functions."""

from skore._externals import _lazy_loader as lazy
from skore._sklearn.train_test_split.train_test_split import (
    TrainTestSplit as TrainTestSplit,
)
from skore._sklearn.train_test_split.train_test_split import (
    train_test_split as train_test_split,
)

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
