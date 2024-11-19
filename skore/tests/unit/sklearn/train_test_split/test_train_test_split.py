import contextlib

import pytest
from skore.sklearn.train_test_split.train_test_split import (
    train_test_split,
)
from skore.sklearn.train_test_split.warning import (
    HighClassImbalanceTooFewExamplesWarning,
    HighClassImbalanceWarning,
)


def test_train_test_split_warns():
    with pytest.warns(HighClassImbalanceWarning, match=HighClassImbalanceWarning.MSG):
        train_test_split([[1]] * 4, [0, 1, 1, 1])


def test_train_test_split_no_y():
    """When calling `train_test_split` with one array argument,
    this array is assumed to be `X` and not `y`."""
    with contextlib.nullcontext():
        train_test_split([[1]] * 4)


def test_train_test_split_no_warn():
    with contextlib.nullcontext():
        train_test_split([[1]] * 4, [0, 0, 1, 1])
