import contextlib

import numpy
import pandas
import pytest
from skore.sklearn.train_test_split.train_test_split import (
    HighClassImbalanceWarning,
    train_test_split,
)

target = [0] * 100 + [1] * 100 + [2] * 300


@pytest.mark.parametrize(
    "y",
    [
        pytest.param(target, id="list"),
        pytest.param(pandas.Series(target), id="pandas-series"),
        pytest.param(numpy.array(target), id="numpy"),
    ],
)
def test_check_high_class_imbalance(y):
    check = HighClassImbalanceWarning.check(
        y=y,
        stratify=None,
        ml_task="multiclass-classification",
    )

    assert check is False


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
