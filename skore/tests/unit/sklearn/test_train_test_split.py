import contextlib

import numpy
import pandas
import pytest
from skore.sklearn.train_test_split.train_test_split import (
    HighClassImbalanceWarning,
    train_test_split,
)

my_list = [0] * 100 + [1] * 100 + [2] * 300


@pytest.mark.parametrize(
    "y",
    [
        pytest.param(my_list, id="list"),
        pytest.param(pandas.Series(my_list), id="pandas-series"),
        pytest.param(numpy.array(my_list), id="numpy"),
    ],
)
def test_check_high_class_imbalance(y):
    check = HighClassImbalanceWarning.check(
        y=y,
        stratify=None,
        ml_task="multiclass-classification",
    )

    assert check is False


@pytest.mark.parametrize(
    "x,y,context",
    [
        pytest.param(
            [[1]] * 4,
            [0, 1, 1, 1],
            pytest.warns(
                HighClassImbalanceWarning, match=HighClassImbalanceWarning.MSG
            ),
            id="warn",
        ),
        pytest.param(
            [[1]] * 4,
            [0, 0, 1, 1],
            contextlib.nullcontext(),
            id="no-warn",
        ),
    ],
)
def test_train_test_split(x, y, context):
    with context:
        train_test_split(x, y)
