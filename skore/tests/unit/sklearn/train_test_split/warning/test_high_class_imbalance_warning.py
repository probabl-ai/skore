import numpy
import pandas
import pytest
from skore.sklearn.train_test_split.warning import (
    HighClassImbalanceWarning,
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
    warning = HighClassImbalanceWarning.check(
        y=y,
        stratify=None,
        ml_task="multiclass-classification",
    )

    assert warning == HighClassImbalanceWarning.MSG
