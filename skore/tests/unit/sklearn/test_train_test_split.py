import numpy
import pandas
import pytest
from skore.sklearn.train_test_split import HighClassImbalanceWarning

my_list = [0] * 100 + [1] * 100 + [2] * 300


@pytest.mark.parametrize(
    "y_test",
    [
        pytest.param(my_list, id="list"),
        pytest.param(pandas.Series(my_list), id="pandas-series"),
        pytest.param(numpy.array(my_list), id="numpy"),
    ],
)
def test_check_high_class_imbalance(y_test):
    check = HighClassImbalanceWarning.check(
        y_test=y_test,
        stratify=False,
        ml_task="multiclass-classification",
    )
    assert check is True
