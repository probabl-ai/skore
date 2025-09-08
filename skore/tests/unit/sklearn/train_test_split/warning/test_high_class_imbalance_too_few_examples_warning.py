import numpy
import pytest
from skore._sklearn.train_test_split.warning import (
    HighClassImbalanceTooFewExamplesWarning,
)


@pytest.mark.parametrize(
    "y_test",
    [
        pytest.param([0] * 100, id="list"),
        pytest.param(numpy.array([0] * 100).reshape(-1, 1), id="numpy-column"),
    ],
)
def test_check_high_class_imbalance_too_few_examples(y_test):
    y_labels = [0, 1]

    warning = HighClassImbalanceTooFewExamplesWarning.check(
        y_test=y_test,
        y_labels=y_labels,
        stratify=None,
        ml_task="multiclass-classification",
    )

    assert warning == HighClassImbalanceTooFewExamplesWarning.MSG
