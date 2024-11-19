import warnings

import pytest
from skore.sklearn.train_test_split.train_test_split import (
    train_test_split,
)
from skore.sklearn.train_test_split.warning import (
    HighClassImbalanceTooFewExamplesWarning,
    HighClassImbalanceWarning,
)


def test_train_test_split_warns():
    warnings.simplefilter("ignore")

    with pytest.warns(HighClassImbalanceWarning, match=HighClassImbalanceWarning.MSG):
        train_test_split([[1]] * 4, [0, 1, 1, 1])


def test_train_test_split_too_few_examples_warns():
    warnings.simplefilter("ignore")

    with pytest.warns(
        HighClassImbalanceTooFewExamplesWarning,
        match=HighClassImbalanceTooFewExamplesWarning.MSG,
    ):
        train_test_split([[1]] * 4, [0, 1, 1, 1])


def test_train_test_split_no_y():
    """When calling `train_test_split` with one array argument,
    this array is assumed to be `X` and not `y`."""
    warnings.simplefilter("error")

    # Since the array is `X` and we do no checks on it, this should produce no
    # warning
    train_test_split([[1]] * 4)


def test_train_test_split_no_warn():
    warnings.simplefilter("error")

    train_test_split([[1]] * 2000, [0] * 1000 + [1] * 1000, random_state=0)
