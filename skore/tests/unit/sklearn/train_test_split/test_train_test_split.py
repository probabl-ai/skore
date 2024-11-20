import warnings

import pytest
from skore.sklearn.train_test_split.train_test_split import (
    train_test_split,
)
from skore.sklearn.train_test_split.warning import (
    HighClassImbalanceTooFewExamplesWarning,
    HighClassImbalanceWarning,
    StratifyWarning,
)


def case_high_class_imbalance():
    args = ([[1]] * 4, [0, 1, 1, 1])
    kwargs = {}
    return args, kwargs, HighClassImbalanceWarning


def case_high_class_imbalance_too_few_examples():
    args = ([[1]] * 4, [0, 1, 1, 1])
    kwargs = {}
    return args, kwargs, HighClassImbalanceTooFewExamplesWarning


def case_stratify():
    args = ([0] * 10 + [1] * 10,)
    kwargs = dict(stratify=[0] * 10 + [1] * 10)
    return args, kwargs, StratifyWarning


@pytest.mark.parametrize(
    "params",
    [
        case_high_class_imbalance,
        case_high_class_imbalance_too_few_examples,
        case_stratify,
    ],
)
def test_train_test_split_warns(params):
    """When train_test_split is called with these args and kwargs, the corresponding
    warning should fire."""
    warnings.simplefilter("ignore")
    args, kwargs, warning_cls = params()

    with pytest.warns(warning_cls):
        train_test_split(*args, **kwargs)


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


def test_train_test_split_kwargs():
    """Passing data by keyword arguments should produce the same results as passing
    them by position."""
    warnings.simplefilter("ignore")

    X = [[1]] * 20
    y = [0] * 10 + [1] * 10
    output1 = train_test_split(X, y, random_state=0)
    output2 = train_test_split(X=X, y=y, random_state=0)

    assert output1 == output2
