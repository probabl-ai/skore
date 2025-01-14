import warnings
from datetime import datetime

import pandas
import polars
import pytest
from skore.sklearn.train_test_split.train_test_split import (
    train_test_split,
)
from skore.sklearn.train_test_split.warning import (
    HighClassImbalanceTooFewExamplesWarning,
    HighClassImbalanceWarning,
    RandomStateUnsetWarning,
    ShuffleTrueWarning,
    StratifyWarning,
    TimeBasedColumnWarning,
)


def case_high_class_imbalance():
    args = ([[1]] * 4, [0, 1, 1, 1])
    kwargs = {}
    return args, kwargs, HighClassImbalanceWarning


def case_high_class_imbalance_too_few_examples():
    args = ([[1]] * 4, [0, 1, 1, 1])
    kwargs = {}
    return args, kwargs, HighClassImbalanceTooFewExamplesWarning


def case_high_class_imbalance_too_few_examples_kwargs():
    args = ()
    kwargs = dict(X=[[1]] * 4, y=[0, 1, 1, 1])
    return args, kwargs, HighClassImbalanceTooFewExamplesWarning


def case_high_class_imbalance_too_few_examples_kwargs_mixed():
    args = ([[1]] * 4,)
    kwargs = dict(y=[0, 1, 1, 1])
    return args, kwargs, HighClassImbalanceTooFewExamplesWarning


def case_stratify():
    args = ([0] * 10 + [1] * 10,)
    kwargs = dict(stratify=[0] * 10 + [1] * 10)
    return args, kwargs, StratifyWarning


def case_random_state_unset():
    # By default shuffle is True and random_state is None
    args = ([[1]] * 4, [0, 1, 1, 1])
    kwargs = {}
    return args, kwargs, RandomStateUnsetWarning


def case_shuffle_true():
    args = ([[1]] * 4, [0, 1, 1, 1])
    kwargs = dict(shuffle=True)
    return args, kwargs, ShuffleTrueWarning


def case_shuffle_none():
    args = ([[1]] * 4, [0, 1, 1, 1])
    kwargs = {}
    return args, kwargs, ShuffleTrueWarning


def case_time_based_column():
    """If a column has dtype "datetime", the warning should fire"""
    X = pandas.DataFrame(
        {"ints": [0, 1], "dates": [datetime(2024, 11, 25), datetime(2024, 11, 26)]}
    )
    args = (X,)
    kwargs = {}
    return args, kwargs, TimeBasedColumnWarning


def case_time_based_columns_several():
    """If a column has dtype "datetime", the warning should fire"""
    X = pandas.DataFrame(
        {
            "ints": [0, 1],
            "dates1": [datetime(2024, 11, 25), datetime(2024, 11, 26)],
            # NOTE: Column name is an int
            2: [datetime(2024, 11, 25), datetime(2024, 11, 26)],
        }
    )
    # NOTE: DataFrame has a name
    X.name = "my_df"
    args = (X,)
    kwargs = {}
    return args, kwargs, TimeBasedColumnWarning


def case_time_based_column_polars():
    X = polars.DataFrame(
        {"ints": [0, 1], "dates": [datetime(2024, 11, 25), datetime(2024, 11, 26)]}
    )
    args = (X,)
    kwargs = {}
    return args, kwargs, TimeBasedColumnWarning


def case_time_based_column_polars_dates():
    X = polars.DataFrame(
        {"ints": [0, 1], "dates": [datetime(2024, 11, 25), datetime(2024, 11, 26)]},
        schema={"ints": polars.Int8, "dates": polars.Date},
    )
    args = (X,)
    kwargs = {}
    return args, kwargs, TimeBasedColumnWarning


@pytest.mark.parametrize(
    "params",
    [
        case_high_class_imbalance,
        case_high_class_imbalance_too_few_examples,
        case_high_class_imbalance_too_few_examples_kwargs,
        case_high_class_imbalance_too_few_examples_kwargs_mixed,
        case_stratify,
        case_random_state_unset,
        case_shuffle_true,
        case_shuffle_none,
        case_time_based_column,
        case_time_based_columns_several,
        case_time_based_column_polars,
        case_time_based_column_polars_dates,
    ],
)
def test_train_test_split_warns(params, capsys):
    """When train_test_split is called with these args and kwargs, the corresponding
    warning should be printed to the console."""
    args, kwargs, warning_cls = params()

    train_test_split(*args, **kwargs)

    captured = capsys.readouterr()
    assert warning_cls.__name__ in captured.out


@pytest.mark.parametrize(
    "params",
    [
        case_high_class_imbalance,
        case_high_class_imbalance_too_few_examples,
        case_high_class_imbalance_too_few_examples_kwargs,
        case_high_class_imbalance_too_few_examples_kwargs_mixed,
        case_stratify,
        case_random_state_unset,
        case_shuffle_true,
        case_shuffle_none,
        case_time_based_column,
        case_time_based_columns_several,
        case_time_based_column_polars,
        case_time_based_column_polars_dates,
    ],
)
def test_train_test_split_warns_suppressed(params, capsys):
    """Verify that warnings can be suppressed and don't appear in the console output."""
    args, kwargs, warning_cls = params()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=warning_cls)
        train_test_split(*args, **kwargs)

    captured = capsys.readouterr()
    assert warning_cls.__name__ not in captured.out


def test_train_test_split_kwargs():
    """Passing data by keyword arguments should produce the same results as passing
    them by position."""
    warnings.simplefilter("ignore")

    X = [[1]] * 20
    y = [0] * 10 + [1] * 10
    output1 = train_test_split(X, y, random_state=0)
    output2 = train_test_split(X=X, y=y, random_state=0)

    assert output1 == output2
