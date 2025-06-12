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


def test_train_test_split_single_posargs():
    """Passing single positional argument with as_dict=True should return as dict."""

    X = [[1]] * 20

    output = train_test_split(X, as_dict=True)
    assert isinstance(output, dict)
    assert set(output.keys()) == {"X_train", "X_test"}


def test_train_test_split_two_posargs():
    """Passing two positional argument with as_dict=True should return as dict."""

    X = [[1]] * 20
    y = [0] * 10 + [1] * 10

    output = train_test_split(X, y, as_dict=True)
    assert isinstance(output, dict)
    assert set(output.keys()) == {"X_train", "X_test", "y_train", "y_test"}


def test_train_test_split_dict_pos_kwargs_conflict():
    """Passing X or y by both position and keyword with as_dict=True
    should throw an error."""

    X = [[1]] * 20
    y = [0] * 10 + [1] * 10

    err_msg = (
        "With as_dict=True, expected {} to be passed either "
        "by position or keyword, not both."
    )

    import re

    with pytest.raises(ValueError, match=re.escape(err_msg.format("X"))):
        train_test_split(X, X=X, as_dict=True)
    with pytest.raises(ValueError, match=re.escape(err_msg.format("y"))):
        train_test_split(y, y=y, as_dict=True)


def test_train_test_split_mix_args():
    """Passing mixed positional and keyword argument with as_dict=True
    should return as dict."""

    X = [[1]] * 20
    y = [0] * 10 + [1] * 10
    z = [0] * 10 + [1] * 10

    output = train_test_split(X, y, z=z, as_dict=True)
    assert isinstance(output, dict)
    assert set(output.keys()) == {
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "z_train",
        "z_test",
    }


def test_train_test_split_dict_kwargs():
    """Passing three or more keyword arguments with as_dict=True
    should raise ValueError."""

    X = [[1]] * 20
    y = [0] * 10 + [1] * 10
    z = [0] * 10 + [1] * 10

    err_msg = (
        "With as_dict=True, expected no more than two positional arguments "
        "(which will be interpreted as X and y). "
        "The remaining arrays must be passed by keyword, "
        "e.g. train_test_split(X, y, z=z, sw=sample_weights, as_dict=True)."
    )
    import re

    with pytest.raises(ValueError, match=re.escape(err_msg)):
        train_test_split(X, y, z, as_dict=True)


def test_train_test_split_check_dict_unsupervised_case():
    """If `as_dict` is True and only `X` is passed,
    the result is a dict with 2 keys."""

    X = [[1]] * 20
    output = train_test_split(X=X, random_state=0, as_dict=True)
    assert len(output.keys()) == 2


def test_train_test_split_check_dict_no_X_no_y():
    """If the input is a keyword argument and `X` and `y` are None,
    then the result is a dict with 2 keys."""

    z = [[1]] * 20
    output = train_test_split(z=z, random_state=0, as_dict=True)
    keys = output.keys()
    assert list(keys) == ["z_train", "z_test"]
