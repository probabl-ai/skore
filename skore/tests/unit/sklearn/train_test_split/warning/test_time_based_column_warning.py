from datetime import datetime

import pandas
import polars
import pytest

from skore._sklearn.train_test_split.warning import TimeBasedColumnWarning

target = [0] * 100 + [1] * 100 + [2] * 300


def case_time_based_column():
    """If a column has dtype "datetime", the warning should fire"""
    return pandas.DataFrame(
        {"ints": [0, 1], "dates": [datetime(2024, 11, 25), datetime(2024, 11, 26)]}
    )


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
    return X


def case_time_based_column_polars():
    return polars.DataFrame(
        {"ints": [0, 1], "dates": [datetime(2024, 11, 25), datetime(2024, 11, 26)]}
    )


def case_time_based_column_polars_dates():
    return polars.DataFrame(
        {"ints": [0, 1], "dates": [datetime(2024, 11, 25), datetime(2024, 11, 26)]},
        schema={"ints": polars.Int8, "dates": polars.Date},
    )


@pytest.mark.parametrize(
    "X",
    [
        case_time_based_column,
        case_time_based_columns_several,
        case_time_based_column_polars,
        case_time_based_column_polars_dates,
    ],
)
def test_check_time_based_column(X):
    warning = TimeBasedColumnWarning.check(X=X())

    assert warning is not None
