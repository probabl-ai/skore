import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from skore import CrossValidationReport, Display, EstimatorReport
from skore._sklearn._plot.data.table_report import (
    _truncate_top_k_categories,
)
from skrub import tabular_learner
from skrub.datasets import fetch_employee_salaries


@pytest.fixture
def estimator_report():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    X["gender"] = X["gender"].astype("category")
    X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
    X["cents"] = 100 * y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        tabular_learner("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@pytest.fixture
def cross_validation_report():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    return CrossValidationReport(tabular_learner("regressor"), X=X, y=y)


@pytest.fixture
def display():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    report = CrossValidationReport(tabular_learner("regressor"), X=X, y=y)
    return report.data.analyze()


@pytest.mark.parametrize("col", [None, pd.Series(range(10))])
def test_truncate_top_k_categories_return_as_is(col):
    """Check the behaviour of `_truncate_top_k_categories` when `col` is None or
    numeric where no changes are made."""
    assert _truncate_top_k_categories(col, k=3) is col


@pytest.mark.parametrize("dtype", ["category", "object"])
@pytest.mark.parametrize("other_label", ["other", "xxx"])
def test_truncate_top_k_categories(dtype, other_label):
    """Check the behaviour of `_truncate_top_k_categories` when `col` is a categorical
    column."""
    col = pd.Series(
        ["a", "a", "b", "b", "b", "c", "c", "c", "c", "c", "d", "e", np.nan, np.nan],
        dtype=dtype,
    )
    expected_col = pd.Series(
        [
            "a",
            "a",
            "b",
            "b",
            "b",
            "c",
            "c",
            "c",
            "c",
            "c",
            other_label,
            other_label,
            np.nan,
            np.nan,
        ],
        dtype=dtype,
    )
    truncated_col = _truncate_top_k_categories(col, k=3, other_label=other_label)
    pd.testing.assert_series_equal(truncated_col, expected_col)


@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            dict(kind="dist"),
            "When kind='dist', at least one of x, y must be provided and",
        ),
        (
            dict(kind="dist", hue="current_annual_salary"),
            "When kind='dist', at least one of x, y must be provided and",
        ),
        (
            dict(kind="corr", x="current_annual_salary"),
            "When kind='corr', 'x' argument must be None.",
        ),
        (
            dict(kind="unknown"),
            "'kind' options are 'dist', 'corr', got 'unknown'.",
        ),
    ],
)
def test_table_report_display_plot_error(display, params, err_msg):
    """Check the value that are stored in the display constructor."""
    with pytest.raises(ValueError, match=err_msg):
        display.plot(**params)


def test_table_report_display_constructor(display):
    """Check the value that are stored in the display constructor."""
    assert isinstance(display, Display)

    assert hasattr(display, "summary")
    assert isinstance(display.summary, dict)
    assert list(display.summary.keys()) == [
        "dataframe",
        "dataframe_module",
        "n_rows",
        "n_columns",
        "columns",
        "dataframe_is_empty",
        "plots_skipped",
        "sample_table",
        "n_constant_columns",
        "top_associations",
    ]


def test_table_report_display_frame(cross_validation_report):
    display = cross_validation_report.data.analyze()
    dataset = display.frame(kind="dataset")

    pd.testing.assert_frame_equal(
        dataset,
        pd.concat([cross_validation_report.X, cross_validation_report.y], axis=1),
    )

    associations = display.frame(kind="top-associations")
    pd.testing.assert_frame_equal(
        associations, pd.DataFrame(display.summary["top_associations"])
    )
