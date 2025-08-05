import json

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore._sklearn._plot.data.table_report import (
    _compute_contingency_table,
    _resize_categorical_axis,
    _truncate_top_k_categories,
)
from skrub import tabular_pipeline
from skrub.datasets import fetch_employee_salaries


@pytest.fixture
def display():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    X["gender"] = X["gender"].astype("category")
    X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
    X["timedelta_hired"] = (
        pd.Timestamp.now() - X["date_first_hired"]
    ).dt.to_pytimedelta()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    report = EstimatorReport(
        tabular_pipeline("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    return report.data.analyze()


@pytest.fixture
def estimator_report():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    X["gender"] = X["gender"].astype("category")
    X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
    X["cents"] = 100 * y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        tabular_pipeline("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


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


@pytest.mark.parametrize("is_x_axis", [True, False])
def test_resize_categorical_axis(pyplot, is_x_axis):
    """Check the behaviour of the `_resize_categorical_axis` function."""
    figure, ax = pyplot.subplots(figsize=(10, 10))
    _resize_categorical_axis(
        figure=figure,
        ax=ax,
        n_categories=1,
        is_x_axis=is_x_axis,
        size_per_category=0.5,
    )

    fig_width, fig_height = figure.get_size_inches()
    if is_x_axis:
        assert 0.5 < fig_width < 1.0
        assert 10.0 < fig_height < 13.0
    else:
        assert 0.5 < fig_height < 1.0
        assert 10.0 < fig_width < 13.0


@pytest.mark.parametrize("col", [None, pd.Series(range(10))])
def test_truncate_top_k_categories_return_as_is(col):
    """Check the behaviour of `_truncate_top_k_categories` when `col` is None or
    numeric where no changes are made."""
    assert _truncate_top_k_categories(col, k=3) is col


def test_corr_plot(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    display.plot(kind="corr")
    assert isinstance(display.ax_.collections[0], QuadMesh)
    assert len(display.ax_.get_xticklabels()) == 10
    assert len(display.ax_.get_yticklabels()) == 10
    assert display.ax_.title.get_text() == "Cramer's V Correlation"


def test_repr(display):
    repr = display.__repr__()
    assert repr == "<TableReportDisplay(...)>"


def test_compute_contingency_table_error():
    """Check that we raise an error when the series x and y don't have a name."""
    series = pd.Series(["a", "a", "b", "b", "b", "c", "c", "c", "c", "c", "d", "e"])
    err_msg = "The series x and y must have a name."
    with pytest.raises(ValueError, match=err_msg):
        _compute_contingency_table(x=series, y=series, hue=None, k=1)


@pytest.mark.parametrize("dtype", ["category", "object"])
def test_compute_contingency_table(dtype):
    """Check the behaviour of the `_compute_contingency_table` function."""
    x = pd.Series(
        ["a", "a", "b", "b", "b", "c", "c", "c", "c", "c", "d", "e"],
        name="x",
        dtype=dtype,
    )
    y = pd.Series(
        ["a", "a", "b", "b", "b", "c", "c", "c", "c", "c", "d", "e"],
        name="y",
        dtype=dtype,
    )
    contingency_table = _compute_contingency_table(x, y, hue=None, k=100)
    assert contingency_table.sum().sum() == len(x)
    assert sorted(contingency_table.columns.tolist()) == sorted(x.unique().tolist())
    assert sorted(contingency_table.index.tolist()) == sorted(y.unique().tolist())

    hue = pd.Series(np.ones_like(x) * 2.0)
    contingency_table = _compute_contingency_table(x, y, hue, k=100)
    assert contingency_table.sum().sum() == pytest.approx(x.unique().size * 2)
    assert sorted(contingency_table.columns.tolist()) == sorted(x.unique().tolist())
    assert sorted(contingency_table.index.tolist()) == sorted(y.unique().tolist())

    contingency_table = _compute_contingency_table(x, y, hue=None, k=2)
    assert contingency_table.index.tolist() == ["b", "c"]
    assert contingency_table.columns.tolist() == ["b", "c"]
    assert contingency_table.sum().sum() == 8

    contingency_table = _compute_contingency_table(x, y, hue=hue, k=2)
    assert contingency_table.index.tolist() == ["b", "c"]
    assert contingency_table.columns.tolist() == ["b", "c"]
    assert contingency_table.sum().sum() == 4


def test_json_dump(display):
    json_dict = json.loads(display._to_json())
    assert isinstance(json_dict, dict)
