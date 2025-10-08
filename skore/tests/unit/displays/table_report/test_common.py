import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from sklearn.datasets import make_regression
from skore import CrossValidationReport, EstimatorReport, train_test_split
from skore._externals._skrub_compat import tabular_pipeline
from skore._sklearn._plot.data.table_report import (
    _compute_contingency_table,
    _resize_categorical_axis,
    _truncate_top_k_categories,
)


@pytest.fixture
def X_y():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(5)])
    y = pd.Series(y, name="Target_")
    return X, y


@pytest.fixture
def estimator_report(X_y):
    X, y = X_y
    split_data = train_test_split(X, y, random_state=0, as_dict=True)
    return EstimatorReport(tabular_pipeline("regressor"), **split_data)


@pytest.fixture
def cross_validation_report(X_y):
    X, y = X_y
    return CrossValidationReport(tabular_pipeline("regressor"), X=X, y=y)


@pytest.fixture(params=["estimator_report", "cross_validation_report"])
def display(request):
    report = request.getfixturevalue(request.param)
    return report.data.analyze()


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
    assert len(display.ax_.get_xticklabels()) == 6
    assert len(display.ax_.get_yticklabels()) == 6
    assert display.ax_.title.get_text() == "Cramer's V Correlation"


def test_repr(display):
    assert repr(display) == "<TableReportDisplay(...)>"


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


def test_html_repr(display, X_y):
    """Check the HTML representation of the `TableReportDisplay`."""
    str_html = display._repr_html_()
    X, _ = X_y
    assert all(col in str_html for col in X.columns)
    assert "<skrub-table-report" in str_html
