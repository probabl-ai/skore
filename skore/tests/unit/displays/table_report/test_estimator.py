import json

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from sklearn.model_selection import train_test_split
from skore import Display, EstimatorReport
from skore._sklearn._plot.data.table_report import (
    _compute_contingency_table,
    _resize_categorical_axis,
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
        tabular_learner("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
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
def test_error_wrong_param(display, params, err_msg):
    """Check the value that are stored in the display constructor."""
    with pytest.raises(ValueError, match=err_msg):
        display.plot(**params)


def test_constructor(display):
    """Check the value that are stored in the display constructor."""
    assert isinstance(display, Display)

    assert hasattr(display, "summary")
    assert isinstance(display.summary, dict)


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_frame(estimator_report, data_source):
    """Check the behaviour of the `.frame` method."""
    display = estimator_report.data.analyze(data_source=data_source)
    dataset = display.frame(kind="dataset")

    if data_source == "train":
        pd.testing.assert_frame_equal(
            dataset,
            pd.concat([estimator_report.X_train, estimator_report.y_train], axis=1),
        )
    else:
        pd.testing.assert_frame_equal(
            dataset,
            pd.concat([estimator_report.X_test, estimator_report.y_test], axis=1),
        )

    associations = display.frame(kind="top-associations")
    pd.testing.assert_frame_equal(
        associations, pd.DataFrame(display.summary["top_associations"])
    )


def test_categorical_plots_1d(pyplot, estimator_report):
    """Check the plot output with categorical data in 1-d."""
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="gender")
    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert display.ax_.get_xlabel() == "gender"
    assert [label.get_text() for label in display.ax_.get_xticklabels()] == ["M", "F"]
    labels = display.ax_.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "4000"
    assert display.ax_.get_ylabel() == "Count"
    # orange
    assert display.ax_.containers[0].patches[0].get_facecolor() == (
        1.0,
        0.4980392156862745,
        0.054901960784313725,
        0.75,
    )

    display.plot(y="gender", histplot_kwargs={"color": "blue"})
    assert display.ax_.get_xlabel() == "Count"
    assert display.ax_.get_ylabel() == "gender"
    # blue
    assert display.ax_.containers[0].patches[0].get_facecolor() == (0.0, 0.0, 1.0, 0.75)


def test_numeric_plots_1d(pyplot, estimator_report):
    """Check the plot output with numeric data in 1-d."""
    display = estimator_report.data.analyze(data_source="train")
    ## for integers numeric values
    display.plot(x="year_first_hired", histplot_kwargs={"color": "red"})
    assert display.ax_.get_xlabel() == "year_first_hired"
    labels = display.ax_.get_xticklabels()
    assert labels[0].get_text() == "1970"
    assert labels[-1].get_text() == "2010"
    labels = display.ax_.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "500"
    assert display.ax_.get_ylabel() == "Count"
    # red
    assert display.ax_.containers[0].patches[0].get_facecolor() == (1.0, 0.0, 0.0, 0.75)

    display.plot(y="year_first_hired")
    assert display.ax_.get_xlabel() == "Count"
    assert display.ax_.get_ylabel() == "year_first_hired"


def test_top_k_categorical_plots_1d(pyplot, estimator_report):
    """Check the plot output with categorical data in 1-d and top k categories."""
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="division")
    assert len(display.ax_.get_xticklabels()) == 20
    display.plot(x="division", top_k_categories=30)
    assert len(display.ax_.get_xticklabels()) == 30


def test_hue_plots_1d(pyplot, estimator_report):
    """Check the plot output with hue in 1-d."""
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="gender", hue="current_annual_salary")
    assert "BoxPlotContainer" in display.ax_.containers[0].__class__.__name__
    legend_labels = display.ax_.legend_.texts
    assert legend_labels[0].get_text() == "50000"
    assert legend_labels[-1].get_text() == "300000"
    assert display.ax_.legend_.get_title().get_text() == "current_annual_salary"

    display.plot(y="gender", hue="current_annual_salary")
    assert "BoxPlotContainer" in display.ax_.containers[0].__class__.__name__

    msg = "If 'x' and 'y' are categories, 'hue' must be continuous"
    with pytest.raises(ValueError, match=msg):
        display.plot(x="gender", hue="division", top_k_categories=30)

    display.plot(y="year_first_hired", hue="current_annual_salary")
    assert display.ax_.get_xlabel() == "current_annual_salary"
    assert display.ax_.get_ylabel() == "year_first_hired"
    assert display.ax_.legend_.get_title().get_text() == "current_annual_salary"


def test_plot_duration_data_1d(pyplot, display):
    """Check the plot output with duration data in 1-d."""
    ## 1D - timedelta as x
    display.plot(x="timedelta_hired")
    assert display.ax_.get_xlabel() == "Years"

    ## 1D - timedelta as y
    display.plot(y="timedelta_hired")
    assert display.ax_.get_ylabel() == "Years"


def test_plots_2d(pyplot, estimator_report):
    """Check the general behaviour of the 2-d plots."""
    display = estimator_report.data.analyze(data_source="train")
    # scatter plot
    display.plot(y="current_annual_salary", x="year_first_hired")
    assert display.ax_.get_xlabel() == "year_first_hired"
    assert display.ax_.get_ylabel() == "current_annual_salary"
    labels = display.ax_.get_xticklabels()
    assert labels[0].get_text() == "1970"
    assert labels[-1].get_text() == "2010"
    labels = display.ax_.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "300000"

    # box plot
    display.plot(x="cents", y="division")
    assert display.ax_.get_ylabel() == "division"
    assert display.ax_.get_xlabel() == "cents"
    assert len(display.ax_.lines) == 147
    assert display.ax_.get_xticklabels()[-1].get_text() == "3.0"

    # with categories on the x-axis, the tick labels are rotated
    display.plot(x="department_name", y="current_annual_salary")
    x_tick_labels = display.ax_.get_xticklabels()
    assert all(label.get_rotation() == 45.0 for label in x_tick_labels)

    # heatmap
    display.plot(x="gender", y="division")
    assert len(display.ax_.get_yticklabels()) == 19
    assert display.ax_.get_ylabel() == "division"
    assert display.ax_.get_xlabel() == "gender"
    assert isinstance(display.ax_.collections[0], QuadMesh)
    # check that with small numbers, we don't use scientific notation
    annotations = [text.get_text() for text in display.ax_.texts]
    assert not any("e+" in annotation for annotation in annotations)

    # check that we use scientific notation when numbers are too large
    display.plot(x="gender", y="department_name", hue="current_annual_salary")
    annotations = [text.get_text() for text in display.ax_.texts]
    assert any("e+" in annotation for annotation in annotations)


def test_hue_plots_2d(pyplot, estimator_report):
    """Check the plot output with hue parameter in 2-d."""
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="year_first_hired", y="current_annual_salary", hue="division")
    assert len(display.ax_.legend_.texts) == 21
    assert display.ax_.legend_.get_title().get_text() == "division"

    display.plot(x="year_first_hired", y="gender", hue="division")
    assert len(display.ax_.lines) == 35
    assert len(display.ax_.legend_.texts) == 21
    assert display.ax_.legend_.get_title().get_text() == "division"

    display.plot(x="gender", y="division", hue="current_annual_salary")
    assert isinstance(display.ax_.collections[0], QuadMesh)
    colorbar = display.ax_.collections[0].colorbar
    assert colorbar.vmin == pytest.approx(17184.21, rel=1e-1)
    assert colorbar.vmax == pytest.approx(82980.51, rel=1e-1)

    # Can't have categorical hue when x and y are categories
    msg = "If 'x' and 'y' are categories, 'hue' must be continuous."
    with pytest.raises(ValueError, match=msg):
        display.plot(x="gender", y="division", hue="department_name")


def test_corr_plot(pyplot, estimator_report):
    """Check the correlation plot."""
    display = estimator_report.data.analyze(data_source="train")
    display.plot(kind="corr")
    assert isinstance(display.ax_.collections[0], QuadMesh)
    assert len(display.ax_.get_xticklabels()) == 10
    assert len(display.ax_.get_yticklabels()) == 10
    assert display.ax_.title.get_text() == "Cramer's V Correlation"


def test_json_dump(display):
    """Check the JSON serialization of the `TableReportDisplay`."""
    json_dict = json.loads(display._to_json())
    assert "columns" in json_dict
    assert "extract" in json_dict
    assert "top_associations" in json_dict
    assert "n_rows" in json_dict
    assert "n_columns" in json_dict
    assert "cramer_v_correlation" in json_dict
    assert isinstance(json_dict, dict)


def test_repr(display):
    """Check the string representation of the `TableReportDisplay`."""
    repr = display.__repr__()
    assert repr == "<TableReportDisplay(...)>"


def test_html_repr(estimator_report):
    """Check the HTML representation of the `TableReportDisplay`."""
    display = estimator_report.data.analyze(data_source="train")
    str_html = display._repr_html_()
    for col in estimator_report.X_train.columns:
        assert col in str_html

    assert "<skrub-table-report" in str_html
