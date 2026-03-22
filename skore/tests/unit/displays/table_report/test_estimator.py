import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from sklearn.dummy import DummyRegressor
from skrub.datasets import fetch_employee_salaries

from skore import Display, EstimatorReport, train_test_split
from skore._externals._skrub_compat import tabular_pipeline
from skore._sklearn._plot.data.table_report import TableReportDisplay


@pytest.fixture(scope="module")
def estimator_report():
    data = fetch_employee_salaries()
    X, y = data.X, data.y
    X["gender"] = X["gender"].astype("category")
    X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
    X["timedelta_hired"] = (
        pd.Timestamp.now() - X["date_first_hired"]
    ).dt.to_pytimedelta()
    X["cents"] = 100 * y
    split_data = train_test_split(X, y, random_state=0, as_dict=True)
    return EstimatorReport(tabular_pipeline(DummyRegressor()), **split_data)


@pytest.fixture(scope="module")
def display(estimator_report):
    return estimator_report.data.analyze()


@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {"kind": "dist"},
            "When kind='dist', at least one of x, y must be provided and",
        ),
        (
            {"kind": "dist", "hue": "current_annual_salary"},
            "When kind='dist', at least one of x, y must be provided and",
        ),
        (
            {"kind": "corr", "x": "current_annual_salary"},
            "When kind='corr', 'x' argument must be None.",
        ),
        (
            {"kind": "unknown"},
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


@pytest.mark.parametrize(
    "X",
    [
        np.ones((100, 5)),
        pd.DataFrame(
            np.ones((100, 5)), columns=[f"Feature number {i}" for i in range(5)]
        ),
        pd.DataFrame(np.ones((100, 5))),
        pd.DataFrame(np.ones((100, 5)), columns=["a", 1, "c", 3, "e"]),
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        np.ones((100, 1)),
        np.ones(100),
        pd.Series(np.ones(100)),
        pd.DataFrame(np.ones((100, 1)), columns=["Target"]),
        pd.DataFrame(np.ones((100, 1))),
    ],
)
def test_X_y(X, y):
    split_data = train_test_split(X, y, random_state=0, as_dict=True)
    report = EstimatorReport(tabular_pipeline(DummyRegressor()), **split_data)
    display = report.data.analyze()
    assert isinstance(display, TableReportDisplay)


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


def test_categorical_plots_1d(pyplot, display):
    """Check the plot output with categorical data in 1-d."""
    fig = display.plot(x="gender")
    ax = fig.axes[0]
    assert isinstance(fig, Figure)
    assert ax.get_xlabel() == "gender"
    assert [label.get_text() for label in ax.get_xticklabels()] == ["M", "F"]
    labels = ax.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "5000"
    assert ax.get_ylabel() == "Count"
    # orange
    assert ax.containers[0].patches[0].get_facecolor() == (
        1.0,
        0.4980392156862745,
        0.054901960784313725,
        0.75,
    )

    display.set_style(histplot_kwargs={"color": "blue"})
    fig = display.plot(y="gender")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Count"
    assert ax.get_ylabel() == "gender"
    # blue
    assert ax.containers[0].patches[0].get_facecolor() == (0.0, 0.0, 1.0, 0.75)


def test_numeric_plots_1d(pyplot, estimator_report):
    """Check the plot output with numeric data in 1-d."""
    display = estimator_report.data.analyze(data_source="train")
    ## for integers numeric values
    display.set_style(histplot_kwargs={"color": "red"})
    fig = display.plot(x="year_first_hired")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "year_first_hired"
    labels = ax.get_xticklabels()
    assert labels[0].get_text() == "1970"
    assert labels[-1].get_text() == "2010"
    labels = ax.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "500"
    assert ax.get_ylabel() == "Count"
    # red
    assert ax.containers[0].patches[0].get_facecolor() == (1.0, 0.0, 0.0, 0.75)

    display.set_style()
    fig = display.plot(y="year_first_hired")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Count"
    assert ax.get_ylabel() == "year_first_hired"


def test_top_k_categorical_plots_1d(pyplot, display):
    """Check the plot output with categorical data in 1-d and top k categories."""
    fig = display.plot(x="division")
    ax = fig.axes[0]
    assert len(ax.get_xticklabels()) == 20
    fig = display.plot(x="division", top_k_categories=30)
    ax = fig.axes[0]
    assert len(ax.get_xticklabels()) == 30


def test_hue_plots_1d(pyplot, display):
    """Check the plot output with hue in 1-d."""
    fig = display.plot(x="gender", hue="current_annual_salary")
    ax = fig.axes[0]
    assert "BoxPlotContainer" in ax.containers[0].__class__.__name__
    legend_labels = ax.legend_.texts
    assert legend_labels[0].get_text() == "50000"
    assert legend_labels[-1].get_text() == "300000"
    assert ax.legend_.get_title().get_text() == "current_annual_salary"

    fig = display.plot(y="gender", hue="current_annual_salary")
    ax = fig.axes[0]
    assert "BoxPlotContainer" in ax.containers[0].__class__.__name__

    msg = "If 'x' and 'y' are categories, 'hue' must be continuous"
    with pytest.raises(ValueError, match=msg):
        display.plot(x="gender", hue="division", top_k_categories=30)

    fig = display.plot(y="year_first_hired", hue="current_annual_salary")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "current_annual_salary"
    assert ax.get_ylabel() == "year_first_hired"
    assert ax.legend_.get_title().get_text() == "current_annual_salary"


def test_plot_duration_data_1d(pyplot, display):
    """Check the plot output with duration data in 1-d."""
    ## 1D - timedelta as x
    fig = display.plot(x="timedelta_hired")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Years"

    ## 1D - timedelta as y
    fig = display.plot(y="timedelta_hired")
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Years"


def test_plots_2d(pyplot, display):
    """Check the general behaviour of the 2-d plots."""
    # scatter plot
    fig = display.plot(y="current_annual_salary", x="year_first_hired")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "year_first_hired"
    assert ax.get_ylabel() == "current_annual_salary"
    labels = ax.get_xticklabels()
    assert labels[0].get_text() == "1970"
    assert labels[-1].get_text() == "2010"
    labels = ax.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "300000"

    # box plot
    fig = display.plot(x="cents", y="division")
    ax = fig.axes[0]
    assert ax.get_ylabel() == "division"
    assert ax.get_xlabel() == "cents"
    assert len(ax.lines) == 147
    assert ax.get_xticklabels()[-1].get_text() == "3.0"

    # with categories on the x-axis, the tick labels are rotated
    fig = display.plot(x="department_name", y="current_annual_salary")
    ax = fig.axes[0]
    x_tick_labels = ax.get_xticklabels()
    assert all(label.get_rotation() == 45.0 for label in x_tick_labels)

    # heatmap
    fig = display.plot(x="gender", y="division")
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == 20
    assert ax.get_ylabel() == "division"
    assert ax.get_xlabel() == "gender"
    assert isinstance(ax.collections[0], QuadMesh)
    # check that with small numbers, we don't use scientific notation
    annotations = [text.get_text() for text in ax.texts]
    assert not any("e+" in annotation for annotation in annotations)

    # check that we use scientific notation when numbers are too large
    fig = display.plot(x="gender", y="department_name", hue="current_annual_salary")
    ax = fig.axes[0]
    annotations = [text.get_text() for text in ax.texts]
    assert any("e+" in annotation for annotation in annotations)


def test_hue_plots_2d(pyplot, display):
    """Check the plot output with hue parameter in 2-d."""
    fig = display.plot(x="year_first_hired", y="current_annual_salary", hue="division")
    ax = fig.axes[0]
    assert len(ax.legend_.texts) == 21
    assert ax.legend_.get_title().get_text() == "division"

    fig = display.plot(x="year_first_hired", y="gender", hue="division")
    ax = fig.axes[0]
    assert len(ax.lines) == 35
    assert len(ax.legend_.texts) == 21
    assert ax.legend_.get_title().get_text() == "division"

    fig = display.plot(x="gender", y="division", hue="current_annual_salary")
    ax = fig.axes[0]
    assert isinstance(ax.collections[0], QuadMesh)
    colorbar = ax.collections[0].colorbar
    assert colorbar.vmin == pytest.approx(17184.21, rel=1e-1)
    assert colorbar.vmax == pytest.approx(82980.51, rel=1e-1)

    # Can't have categorical hue when x and y are categories
    msg = "If 'x' and 'y' are categories, 'hue' must be continuous."
    with pytest.raises(ValueError, match=msg):
        display.plot(x="gender", y="division", hue="department_name")
