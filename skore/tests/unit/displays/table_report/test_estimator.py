import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from skrub.datasets import fetch_employee_salaries

from skore import Display, EstimatorReport, train_test_split
from skore._externals._skrub_compat import tabular_pipeline
from skore._sklearn._plot.data.table_report import TableReportDisplay


@pytest.fixture
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
    return EstimatorReport(tabular_pipeline("regressor"), **split_data)


@pytest.fixture
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


@pytest.mark.filterwarnings(
    # ignore warning due to the data format used for user testing purpose, raised by
    # `scikit-learn` and `skrub`
    (
        "ignore:"
        "A column-vector y was passed when a 1d array was expected.*:"
        "sklearn.exceptions.DataConversionWarning"
    ),
    "ignore:Some dataframe column names are not strings.*:UserWarning",
    "ignore:Only pandas and polars DataFrames are supported.*:UserWarning",
)
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
    report = EstimatorReport(tabular_pipeline("regressor"), **split_data)
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
    ## 1D - timedelta as x
    display.plot(x="timedelta_hired")
    assert display.ax_.get_xlabel() == "Years"

    ## 1D - timedelta as y
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
